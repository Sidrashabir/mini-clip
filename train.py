"""
Train Mini-CLIP.

    python train.py                            # preset "cpu": frozen backbones, cached features (laptop)
    python train.py --preset cpu --aug_views 2 # + 2 augmented views per image (better, slower first run)
    python train.py --preset gpu               # fine-tune top layers of both backbones (needs a GPU)
    python train.py --preset test              # tiny offline backbones on data/synthetic (1-minute check)

    Any setting can be overridden, e.g.  --image_backbone vit_small_patch16_224 --embed_dim 512
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ImageCaptionDataset, build_or_load, flatten_captions, load_split
from encoders import TEST_BACKBONE
from model import CLIPConfig, MiniCLIP, pick_device, retrieval_metrics

PRESETS = {
    "cpu": dict(mode="frozen", data="data/flickr8k", image_backbone="resnet50",
                text_backbone="distilbert-base-uncased", epochs=40, batch_size=256, head_lr=1e-3, patience=6),
    "gpu": dict(mode="finetune", data="data/flickr8k", image_backbone="resnet50",
                text_backbone="distilbert-base-uncased", epochs=12, batch_size=64, head_lr=1e-3,
                backbone_lr=2e-5, unfreeze_image=1, unfreeze_text=2, patience=4),
    "test": dict(mode="finetune", data="data/synthetic", image_backbone=TEST_BACKBONE,
                 text_backbone=TEST_BACKBONE, epochs=25, batch_size=64, head_lr=3e-3, backbone_lr=3e-3,
                 unfreeze_image=99, unfreeze_text=99, patience=25),
}
DEFAULTS = dict(embed_dim=256, head_dropout=0.1, weight_decay=0.05, warmup_frac=0.05, aug_views=0,
                backbone_lr=1e-5, unfreeze_image=0, unfreeze_text=0, seed=42, device="auto",
                num_workers=0, out_dir=None, cache_dir="cache", amp=True)


def get_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="cpu")
    keys = set(DEFAULTS) | {k for v in PRESETS.values() for k in v}
    for k in sorted(keys):
        sample = DEFAULTS.get(k, PRESETS["cpu"].get(k, PRESETS["gpu"].get(k)))
        if isinstance(sample, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            p.add_argument(f"--{k}", type=type(sample) if sample is not None else str, default=None)
    args = vars(p.parse_args())
    cfg = {**DEFAULTS, **PRESETS[args["preset"]]}
    cfg.update({k: v for k, v in args.items() if v is not None})
    if cfg["out_dir"] is None:
        cfg["out_dir"] = os.path.join("runs", f"{os.path.basename(os.path.normpath(cfg['data']))}_{cfg['preset']}")
    return cfg


def cosine_lr(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


@torch.no_grad()
def embed_split_finetune(model, items, device, batch_size=128, num_workers=0):
    model.eval()
    tf = model.image_encoder.transform(train=False)
    from data import encode_images
    img_feats = encode_images(model.image_encoder, [p for p, _ in items], tf, device, batch_size,
                              num_workers, desc="val images")
    img = model.embed_image_features(img_feats.to(device))
    captions, owner = flatten_captions(items)
    txt = []
    for i in range(0, len(captions), 256):
        tok = model.text_encoder.tokenize(captions[i:i + 256])
        txt.append(model.encode_text(tok["input_ids"].to(device), tok["attention_mask"].to(device)))
    return img, torch.cat(txt), owner


def main():
    cfg = get_cfg()
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    frozen = cfg["mode"] == "frozen"
    print(f"Mode: {cfg['mode']} | device: {device} | {cfg['image_backbone']} + {cfg['text_backbone']}")

    model_cfg = CLIPConfig(image_backbone=cfg["image_backbone"], text_backbone=cfg["text_backbone"],
                           embed_dim=cfg["embed_dim"], head_dropout=cfg["head_dropout"])
    model = MiniCLIP(model_cfg).to(device)

    # ------------------------------------------------------------------ data
    if frozen:
        train_f = build_or_load(model, cfg["data"], "train", device, cfg["cache_dir"], cfg["aug_views"],
                                num_workers=cfg["num_workers"])
        val_f = build_or_load(model, cfg["data"], "val", device, cfg["cache_dir"], 0,
                              num_workers=cfg["num_workers"])
        # backbones are no longer needed: free their memory, train heads only
        model.image_encoder = model.text_encoder = None
        n_train = train_f["image"].size(0)
        caps_of = [[] for _ in range(n_train)]
        for j, i in enumerate(train_f["owner"].tolist()):
            caps_of[i].append(j)
        steps_per_epoch = math.ceil(n_train / cfg["batch_size"])
    else:
        train_items, val_items = load_split(cfg["data"], "train"), load_split(cfg["data"], "val")
        model.image_encoder.unfreeze_top(cfg["unfreeze_image"])
        model.text_encoder.unfreeze_top(cfg["unfreeze_text"])
        loader = DataLoader(ImageCaptionDataset(train_items, model.image_encoder.transform(train=True)),
                            batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                            num_workers=cfg["num_workers"], pin_memory=device == "cuda")
        steps_per_epoch = len(loader)

    # ------------------------------------------------------------- optimiser
    head_params = [p for n, p in model.named_parameters() if not n.startswith(("image_encoder", "text_encoder"))]
    groups = [{"params": head_params, "lr": cfg["head_lr"]}]
    if not frozen:
        bb = [p for n, p in model.named_parameters()
              if n.startswith(("image_encoder", "text_encoder")) and p.requires_grad]
        if bb:
            groups.append({"params": bb, "lr": cfg["backbone_lr"]})
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    total = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(total * cfg["warmup_frac"]))
    use_amp = cfg["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable / 1e6:.2f}M | {steps_per_epoch} steps/epoch")

    log_path = os.path.join(cfg["out_dir"], "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_mean_recall", "val_t2i_R@1",
                                "val_t2i_R@5", "val_i2t_R@1", "temperature", "elapsed_s"])

    best, bad_epochs, step, t0 = -1.0, 0, 0, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        if frozen:
            order = torch.randperm(n_train)
            batches = [order[i:i + cfg["batch_size"]] for i in range(0, n_train, cfg["batch_size"])]
        else:
            batches = loader
        for batch in batches:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * cosine_lr(step, total, warmup)
            if frozen:
                idx = batch
                views = torch.randint(0, train_f["image"].size(1), (len(idx),))
                img_feats = train_f["image"][idx, views].to(device)
                cap_idx = torch.tensor([random.choice(caps_of[i]) for i in idx.tolist()])
                txt_feats = train_f["text"][cap_idx].to(device)
                loss = model(image_feats=img_feats, text_feats=txt_feats)
            else:
                pixels, caps, img_ids = batch
                tok = model.text_encoder.tokenize(caps)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(pixels=pixels.to(device), input_ids=tok["input_ids"].to(device),
                                 attention_mask=tok["attention_mask"].to(device), labels=img_ids.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1

        # ------------------------------------------------------------ validate
        model.eval()
        with torch.no_grad():
            if frozen:
                img = model.embed_image_features(val_f["image"][:, 0].to(device))
                txt = model.embed_text_features(val_f["text"].to(device))
                owner = val_f["owner"]
            else:
                img, txt, owner = embed_split_finetune(model, val_items, device, num_workers=cfg["num_workers"])
            # validation loss on (image, first caption) pairs
            first = torch.tensor([owner.tolist().index(i) for i in range(img.size(0))])
            val_loss = sum(
                torch.nn.functional.cross_entropy(model.scale() * img[c] @ txt[first[c]].t(),
                                                  torch.arange(len(c), device=device)).item()
                for c in torch.arange(img.size(0)).split(cfg["batch_size"])
            ) / math.ceil(img.size(0) / cfg["batch_size"])
            m = retrieval_metrics(img, txt, owner)

        temp = 1 / model.scale().item()
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d} | train loss {np.mean(losses):.4f} | val loss {val_loss:.4f} | "
              f"mean recall {m['mean_recall']:5.1f} | T->I R@1 {m['t2i_R@1']:5.1f} R@5 {m['t2i_R@5']:5.1f} | "
              f"temp {temp:.4f} | {elapsed:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{np.mean(losses):.4f}", f"{val_loss:.4f}", m["mean_recall"],
                                    m["t2i_R@1"], m["t2i_R@5"], m["i2t_R@1"], f"{temp:.4f}", f"{elapsed:.1f}"])

        if m["mean_recall"] > best:
            best, bad_epochs = m["mean_recall"], 0
            torch.save({"mode": cfg["mode"], "model_cfg": model.config_dict(),
                        "state": model.head_state_dict() if frozen else model.state_dict(),
                        "epoch": epoch, "val_metrics": m, "train_cfg": cfg},
                       os.path.join(cfg["out_dir"], "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"Early stopping: no improvement for {cfg['patience']} epochs")
                break

    with open(os.path.join(cfg["out_dir"], "train_config.json"), "w") as f:
        json.dump({"train_cfg": cfg, "best_val_mean_recall": best}, f, indent=2)
    print(f"Done. Best validation mean recall {best:.2f}. Checkpoint: {cfg['out_dir']}/best.pt")


if __name__ == "__main__":
    main()
