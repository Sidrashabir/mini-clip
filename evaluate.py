"""
Evaluate on the held-out TEST split (1,000 images / 5,000 captions for Flickr8k).

    python evaluate.py --run runs/flickr8k_cpu
    python evaluate.py --run runs/flickr8k_cpu --compare_clip openai/clip-vit-base-patch32

Reports Recall@1/5/10 for text->image and image->text search, and median rank.
--compare_clip also evaluates OpenAI's CLIP zero-shot (trained on 400M image-text pairs)
on the same test set, as a reference point.
"""
import argparse
import json
import os

import torch

from data import encode_images, encode_texts, flatten_captions, load_split
from model import load_checkpoint, pick_device, retrieval_metrics


@torch.no_grad()
def embed_test_set(model, items, device, num_workers=0):
    img_feats = encode_images(model.image_encoder, [p for p, _ in items], model.image_encoder.transform(False),
                              device, num_workers=num_workers, desc="test images")
    captions, owner = flatten_captions(items)
    txt_feats = encode_texts(model.text_encoder, captions, device)
    return (model.embed_image_features(img_feats.to(device)),
            model.embed_text_features(txt_feats.to(device)), owner)


@torch.no_grad()
def clip_image_embed(model, pixel_values):
    """Projected, normalised CLIP image embeddings (stable across transformers versions)."""
    pooled = model.vision_model(pixel_values=pixel_values).pooler_output
    return torch.nn.functional.normalize(model.visual_projection(pooled), dim=-1)


@torch.no_grad()
def clip_text_embed(model, input_ids, attention_mask):
    pooled = model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    return torch.nn.functional.normalize(model.text_projection(pooled), dim=-1)


@torch.no_grad()
def evaluate_openai_clip(name, items, device):
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    paths = [p for p, _ in items]
    captions, owner = flatten_captions(items)
    img, txt = [], []
    for i in range(0, len(paths), 32):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 32]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(device)
        img.append(clip_image_embed(model, px))
        print(f"\r  CLIP images: {min(i + 32, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    for i in range(0, len(captions), 256):
        tok = proc(text=captions[i:i + 256], return_tensors="pt", padding=True, truncation=True).to(device)
        txt.append(clip_text_embed(model, tok["input_ids"], tok["attention_mask"]))
    params = sum(p.numel() for p in model.parameters())
    return retrieval_metrics(torch.cat(img), torch.cat(txt), owner), params


def print_table(rows):
    cols = ["t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_R@1", "i2t_R@5", "i2t_R@10", "mean_recall"]
    print(f"\n{'model':44s}" + "".join(f"{c:>10s}" for c in ["T→I R@1", "R@5", "R@10", "I→T R@1", "R@5",
                                                                  "R@10", "mean"]))
    for r in rows:
        print(f"{r['model']:44s}" + "".join(f"{r['metrics'][c]:>10.1f}" for c in cols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--compare_clip", default=None, help="e.g. openai/clip-vit-base-patch32")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.run, device)
    items = load_split(args.data or ckpt["train_cfg"]["data"], args.split)
    print(f"{args.split} set: {len(items)} images, {sum(len(c) for _, c in items)} captions")

    img, txt, owner = embed_test_set(model, items, device)
    trainable = sum(v.numel() for v in model.head_state_dict().values()) if ckpt["mode"] == "frozen" else \
        sum(p.numel() for p in model.parameters())
    cfg = ckpt["model_cfg"]
    rows = [{"model": f"Mini-CLIP ({cfg['image_backbone']} + {cfg['text_backbone'].split('/')[-1]})",
             "trained_params_M": round(trainable / 1e6, 2), "training_pairs": "6k images",
             "metrics": retrieval_metrics(img, txt, owner)}]
    if args.compare_clip:
        m, params = evaluate_openai_clip(args.compare_clip, items, device)
        rows.append({"model": f"{args.compare_clip} (zero-shot)", "trained_params_M": round(params / 1e6, 1),
                     "training_pairs": "400M", "metrics": m})

    print_table(rows)
    r = rows[0]["metrics"]
    print(f"\nMedian rank: text→image {r['t2i_median_rank']:.0f}, image→text {r['i2t_median_rank']:.0f} "
          f"(out of {len(items)} images)")
    with open(os.path.join(args.run, "eval.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
