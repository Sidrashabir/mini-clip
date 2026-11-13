"""
Export the live demo website into docs/ (served for free by GitHub Pages, always on).

    python export_web.py --run runs/flickr8k_cpu

The website runs the model in the visitor's browser:
  - DistilBERT is loaded from the Hugging Face Hub by transformers.js (the same pretrained
    weights the model was trained with, converted to ONNX)
  - the trained text projection head runs in plain JavaScript (weights in docs/model.bin)
  - photo and caption embeddings are computed here, once, and shipped with the page
  - thumbnails are packed into a few image sheets (atlases) to keep the repository small

Output: docs/index.html (already in the repo), docs/data.json, docs/model.bin, docs/atlas_*.jpg
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image

from data import encode_images, encode_texts, flatten_captions, load_split, open_rgb
from gallery import name_clusters, tsne_map
from model import load_checkpoint

# Browser-ready (ONNX) copies of the text encoders, used by transformers.js
BROWSER_TEXT_MODELS = {"distilbert-base-uncased": "Xenova/distilbert-base-uncased"}
HEAD_ORDER = ["text_head.proj.weight", "text_head.proj.bias", "text_head.mlp.1.weight", "text_head.mlp.1.bias",
              "text_head.norm.weight", "text_head.norm.bias"]
CHECK_TEXTS = ["a dog runs across the grass", "two children playing on the beach", "a man climbs a rock wall"]


def square_tile(path, size):
    im = open_rgb(path)
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.LANCZOS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="docs")
    p.add_argument("--split", default="test", help="photos to show: test (unseen, default), val, train")
    p.add_argument("--limit", type=int, default=1000, help="maximum number of photos")
    p.add_argument("--tile", type=int, default=224, help="thumbnail size in pixels")
    p.add_argument("--per_sheet", type=int, default=100, help="thumbnails per image sheet")
    p.add_argument("--examples", nargs="*", default=None, help="example searches shown on the page")
    p.add_argument("--repo", default="https://github.com/Sidrashabir/mini-clip", help="link shown on the page")
    p.add_argument("--allow_test_model", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    model, ckpt = load_checkpoint(args.run, "cpu")
    cfg = ckpt["model_cfg"]
    tcfg = ckpt["train_cfg"]
    browser_model = BROWSER_TEXT_MODELS.get(cfg["text_backbone"])
    if not args.allow_test_model:
        if browser_model is None:
            raise SystemExit(f"The browser demo supports the text encoders {list(BROWSER_TEXT_MODELS)}, "
                             f"not {cfg['text_backbone']}.")
        if ckpt["mode"] != "frozen" and tcfg.get("unfreeze_text", 0) > 0:
            raise SystemExit("This run fine-tuned the text encoder, so the stock browser copy would not match. "
                             "Use a run trained with the default frozen mode.")
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if f.startswith("atlas_") or f in ("data.json", "model.bin"):
            os.remove(os.path.join(args.out, f))

    items = load_split(tcfg["data"], args.split)[: args.limit]
    paths = [pth for pth, _ in items]
    captions, owner = flatten_captions(items)
    print(f"{len(items)} photos, {len(captions)} captions")

    with torch.no_grad():
        feats = encode_images(model.image_encoder, paths, model.image_encoder.transform(False), "cpu",
                              desc="photo embeddings")
        img_emb = model.embed_image_features(feats).numpy().astype(np.float32)
        cap_emb = model.embed_text_features(encode_texts(model.text_encoder, captions, "cpu")).numpy()
        check_emb = model.embed_text_features(encode_texts(model.text_encoder, CHECK_TEXTS, "cpu")).numpy()

    # ---- binary: text head weights, photo embeddings, caption embeddings (little-endian)
    state = model.state_dict()
    sections, blobs, offset = {}, [], 0
    # head weights in float32 for exactness; embeddings in float16 to halve the download
    parts = [(k, state[k].numpy(), "<f4") for k in HEAD_ORDER] + [("image_emb", img_emb, "<f2"),
                                                                 ("caption_emb", cap_emb, "<f2")]
    for name, arr, dtype in parts:
        a = np.ascontiguousarray(arr, dtype=dtype)
        sections[name] = {"offset": offset, "shape": list(a.shape), "dtype": "f32" if dtype == "<f4" else "f16"}
        blobs.append(a.tobytes())
        offset += a.nbytes
    with open(os.path.join(args.out, "model.bin"), "wb") as f:
        f.write(b"".join(blobs))

    # ---- image sheets
    cols = int(math.ceil(math.sqrt(args.per_sheet)))
    n_sheets = int(math.ceil(len(paths) / args.per_sheet))
    print(f"Packing thumbnails into {n_sheets} image sheets ...")
    for s in range(n_sheets):
        chunk = paths[s * args.per_sheet:(s + 1) * args.per_sheet]
        rows = int(math.ceil(len(chunk) / cols))
        sheet = Image.new("RGB", (cols * args.tile, rows * args.tile), (240, 240, 240))
        for i, pth in enumerate(chunk):
            sheet.paste(square_tile(pth, args.tile), ((i % cols) * args.tile, (i // cols) * args.tile))
        sheet.save(os.path.join(args.out, f"atlas_{s:02d}.jpg"), "JPEG", quality=82, optimize=True, progressive=True)

    # ---- embedding map and named clusters
    print("Computing the embedding map ...")
    idx, xy = tsne_map(img_emb, max_points=len(img_emb))
    clusters, names = name_clusters(img_emb, [c for _, c in items])

    eval_path = os.path.join(args.run, "eval.json")
    data = {
        "name": "Mini-CLIP",
        "repo": args.repo,
        "browser_text_model": browser_model,
        "model_cfg": {k: cfg[k] for k in ("image_backbone", "text_backbone", "embed_dim", "image_dim", "text_dim")},
        "temperature": round(1 / model.scale().item(), 5),
        "logit_scale": model.scale().item(),
        "dataset": os.path.basename(os.path.normpath(tcfg["data"])),
        "split": args.split,
        "val_metrics": ckpt.get("val_metrics"),
        "eval": json.load(open(eval_path)) if os.path.exists(eval_path) else None,
        "examples": args.examples,
        "sections": sections,
        "items": [{"captions": [c.rstrip(" .") for c in caps]} for _, caps in items],
        "caption_owner": owner.tolist(),
        "sheets": {"count": n_sheets, "per_sheet": args.per_sheet, "cols": cols, "tile": args.tile},
        "map": {"index": idx, "xy": xy, "cluster": [clusters[i] for i in idx], "names": names},
        "checks": [{"text": t, "embedding": [round(float(v), 6) for v in e]} for t, e in zip(CHECK_TEXTS, check_emb)],
    }
    with open(os.path.join(args.out, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 1e6
    print(f"Website exported to {args.out}/ ({size:.1f} MB). Preview: python -m http.server --directory {args.out}")


if __name__ == "__main__":
    main()
