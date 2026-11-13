"""
Package a trained run for deployment: model weights + config + a searchable gallery.

    python export.py --run runs/flickr8k_cpu                     # gallery = all 8,000 Flickr8k images
    python export.py --run runs/flickr8k_cpu --gallery test      # only the 1,000 unseen test images
    python export.py --run runs/flickr8k_cpu --push_to_hub yourname/mini-clip-flickr8k

Output (default ./model):
    config.json   architecture, metrics, gallery info
    weights.pt    projection heads only (frozen mode, ~3 MB) or all weights in float16 (fine-tune mode)
    gallery/      see gallery.py
"""
import argparse
import json
import os
import shutil

import torch

from data import load_split
from gallery import build_gallery
from model import load_checkpoint, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="model")
    p.add_argument("--name", default="Mini-CLIP")
    p.add_argument("--gallery", choices=["all", "test", "none"], default="all")
    p.add_argument("--thumb_size", type=int, default=320)
    p.add_argument("--examples", nargs="*", default=None, help="example search queries shown in the app")
    p.add_argument("--push_to_hub", default=None, help="Hugging Face model repo id (weights + config only)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.run, device)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    frozen = ckpt["mode"] == "frozen"
    state = model.head_state_dict() if frozen else {k: v.half() for k, v in model.state_dict().items()}
    torch.save(state, os.path.join(args.out, "weights.pt"))

    data_dir = ckpt["train_cfg"]["data"]
    eval_path = os.path.join(args.run, "eval.json")
    config = {
        "name": args.name,
        "mode": ckpt["mode"],
        "model_cfg": ckpt["model_cfg"],
        "dataset": os.path.basename(os.path.normpath(data_dir)),
        "val_metrics": ckpt["val_metrics"],
        "eval": json.load(open(eval_path)) if os.path.exists(eval_path) else None,
        "example_queries": args.examples,
        "gallery": args.gallery,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if args.gallery != "none":
        splits = ["train", "val", "test"] if args.gallery == "all" else ["test"]
        items = [it for s in splits for it in load_split(data_dir, s)]
        build_gallery(model, [p for p, _ in items], os.path.join(args.out, "gallery"),
                      captions=[c for _, c in items], device=device, thumb_size=args.thumb_size)

    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.out) for f in fs) / 1e6
    print(f"Exported to {args.out}/ ({size:.0f} MB)")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push_to_hub, ignore_patterns=["gallery/*"])
        print(f"Pushed weights to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
