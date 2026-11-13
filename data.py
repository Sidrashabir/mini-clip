"""
Dataset format used by every script:

    data/<name>/images/...          image files
    data/<name>/captions.csv        columns: image, caption, split   (split = train / val / test)

One image can have several captions (Flickr8k has 5 per image).
"""
import csv
import os
import random
from collections import OrderedDict

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # don't crash on a slightly damaged file


def load_split(data_dir, split):
    """Returns a list of (image_path, [captions]) for one split, in a stable order."""
    groups = OrderedDict()
    with open(os.path.join(data_dir, "captions.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                groups.setdefault(row["image"], []).append(row["caption"].strip())
    return [(os.path.join(data_dir, "images", img), caps) for img, caps in groups.items()]


def flatten_captions(items):
    """All captions of a split, plus the index of the image each caption belongs to."""
    captions, owner = [], []
    for i, (_, caps) in enumerate(items):
        captions.extend(caps)
        owner.extend([i] * len(caps))
    return captions, torch.tensor(owner)


def open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


class ImageCaptionDataset(Dataset):
    """One sample per image; a random caption of that image is drawn each time (fine-tuning mode)."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, caps = self.items[i]
        return self.transform(open_rgb(path)), random.choice(caps), i


class ImageOnlyDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(open_rgb(self.paths[i]))


# --------------------------------------------------------------------------- feature caching
# Pre-compute backbone features once ("frozen" training mode). Running ResNet-50 and DistilBERT
# over every image and caption is the slow part; caching the result means each training epoch
# only runs the small projection heads, so training takes minutes on a laptop CPU.


def cache_path(cache_dir, data_dir, image_backbone, text_backbone, split, aug_views):
    name = f"{os.path.basename(os.path.normpath(data_dir))}__{image_backbone}__{text_backbone.replace('/', '-')}"
    return os.path.join(cache_dir, name, f"{split}_aug{aug_views}.pt")


@torch.no_grad()
def encode_images(image_encoder, paths, transform, device, batch_size=64, num_workers=0, desc="images"):
    loader = DataLoader(ImageOnlyDataset(paths, transform), batch_size=batch_size, num_workers=num_workers)
    out, done = [], 0
    for x in loader:
        out.append(image_encoder(x.to(device)).float().cpu())
        done += len(x)
        print(f"\r  {desc}: {done}/{len(paths)}", end="", flush=True)
    print()
    return torch.cat(out)


@torch.no_grad()
def encode_texts(text_encoder, texts, device, batch_size=256):
    out = []
    for i in range(0, len(texts), batch_size):
        tok = text_encoder.tokenize(texts[i:i + batch_size])
        out.append(text_encoder(tok["input_ids"].to(device), tok["attention_mask"].to(device)).float().cpu())
        print(f"\r  captions: {min(i + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    return torch.cat(out)


def build_or_load(model, data_dir, split, device, cache_dir="cache", aug_views=0, batch_size=64,
                  num_workers=0):
    """
    Returns {"image": (N, V, D_img), "text": (M, D_txt), "owner": (M,), "paths": [...], "captions": [...]}.
    V = 1 + aug_views: view 0 is the plain centre crop; extra views use random augmentation,
    which acts as regularisation when the backbone is frozen.
    """
    path = cache_path(cache_dir, data_dir, model.cfg.image_backbone, model.cfg.text_backbone, split, aug_views)
    if os.path.exists(path):
        return torch.load(path, weights_only=False)

    print(f"Extracting {split} features (one-time, cached to {path})")
    items = load_split(data_dir, split)
    paths = [p for p, _ in items]
    captions, owner = flatten_captions(items)
    enc_img, enc_txt = model.image_encoder.eval(), model.text_encoder.eval()

    views = [encode_images(enc_img, paths, enc_img.transform(train=False), device, batch_size, num_workers)]
    for v in range(aug_views):
        views.append(encode_images(enc_img, paths, enc_img.transform(train=True), device, batch_size,
                                   num_workers, desc=f"augmented view {v + 1}/{aug_views}"))
    feats = {"image": torch.stack(views, 1), "text": encode_texts(enc_txt, captions, device),
             "owner": owner, "paths": paths, "captions": captions}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(feats, path)
    return feats
