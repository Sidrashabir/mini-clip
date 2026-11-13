"""
Prepare a dataset in the standard format (see data.py).

    python prepare.py                       # Flickr8k: 8,000 photos x 5 captions (~1.1 GB download)
    python prepare.py --dataset synthetic   # small generated shapes dataset, for a 1-minute test run
    python prepare.py --dataset csv --csv my_captions.csv --images my_photos/ --name mydata
        (your own data: a CSV with columns image,caption; image paths relative to --images)
"""
import argparse
import csv
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict

FLICKR_IMAGES = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
FLICKR_TEXT = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"


def download(url, path):
    if os.path.exists(path):
        return
    print(f"Downloading {os.path.basename(path)} ...")

    def hook(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, path)
    print()


def write_csv(out_dir, rows):
    with open(os.path.join(out_dir, "captions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption", "split"])
        w.writerows(rows)
    counts = defaultdict(set)
    for img, _, split in rows:
        counts[split].add(img)
    print({s: f"{len(v)} images" for s, v in counts.items()}, f"| {len(rows)} captions")
    print(f"Saved to {out_dir}/")


def prepare_flickr8k(out_dir, keep_zips):
    raw = os.path.join("data", "_downloads")
    os.makedirs(raw, exist_ok=True)
    img_zip, txt_zip = os.path.join(raw, "Flickr8k_Dataset.zip"), os.path.join(raw, "Flickr8k_text.zip")
    download(FLICKR_TEXT, txt_zip)
    download(FLICKR_IMAGES, img_zip)

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    print("Extracting images ...")
    with zipfile.ZipFile(img_zip) as z:
        for m in z.infolist():
            name = os.path.basename(m.filename)
            if name.lower().endswith(".jpg") and "__MACOSX" not in m.filename:
                target = os.path.join(img_dir, name)
                # (re-)extract if missing or incomplete, e.g. after an interrupted run
                if not os.path.exists(target) or os.path.getsize(target) != m.file_size:
                    with z.open(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    with zipfile.ZipFile(txt_zip) as z:
        def read(name):
            member = next(n for n in z.namelist() if n.endswith(name) and "__MACOSX" not in n)
            return z.read(member).decode("utf-8").splitlines()
        tokens = read("Flickr8k.token.txt")
        split_of = {}
        for split, fname in [("train", "Flickr_8k.trainImages.txt"), ("val", "Flickr_8k.devImages.txt"),
                             ("test", "Flickr_8k.testImages.txt")]:
            for line in read(fname):
                if line.strip():
                    split_of[line.strip()] = split

    available = set(os.listdir(img_dir))
    rows = []
    for line in tokens:
        if "\t" not in line:
            continue
        key, caption = line.split("\t", 1)
        img = key.split("#")[0]
        if img in available and img in split_of:   # official 6000 / 1000 / 1000 split
            rows.append((img, caption.strip(), split_of[img]))
    write_csv(out_dir, rows)
    if not keep_zips:
        shutil.rmtree(raw, ignore_errors=True)


def prepare_synthetic(out_dir, n_images=1200, seed=0):
    """Coloured shapes on backgrounds with templated captions. Learnable in a minute on CPU."""
    from PIL import Image, ImageDraw
    rnd = random.Random(seed)
    colors = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220), "yellow": (240, 210, 40),
              "purple": (140, 60, 180), "orange": (245, 140, 30), "white": (245, 245, 245), "black": (20, 20, 20)}
    backgrounds = {"grey": (128, 128, 128), "light blue": (170, 210, 240), "beige": (235, 220, 190)}
    shapes = ["circle", "square", "triangle"]
    positions = {"on the left": 0.28, "in the middle": 0.5, "on the right": 0.72}
    sizes = {"small": 0.14, "large": 0.3}
    templates = ["a {size} {color} {shape} {pos}", "a {color} {shape} {pos} on a {bg} background",
                 "{size} {color} {shape} against {bg}", "there is a {color} {shape} {pos}",
                 "an image of a {size} {shape} that is {color}"]
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i in range(n_images):
        c, s, p, z, b = (rnd.choice(list(colors)), rnd.choice(shapes), rnd.choice(list(positions)),
                         rnd.choice(list(sizes)), rnd.choice(list(backgrounds)))
        if c == "white" and b == "beige":
            c = "black"
        im = Image.new("RGB", (128, 128), backgrounds[b])
        d = ImageDraw.Draw(im)
        cx, cy, r = positions[p] * 128, 64 + rnd.uniform(-12, 12), sizes[z] * 128
        box = [cx - r, cy - r, cx + r, cy + r]
        if s == "circle":
            d.ellipse(box, fill=colors[c])
        elif s == "square":
            d.rectangle(box, fill=colors[c])
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=colors[c])
        name = f"shape_{i:05d}.png"
        im.save(os.path.join(img_dir, name))
        split = "train" if i < n_images * 0.8 else "val" if i < n_images * 0.9 else "test"
        for t in rnd.sample(templates, 3):
            rows.append((name, t.format(size=z, color=c, shape=s, pos=p, bg=b), split))
    write_csv(out_dir, rows)


def prepare_csv(out_dir, csv_path, images_root, val_frac=0.1, test_frac=0.1, seed=0):
    by_image = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_image[row["image"]].append(row["caption"])
    images = sorted(by_image)
    random.Random(seed).shuffle(images)
    n_test, n_val = int(len(images) * test_frac), int(len(images) * val_frac)
    split_of = {img: "test" if i < n_test else "val" if i < n_test + n_val else "train"
                for i, img in enumerate(images)}
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for img, caps in by_image.items():
        src = os.path.join(images_root, img)
        if not os.path.exists(src):
            print(f"  skipping missing image {src}")
            continue
        flat = img.replace("/", "_").replace("\\", "_")
        shutil.copy(src, os.path.join(img_dir, flat))
        rows.extend((flat, c, split_of[img]) for c in caps)
    write_csv(out_dir, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["flickr8k", "synthetic", "csv"], default="flickr8k")
    p.add_argument("--name", default=None, help="output folder name under data/")
    p.add_argument("--csv", help="(csv mode) file with columns image,caption")
    p.add_argument("--images", help="(csv mode) folder the image paths are relative to")
    p.add_argument("--keep_zips", action="store_true")
    args = p.parse_args()

    out_dir = os.path.join("data", args.name or args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    if args.dataset == "flickr8k":
        prepare_flickr8k(out_dir, args.keep_zips)
    elif args.dataset == "synthetic":
        prepare_synthetic(out_dir)
    else:
        if not (args.csv and args.images):
            p.error("--dataset csv needs --csv and --images")
        prepare_csv(out_dir, args.csv, args.images)


if __name__ == "__main__":
    main()
