"""
A searchable gallery: thumbnails + pre-computed embeddings + a 2D map of the embedding space.

    gallery/
        images/00000.jpg ...   thumbnails shown in the app
        items.json             file name and captions (if any) for each image
        image_emb.npy          (N, D) image embeddings, used for search
        caption_emb.npy        (M, D) caption embeddings, used for image -> caption retrieval
        captions.json          caption text and the image index it belongs to
        map.json               2D t-SNE coordinates for the embedding map

Make your own photos searchable (images are found recursively; nothing leaves your computer):

    python gallery.py --images "C:/Users/you/Pictures/Holiday" --out my_gallery
    python server.py --gallery my_gallery
"""
import argparse
import json
import os

import numpy as np
import torch

from data import encode_images, encode_texts, open_rgb


def make_thumbnail(src, dst, max_side):
    im = open_rgb(src)
    im.thumbnail((max_side, max_side))
    im.save(dst, "JPEG", quality=85, optimize=True)


def tsne_map(emb, max_points=2000, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(emb), min(max_points, len(emb)), replace=False))
    x = emb[idx]
    if x.shape[1] > 50 and len(x) > 50:
        x = PCA(50, random_state=seed).fit_transform(x)
    xy = TSNE(2, perplexity=min(30, max(5, len(x) // 4)), init="pca", random_state=seed).fit_transform(x)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)
    return idx.tolist(), xy.round(4).tolist()


STOPWORDS = set("""a an the of in on at to and with is are while his her their its for from by into over
two three one some other another man woman men women person people there this that has have who near
up down out off front next""".split())


def name_clusters(emb, captions=None, k=None, seed=0):
    """K-means over image embeddings; each cluster is named by its most frequent caption words."""
    from collections import Counter

    from sklearn.cluster import KMeans
    k = k or max(2, min(10, len(emb) // 50))
    labels = KMeans(k, n_init=5, random_state=seed).fit_predict(emb).tolist()
    names = []
    if captions:
        def words(i):
            return {w.strip(".,!?'\"").lower() for cap in captions[i] for w in cap.split()}
        overall = Counter(w for i in range(len(labels)) for w in words(i))
        n_total = len(labels)
    for c in range(k):
        members = [i for i, lab in enumerate(labels) if lab == c]
        if captions and members:
            inside = Counter(w for i in members for w in words(i))
            # distinctive words: much more common inside this cluster than in the whole gallery
            score = {w: n / len(members) - overall[w] / n_total for w, n in inside.items()
                     if w not in STOPWORDS and len(w) > 2 and n >= 3}
            top = sorted(score, key=score.get, reverse=True)[:3]
            names.append(" · ".join(top) if top else f"group {c + 1}")
        else:
            names.append(f"group {c + 1}")
    return labels, names


@torch.no_grad()
def build_gallery(model, image_paths, out_dir, captions=None, device="cpu", thumb_size=320, num_workers=0):
    """captions: optional list (same length as image_paths) of caption lists."""
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    items = []
    print(f"Writing {len(image_paths)} thumbnails ...")
    for i, p in enumerate(image_paths):
        name = f"{i:05d}.jpg"
        make_thumbnail(p, os.path.join(img_dir, name), thumb_size)
        items.append({"file": name, "source": os.path.basename(p), "captions": captions[i] if captions else []})

    model.eval()
    feats = encode_images(model.image_encoder, image_paths, model.image_encoder.transform(False), device,
                          num_workers=num_workers, desc="gallery embeddings")
    image_emb = model.embed_image_features(feats.to(device)).cpu().numpy().astype(np.float32)
    np.save(os.path.join(out_dir, "image_emb.npy"), image_emb)
    with open(os.path.join(out_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)

    if captions:
        flat = [(c, i) for i, caps in enumerate(captions) for c in caps]
        txt = encode_texts(model.text_encoder, [c for c, _ in flat], device)
        cap_emb = model.embed_text_features(txt.to(device)).cpu().numpy().astype(np.float16)
        np.save(os.path.join(out_dir, "caption_emb.npy"), cap_emb)
        with open(os.path.join(out_dir, "captions.json"), "w", encoding="utf-8") as f:
            json.dump([{"text": c, "image": i} for c, i in flat], f, ensure_ascii=False)

    print("Computing embedding map (t-SNE + clusters) ...")
    idx, xy = tsne_map(image_emb)
    clusters, names = name_clusters(image_emb, captions)
    with open(os.path.join(out_dir, "map.json"), "w") as f:
        json.dump({"index": idx, "xy": xy, "cluster": [clusters[i] for i in idx], "cluster_names": names,
                   "all_clusters": clusters}, f)
    print(f"Gallery ready: {out_dir} ({len(items)} images)")


EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main():
    p = argparse.ArgumentParser(description="Build a searchable gallery from a folder of images")
    p.add_argument("--images", required=True, help="folder of images")
    p.add_argument("--model", default="model", help="exported model folder")
    p.add_argument("--out", default="my_gallery")
    p.add_argument("--thumb_size", type=int, default=320)
    args = p.parse_args()

    from inference import Engine
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.images) for f in fs
                   if f.lower().endswith(EXTENSIONS))
    if not paths:
        raise SystemExit(f"No images found in {args.images}")
    print(f"Found {len(paths)} images")
    engine = Engine(args.model, load_gallery=False)
    build_gallery(engine.model, paths, args.out, device=engine.device, thumb_size=args.thumb_size)
    print(f"Start the app with:  python server.py --gallery {args.out}")


if __name__ == "__main__":
    main()
