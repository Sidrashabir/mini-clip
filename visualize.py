"""
Create the figures used in the README, saved into the run folder.

    python visualize.py --run runs/flickr8k_cpu

    training_curves.png     losses, validation recall and the learned temperature per epoch
    retrieval_examples.png  test queries with their top-5 results (green border = the correct photo)
    similarity_matrix.png   cosine similarity between 8 test photos and their captions
"""
import argparse
import csv
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from data import load_split  # noqa: E402
from evaluate import embed_test_set  # noqa: E402
from model import load_checkpoint, pick_device  # noqa: E402


def plot_training(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "log.csv"))))
    ep = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    axes[0].plot(ep, [float(r["val_loss"]) for r in rows], label="validation")
    axes[0].set_title("Contrastive loss"); axes[0].legend()
    axes[1].plot(ep, [float(r["val_mean_recall"]) for r in rows], label="mean recall")
    axes[1].plot(ep, [float(r["val_t2i_R@1"]) for r in rows], label="text→image R@1")
    axes[1].plot(ep, [float(r["val_t2i_R@5"]) for r in rows], label="text→image R@5")
    axes[1].set_title("Validation retrieval (%)"); axes[1].legend()
    axes[2].plot(ep, [float(r["temperature"]) for r in rows], color="tab:purple")
    axes[2].set_title("Learned temperature")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150); plt.close(fig)


def show(ax, path, border=None):
    ax.imshow(Image.open(path).convert("RGB").resize((224, 224)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(border is not None)
        if border:
            s.set_edgecolor(border); s.set_linewidth(5)


@torch.no_grad()
def plot_retrieval(run_dir, items, img, txt, owner, queries=6, k=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    picks = torch.randperm(len(owner), generator=g)[:queries].tolist()
    captions = [c.rstrip(" .") for _, caps in items for c in caps]
    fig, axes = plt.subplots(queries, k + 1, figsize=(2.6 * (k + 1), 2.6 * queries),
                             gridspec_kw={"width_ratios": [1.6] + [1] * k})
    for row, j in enumerate(picks):
        scores = img @ txt[j]
        top = scores.topk(k).indices.tolist()
        ax = axes[row][0]
        ax.axis("off")
        ax.text(0, 0.5, "\n".join(textwrap.wrap(captions[j], 26)), va="center", fontsize=10)
        for col, i in enumerate(top):
            show(axes[row][col + 1], items[i][0], "limegreen" if i == owner[j].item() else None)
            axes[row][col + 1].set_title(f"{scores[i]:.2f}", fontsize=9)
    fig.suptitle("Text → image search on unseen test photos (green = the photo the caption was written for)")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "retrieval_examples.png"), dpi=120); plt.close(fig)


@torch.no_grad()
def plot_similarity(run_dir, items, img, txt, owner, n=8, seed=3):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(items), generator=g)[:n].tolist()
    first = [owner.tolist().index(i) for i in idx]
    sim = (img[idx] @ txt[first].t()).cpu().numpy()
    captions = [c.rstrip(" .") for _, caps in items for c in caps]
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes([0.32, 0.05, 0.6, 0.72])
    ax.imshow(sim, cmap="viridis", aspect="auto")  # fill the axes so columns line up with photos
    for a in range(n):
        for b in range(n):
            ax.text(b, a, f"{sim[a, b]:.2f}", ha="center", va="center",
                    color="white" if sim[a, b] < sim.max() * 0.75 else "black", fontsize=9)
    ax.set_xticks([]); ax.set_yticks(range(n))
    ax.set_yticklabels(["\n".join(textwrap.wrap(captions[f], 40)) for f in first], fontsize=8)
    for b, i in enumerate(idx):                                   # photos along the top
        tile = fig.add_axes([0.32 + b * 0.6 / n, 0.79, 0.6 / n, 0.16])
        show(tile, items[i][0])
    fig.suptitle("Cosine similarity: photos (top) vs captions (left). A bright diagonal means correct matches.",
                 y=0.99)
    fig.savefig(os.path.join(run_dir, "similarity_matrix.png"), dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    plot_training(args.run)
    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.run, device)
    items = load_split(ckpt["train_cfg"]["data"], "test")
    img, txt, owner = embed_test_set(model, items, device)
    img, txt = img.cpu(), txt.cpu()
    plot_retrieval(args.run, items, img, txt, owner)
    plot_similarity(args.run, items, img, txt, owner)
    print(f"Saved training_curves.png, retrieval_examples.png and similarity_matrix.png in {args.run}/")


if __name__ == "__main__":
    main()
