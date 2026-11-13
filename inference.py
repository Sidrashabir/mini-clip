"""
Inference engine shared by the web app and the REST API.

    engine = Engine("model")                  # exported model folder (see export.py)
    engine.search_text("a dog running on the beach", k=12)
    engine.search_image(pil_image, k=12)
    engine.describe(pil_image, k=5)           # best-matching captions from the caption bank
    engine.classify(pil_image, ["cat", "dog", "car"])
    engine.explain(query, gallery_index)      # which words and image regions drove a match

Model location: the path given, else env MODEL_DIR (default "model"), else download the
Hugging Face model repo in env MODEL_REPO. Gallery location: the path given, else env
GALLERY_DIR, else <model>/gallery.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import CLIPConfig, MiniCLIP

try:
    import faiss
except ImportError:  # optional: exact numpy search is used instead
    faiss = None


def resolve_model_dir(path=None):
    path = path or os.environ.get("MODEL_DIR", "model")
    if os.path.exists(os.path.join(path, "weights.pt")):
        return path
    repo = os.environ.get("MODEL_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN"))
    raise FileNotFoundError(f"No model in '{path}'. Run `python export.py --run runs/<run>` first, "
                            "or set MODEL_REPO.")


class VectorIndex:
    """Inner-product search over L2-normalised vectors (= cosine similarity). FAISS if installed."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.faiss = None
        if faiss is not None and len(self.vectors) > 0:
            self.faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self.faiss.add(self.vectors)

    def __len__(self):
        return len(self.vectors)

    def search(self, query: np.ndarray, k: int):
        k = min(k, len(self.vectors))
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        if self.faiss is not None:
            scores, idx = self.faiss.search(q, k)
            return idx[0], scores[0]
        scores = self.vectors @ q[0]
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]


class Engine:
    def __init__(self, model_dir=None, gallery_dir=None, device="cpu", load_gallery=True):
        self.model_dir = resolve_model_dir(model_dir)
        self.device = device
        with open(os.path.join(self.model_dir, "config.json")) as f:
            self.config = json.load(f)
        cfg = CLIPConfig(**self.config["model_cfg"])
        frozen = self.config["mode"] == "frozen"
        self.model = MiniCLIP(cfg, load_backbones=True, pretrained=frozen)
        state = torch.load(os.path.join(self.model_dir, "weights.pt"), map_location="cpu", weights_only=True)
        self.model.load_state_dict({k: v.float() for k, v in state.items()}, strict=not frozen)
        self.model.to(device).eval()
        self.transform = self.model.image_encoder.transform(train=False)
        self.gallery_dir = None
        if load_gallery:
            self.load_gallery(gallery_dir or os.environ.get("GALLERY_DIR") or
                              os.path.join(self.model_dir, "gallery"))

    # ------------------------------------------------------------------ gallery
    def load_gallery(self, gallery_dir):
        if not os.path.exists(os.path.join(gallery_dir, "image_emb.npy")):
            raise FileNotFoundError(f"No gallery in '{gallery_dir}'. Export with a gallery or run gallery.py.")
        self.gallery_dir = gallery_dir
        with open(os.path.join(gallery_dir, "items.json"), encoding="utf-8") as f:
            self.items = json.load(f)
        self.image_index = VectorIndex(np.load(os.path.join(gallery_dir, "image_emb.npy")))
        cap_path = os.path.join(gallery_dir, "caption_emb.npy")
        self.caption_index, self.captions = None, []
        if os.path.exists(cap_path):
            self.caption_index = VectorIndex(np.load(cap_path))
            with open(os.path.join(gallery_dir, "captions.json"), encoding="utf-8") as f:
                self.captions = json.load(f)
        map_path = os.path.join(gallery_dir, "map.json")
        self.map = json.load(open(map_path)) if os.path.exists(map_path) else None

    def image_path(self, i):
        return os.path.join(self.gallery_dir, "images", self.items[i]["file"])

    def _results(self, idx, scores):
        return [{"index": int(i), "path": self.image_path(int(i)), "file": self.items[int(i)]["file"],
                 "score": round(float(s), 4), "captions": self.items[int(i)]["captions"]}
                for i, s in zip(idx, scores)]

    # ------------------------------------------------------------------ encoding
    @torch.no_grad()
    def encode_text(self, texts):
        tok = self.model.text_encoder.tokenize(texts)
        emb = self.model.encode_text(tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device))
        return emb.cpu().numpy()

    @torch.no_grad()
    def encode_image(self, image: Image.Image):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return self.model.encode_image(x).cpu().numpy()[0]

    # ------------------------------------------------------------------ search
    def search_text(self, query, k=12):
        idx, scores = self.image_index.search(self.encode_text([query])[0], k)
        return self._results(idx, scores)

    def search_image(self, image, k=12):
        idx, scores = self.image_index.search(self.encode_image(image), k)
        return self._results(idx, scores)

    def describe(self, image, k=5):
        """Image -> text retrieval: the captions in the bank that best match an image."""
        if self.caption_index is None:
            return []
        idx, scores = self.caption_index.search(self.encode_image(image), k * 10)
        out, seen = [], set()
        for i, s in zip(idx, scores):
            text = self.captions[int(i)]["text"]
            if text.lower() not in seen:
                seen.add(text.lower())
                out.append({"caption": text, "score": round(float(s), 4)})
            if len(out) == k:
                break
        return out

    # ------------------------------------------------------------------ zero-shot
    @torch.no_grad()
    def classify(self, image, labels, template="a photo of {}"):
        """Zero-shot classification: compare the image with a caption built from each label."""
        labels = [lbl.strip() for lbl in labels if lbl.strip()]
        if not labels:
            return {}
        txt = torch.from_numpy(self.encode_text([template.format(lbl) for lbl in labels]))
        img = torch.from_numpy(self.encode_image(image))
        probs = F.softmax(self.model.scale().cpu() * txt @ img, dim=0)
        return {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)}

    # ------------------------------------------------------------------ explainability
    @torch.no_grad()
    def explain(self, query, gallery_index, grid=7):
        """
        Occlusion-based explanation of why an image matches a query.
        Words: similarity drop when each word is removed from the query.
        Regions: similarity drop when each cell of a grid is greyed out in the image.
        """
        image = Image.open(self.image_path(gallery_index)).convert("RGB")
        img_emb = torch.from_numpy(self.encode_image(image))
        words = query.split()
        variants = [query] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        txt = torch.from_numpy(self.encode_text(variants))
        base = float(txt[0] @ img_emb)
        word_importance = [(w, round(base - float(txt[i + 1] @ img_emb), 4)) for i, w in enumerate(words)]

        x = self.transform(image).unsqueeze(0)
        _, _, h, w = x.shape
        batch = []
        for r in range(grid):
            for c in range(grid):
                xo = x.clone()
                xo[:, :, r * h // grid:(r + 1) * h // grid, c * w // grid:(c + 1) * w // grid] = 0.0
                batch.append(xo)
        q = txt[0].to(self.device)
        sims = []
        for chunk in torch.cat(batch).split(16):
            sims.append(self.model.encode_image(chunk.to(self.device)) @ q)
        drop = (base - torch.cat(sims).cpu()).reshape(grid, grid).clamp_min(0).numpy()
        heat = drop / (drop.max() + 1e-9)
        return {"similarity": round(base, 4), "words": word_importance, "heatmap": heat.tolist(),
                "image_size": list(self.transform(image).shape[-2:])}

    # ------------------------------------------------------------------ info
    def info(self):
        cfg = self.config["model_cfg"]
        trained = sum(v.numel() for v in self.model.head_state_dict().values()) if self.config["mode"] == "frozen" \
            else sum(p.numel() for p in self.model.parameters())
        return {
            "name": self.config.get("name", "Mini-CLIP"),
            "mode": self.config["mode"],
            "image_backbone": cfg["image_backbone"],
            "text_backbone": cfg["text_backbone"],
            "embed_dim": cfg["embed_dim"],
            "trained_parameters": trained,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "temperature": round(1 / self.model.scale().item(), 4),
            "dataset": self.config.get("dataset"),
            "gallery_images": len(self.image_index) if self.gallery_dir else 0,
            "caption_bank": len(self.captions),
            "search_backend": "faiss" if faiss is not None else "numpy",
            "val_metrics": self.config.get("val_metrics"),
            "eval": self.config.get("eval"),
        }
