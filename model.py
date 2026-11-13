"""
Mini-CLIP: a two-tower model that maps images and text into one shared embedding space.

    image ─► ImageEncoder ─► ProjectionHead ─┐
                                              ├─► L2-normalise ─► cosine similarity × temperature
    text  ─► TextEncoder  ─► ProjectionHead ─┘

Trained with the symmetric InfoNCE (contrastive) loss used by OpenAI's CLIP: in a batch of
N matching image-caption pairs, each image must pick out its own caption among N candidates,
and each caption must pick out its own image.
"""
import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import ImageEncoder, TextEncoder


@dataclass
class CLIPConfig:
    image_backbone: str = "resnet50"
    text_backbone: str = "distilbert-base-uncased"
    image_dim: int = 2048          # backbone output sizes (filled in automatically)
    text_dim: int = 768
    embed_dim: int = 256           # size of the shared embedding space
    head_dropout: float = 0.1
    init_temperature: float = 0.07
    max_text_len: int = 40


class ProjectionHead(nn.Module):
    """Linear projection plus a residual MLP refinement, then LayerNorm."""

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(out_dim, out_dim), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.proj(x)
        return self.norm(h + self.mlp(h))


def contrastive_loss(img_emb, txt_emb, logit_scale, labels=None):
    """
    Symmetric InfoNCE. img_emb, txt_emb: (N, D) L2-normalised.
    `labels` (optional, length N): an id per row. Pairs sharing an id (e.g. two captions of the
    same image) are treated as positives instead of being wrongly pushed apart.
    """
    logits = logit_scale * img_emb @ txt_emb.t()             # (N, N) similarity matrix
    if labels is None:
        targets = torch.arange(len(img_emb), device=img_emb.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2
    pos = (labels[:, None] == labels[None, :]).float()
    pos = pos / pos.sum(1, keepdim=True)                      # soft targets over all positives
    loss_i = -(pos * F.log_softmax(logits, dim=1)).sum(1).mean()
    loss_t = -(pos * F.log_softmax(logits.t(), dim=1)).sum(1).mean()
    return (loss_i + loss_t) / 2


class MiniCLIP(nn.Module):
    def __init__(self, cfg: CLIPConfig, load_backbones=True, pretrained=True):
        """
        load_backbones=False builds only the heads: used when training on cached features,
        so the large backbones never need to be in memory.
        """
        super().__init__()
        self.cfg = cfg
        self.image_encoder = self.text_encoder = None
        if load_backbones:
            self.image_encoder = ImageEncoder(cfg.image_backbone, pretrained=pretrained)
            self.text_encoder = TextEncoder(cfg.text_backbone, pretrained=pretrained, max_len=cfg.max_text_len)
            cfg.image_dim, cfg.text_dim = self.image_encoder.out_dim, self.text_encoder.out_dim
        self.image_head = ProjectionHead(cfg.image_dim, cfg.embed_dim, cfg.head_dropout)
        self.text_head = ProjectionHead(cfg.text_dim, cfg.embed_dim, cfg.head_dropout)
        # Learnable temperature, stored as log(1/T) like CLIP.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / cfg.init_temperature)))

    # ---- embeddings -------------------------------------------------------
    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)   # cap at 100, as in CLIP

    def embed_image_features(self, feats):
        return F.normalize(self.image_head(feats), dim=-1)

    def embed_text_features(self, feats):
        return F.normalize(self.text_head(feats), dim=-1)

    def encode_image(self, pixels):
        return self.embed_image_features(self.image_encoder(pixels))

    def encode_text(self, input_ids, attention_mask):
        return self.embed_text_features(self.text_encoder(input_ids, attention_mask))

    # ---- training ---------------------------------------------------------
    def forward(self, pixels=None, input_ids=None, attention_mask=None,
                image_feats=None, text_feats=None, labels=None):
        """Accepts raw inputs (fine-tuning) or pre-computed backbone features (frozen mode)."""
        img = self.embed_image_features(image_feats) if image_feats is not None else self.encode_image(pixels)
        txt = self.embed_text_features(text_feats) if text_feats is not None else \
            self.encode_text(input_ids, attention_mask)
        return contrastive_loss(img, txt, self.scale(), labels)

    def head_state_dict(self):
        """Only the trained parts (heads + temperature): a few MB instead of hundreds."""
        return {k: v for k, v in self.state_dict().items()
                if k.startswith(("image_head.", "text_head.", "logit_scale"))}

    def config_dict(self):
        return asdict(self.cfg)


# --------------------------------------------------------------------------- retrieval metrics
@torch.no_grad()
def retrieval_metrics(img_emb, txt_emb, owner, ks=(1, 5, 10)):
    """
    img_emb: (N_img, D) and txt_emb: (N_txt, D), both L2-normalised.
    owner[j] = index of the image that caption j describes.

    text -> image: for each caption, the rank of its image among all images.
    image -> text: for each image, the rank of its best-scoring true caption among all captions.
    """
    owner = owner.to(txt_emb.device)
    sim = txt_emb @ img_emb.t()                                   # (N_txt, N_img)
    t2i_rank = (sim > sim.gather(1, owner[:, None])).sum(1)       # 0 means ranked first

    caps = _caption_index(owner, img_emb.size(0))                 # (N_img, max_caps)
    sim_i = sim.t()                                               # (N_img, N_txt)
    best_true = sim_i.gather(1, caps).max(1).values
    i2t_rank = (sim_i > best_true[:, None]).sum(1)

    out = {}
    for k in ks:
        out[f"t2i_R@{k}"] = round((t2i_rank < k).float().mean().item() * 100, 2)
        out[f"i2t_R@{k}"] = round((i2t_rank < k).float().mean().item() * 100, 2)
    out["t2i_median_rank"] = t2i_rank.float().median().item() + 1
    out["i2t_median_rank"] = i2t_rank.float().median().item() + 1
    out["mean_recall"] = round(sum(v for k, v in out.items() if "R@" in k) / (2 * len(ks)), 2)
    return out


def _caption_index(owner, n_img):
    """Caption indices per image, padded by repeating each image's first caption."""
    lists = [[] for _ in range(n_img)]
    for j, i in enumerate(owner.tolist()):
        lists[i].append(j)
    width = max(len(c) for c in lists)
    return torch.tensor([c + [c[0]] * (width - len(c)) for c in lists], device=owner.device)


# --------------------------------------------------------------------------- device and checkpoints
def pick_device(choice="auto"):
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path_or_dir, device="cpu"):
    """
    Build a full Mini-CLIP (backbones + heads) from a checkpoint.
    Frozen-mode checkpoints hold only the heads, so pretrained backbones are loaded first.
    Fine-tune checkpoints hold every weight, so the backbones are created without downloading weights.
    """
    path = os.path.join(path_or_dir, "best.pt") if os.path.isdir(path_or_dir) else path_or_dir
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CLIPConfig(**ckpt["model_cfg"])
    frozen = ckpt["mode"] == "frozen"
    model = MiniCLIP(cfg, load_backbones=True, pretrained=frozen)
    state = {k: v.float() for k, v in ckpt["state"].items()}
    missing, unexpected = model.load_state_dict(state, strict=not frozen)
    if frozen:
        assert not unexpected and all(k.startswith(("image_encoder", "text_encoder")) for k in missing)
    return model.to(device).eval(), ckpt
