"""
Backbone encoders for Mini-CLIP.

ImageEncoder wraps any `timm` vision model (ResNet, EfficientNet, ViT, ...).
TextEncoder wraps any Hugging Face transformer (DistilBERT, BERT, MiniLM, ...) with
attention-masked mean pooling.

Both expose `out_dim`, `forward()` and helpers to freeze/unfreeze their top layers.

The name "test-tiny" builds a small randomly initialised network instead. It needs no
download and exists so the test suite and CI can run the whole pipeline offline.
"""
import zlib

import torch
import torch.nn as nn

TEST_BACKBONE = "test-tiny"


# --------------------------------------------------------------------------- images
class _TinyCNN(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, dim, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.num_features = dim

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super().__init__()
        self.name = name
        if name == TEST_BACKBONE:
            with torch.random.fork_rng():      # same "pretrained" weights every time, like a real backbone
                torch.manual_seed(0)
                self.backbone = _TinyCNN()
            self.data_cfg = {"input_size": (3, 64, 64), "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
                             "interpolation": "bilinear", "crop_pct": 1.0}
        else:
            import timm
            # num_classes=0 removes the classifier and returns pooled features
            self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
            self.data_cfg = timm.data.resolve_model_data_config(self.backbone)
        self.out_dim = self.backbone.num_features

    def forward(self, pixels):
        return self.backbone(pixels)

    def transform(self, train=False):
        """Preprocessing that matches how the backbone was pretrained (+ augmentation if train)."""
        from torchvision import transforms as T
        size = self.data_cfg["input_size"][-1]
        norm = T.Normalize(self.data_cfg["mean"], self.data_cfg["std"])
        if train:
            return T.Compose([
                T.RandomResizedCrop(size, scale=(0.6, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(), norm,
            ])
        resize = int(round(size / self.data_cfg.get("crop_pct", 0.875)))
        return T.Compose([T.Resize(resize), T.CenterCrop(size), T.ToTensor(), norm])

    def unfreeze_top(self, n_stages: int):
        """Freeze everything, then unfreeze the last n stages (ResNet layers or ViT blocks)."""
        for p in self.parameters():
            p.requires_grad = False
        if n_stages <= 0:
            return
        b = self.backbone
        if hasattr(b, "layer4"):                       # ResNet family
            stages = [b.layer1, b.layer2, b.layer3, b.layer4]
        elif hasattr(b, "blocks"):                     # ViT / DeiT / EfficientNet-style
            stages = list(b.blocks)
        elif hasattr(b, "stages"):                     # ConvNeXt and similar
            stages = list(b.stages)
        else:
            stages = [b]
        for stage in stages[-n_stages:]:
            for p in stage.parameters():
                p.requires_grad = True
        for attr in ("norm", "fc_norm", "head_norm"):  # final norms belong with the top stage
            if getattr(b, attr, None) is not None and isinstance(getattr(b, attr), nn.Module):
                for p in getattr(b, attr).parameters():
                    p.requires_grad = True


# --------------------------------------------------------------------------- text
class _HashTokenizer:
    """Word-hashing tokenizer for the offline test backbone (no vocabulary file needed)."""

    def __init__(self, vocab=4096, max_len=32):
        self.vocab, self.max_len = vocab, max_len

    def __call__(self, texts, **_):
        ids = torch.zeros(len(texts), self.max_len, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, t in enumerate(texts):
            words = t.lower().replace(".", " ").replace(",", " ").split()[: self.max_len]
            for j, w in enumerate(words):
                ids[i, j] = 1 + zlib.crc32(w.encode()) % (self.vocab - 1)
                mask[i, j] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _TinyTextNet(nn.Module):
    def __init__(self, vocab=4096, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.mix = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

    def forward(self, input_ids, attention_mask):
        h = self.mix(self.emb(input_ids))
        return type("Out", (), {"last_hidden_state": h})()


class TextEncoder(nn.Module):
    def __init__(self, name="distilbert-base-uncased", pretrained=True, max_len=40):
        super().__init__()
        self.name, self.max_len = name, max_len
        if name == TEST_BACKBONE:
            self.tokenizer = _HashTokenizer(max_len=max_len)
            with torch.random.fork_rng():
                torch.manual_seed(1)
                self.model = _TinyTextNet()
            self.out_dim = 64
        else:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name) if pretrained else \
                AutoModel.from_config(AutoConfig.from_pretrained(name))
            self.out_dim = self.model.config.hidden_size

    def tokenize(self, texts):
        return self.tokenizer(list(texts), padding="max_length", truncation=True,
                              max_length=self.max_len, return_tensors="pt")

    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)   # mean over real (non-padding) tokens

    def unfreeze_top(self, n_layers: int):
        for p in self.parameters():
            p.requires_grad = False
        if n_layers <= 0:
            return
        layers = None
        for path in ("transformer.layer", "encoder.layer", "layers"):   # DistilBERT, BERT, others
            obj = self.model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                layers = list(obj)
                break
            except AttributeError:
                continue
        for layer in (layers or [self.model])[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
