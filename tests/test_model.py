"""Unit tests for the loss, metrics, model and encoders."""
import math

import torch
import torch.nn.functional as F

from encoders import ImageEncoder, TextEncoder
from model import CLIPConfig, MiniCLIP, contrastive_loss, retrieval_metrics


def unit(n, d=16, seed=0):
    return F.normalize(torch.randn(n, d, generator=torch.Generator().manual_seed(seed)), dim=-1)


def test_loss_is_low_for_matching_pairs_and_ln_n_for_unrelated():
    e = unit(32)
    assert contrastive_loss(e, e, torch.tensor(100.0)).item() < 0.01
    z = torch.zeros(32, 16)  # identical embeddings: every candidate equally likely
    assert abs(contrastive_loss(F.normalize(z + 1, dim=-1), F.normalize(z + 1, dim=-1),
                                torch.tensor(10.0)).item() - math.log(32)) < 1e-4


def test_loss_is_symmetric():
    a, b = unit(16, seed=1), unit(16, seed=2)
    assert torch.allclose(contrastive_loss(a, b, torch.tensor(5.0)), contrastive_loss(b, a, torch.tensor(5.0)))


def test_label_aware_loss_equals_plain_loss_when_labels_are_unique():
    a, b = unit(8, seed=3), unit(8, seed=4)
    assert torch.allclose(contrastive_loss(a, b, torch.tensor(20.0)),
                          contrastive_loss(a, b, torch.tensor(20.0), labels=torch.arange(8)))


def test_duplicate_labels_share_the_target_probability():
    """Rows with the same label are all positives: the target spreads evenly over them."""
    a, b = unit(5, seed=5), unit(5, seed=6)
    labels = torch.tensor([0, 0, 1, 2, 3])
    logits = 20.0 * a @ b.t()
    targets = (labels[:, None] == labels[None, :]).float()
    targets = targets / targets.sum(1, keepdim=True)
    manual = (-(targets * logits.log_softmax(1)).sum(1).mean()
              - (targets * logits.t().log_softmax(1)).sum(1).mean()) / 2
    assert torch.allclose(contrastive_loss(a, b, torch.tensor(20.0), labels), manual)


def test_metrics_perfect_and_multi_caption():
    img = unit(10)
    txt = torch.cat([img, img])                    # two captions per image, both perfect
    owner = torch.cat([torch.arange(10), torch.arange(10)])
    m = retrieval_metrics(img, txt, owner)
    assert m["t2i_R@1"] == 100 and m["i2t_R@1"] == 100 and m["t2i_median_rank"] == 1


def test_metrics_known_ranks():
    img = torch.eye(4)
    txt = torch.eye(4)[[1, 0, 2, 3]]               # captions 0 and 1 point at the wrong image
    m = retrieval_metrics(img, txt, torch.arange(4))
    assert m["t2i_R@1"] == 50 and m["t2i_R@5"] == 100


def test_model_raw_and_feature_paths_agree():
    model = MiniCLIP(CLIPConfig(image_backbone="test-tiny", text_backbone="test-tiny", embed_dim=32)).eval()
    x = torch.randn(4, 3, 64, 64)
    tok = model.text_encoder.tokenize(["a red circle", "a blue square", "a cat", "a dog"])
    with torch.no_grad():
        raw = model(pixels=x, **tok)
        feats = model(image_feats=model.image_encoder(x), text_feats=model.text_encoder(**tok))
        emb = model.encode_image(x)
    assert torch.allclose(raw, feats) and torch.allclose(emb.norm(dim=-1), torch.ones(4))
    assert set(k.split(".")[0] for k in model.head_state_dict()) == {"image_head", "text_head", "logit_scale"}


def test_temperature_is_capped():
    model = MiniCLIP(CLIPConfig(image_backbone="test-tiny", text_backbone="test-tiny"), load_backbones=False)
    model.logit_scale.data.fill_(10.0)
    assert model.scale().item() == 100.0


def test_unfreeze_top_and_padding_is_ignored():
    enc = ImageEncoder("test-tiny")
    enc.unfreeze_top(0)
    assert not any(p.requires_grad for p in enc.parameters())
    txt = TextEncoder("test-tiny", max_len=16)
    a = txt(**txt.tokenize(["a red circle"]))
    b = txt(**txt.tokenize(["a red circle"]))     # mean pooling must ignore padding positions
    assert torch.allclose(a, b)


def test_test_backbones_are_deterministic():
    a, b = ImageEncoder("test-tiny"), ImageEncoder("test-tiny")
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
