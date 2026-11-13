"""End-to-end: data -> training (both modes) -> export -> inference engine -> REST API."""
import io
import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api import build_router
from data import load_split
from inference import Engine


@pytest.fixture(scope="module")
def engine(workspace):
    return Engine(str(workspace / "model"))


@pytest.fixture(scope="module")
def client(engine):
    app = FastAPI()
    app.include_router(build_router(engine))
    return TestClient(app)


def png_bytes(color=(220, 40, 40)):
    buf = io.BytesIO()
    Image.new("RGB", (96, 96), color).save(buf, "PNG")
    return buf.getvalue()


def test_dataset_splits(workspace):
    d = str(workspace / "data" / "synthetic")
    sizes = {s: len(load_split(d, s)) for s in ("train", "val", "test")}
    assert sizes == {"train": 240, "val": 30, "test": 30}
    assert all(len(caps) == 3 for _, caps in load_split(d, "test"))


def test_training_learns_and_saves(workspace):
    for run in ("ft", "frozen"):
        assert os.path.exists(workspace / "runs" / run / "best.pt")
    cfg = json.load(open(workspace / "runs" / "ft" / "train_config.json"))
    assert cfg["best_val_mean_recall"] > 25          # random guessing on 30 images is ~20


def test_frozen_mode_cached_features(workspace):
    caches = [f for _, _, fs in os.walk(workspace / "cache") for f in fs]
    assert any(f.startswith("train_aug1") for f in caches) and any(f.startswith("val_aug0") for f in caches)


def test_export_contents(workspace):
    m = workspace / "model"
    for f in ("config.json", "weights.pt", "gallery/items.json", "gallery/image_emb.npy",
              "gallery/caption_emb.npy", "gallery/map.json"):
        assert os.path.exists(m / f), f
    assert len(os.listdir(m / "gallery" / "images")) == 300


def test_search_text_sorted(engine):
    res = engine.search_text("a red circle", k=7)
    scores = [r["score"] for r in res]
    assert len(res) == 7 and scores == sorted(scores, reverse=True)


def test_search_image_finds_itself(engine):
    img = Image.open(engine.image_path(5))
    assert engine.search_image(img, k=1)[0]["index"] == 5


def test_classify_and_describe(engine):
    probs = engine.classify(Image.new("RGB", (64, 64), "red"), ["circle", "square", "triangle"])
    assert abs(sum(probs.values()) - 1) < 1e-3
    assert 1 <= len(engine.describe(Image.new("RGB", (64, 64), "blue"), k=3)) <= 3


def test_explain_shapes(engine):
    ex = engine.explain("a large red circle", 0, grid=5)
    assert len(ex["words"]) == 4 and len(ex["heatmap"]) == 5 and max(map(max, ex["heatmap"])) <= 1.0


def test_api_endpoints(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/model").json()["gallery_images"] == 300
    r = client.post("/api/search/text", json={"query": "a blue square", "k": 3}).json()["results"]
    assert len(r) == 3 and r[0]["url"].startswith("/api/images/")
    assert client.get(r[0]["url"]).status_code == 200
    files = {"file": ("x.png", png_bytes(), "image/png")}
    assert len(client.post("/api/search/image", files=files, data={"k": 4}).json()["results"]) == 4
    probs = client.post("/api/classify", files=files, data={"labels": "red, green, blue"}).json()["probabilities"]
    assert set(probs) == {"red", "green", "blue"}
    assert "captions" in client.post("/api/describe", files=files).json()
    assert "heatmap" in client.post("/api/explain", json={"query": "a red circle", "index": 0}).json()


def test_api_validation_and_security(client):
    assert client.post("/api/search/text", json={"query": "", "k": 3}).status_code == 422
    assert client.post("/api/search/text", json={"query": "x", "k": 1000}).status_code == 422
    bad = {"file": ("x.txt", b"not an image", "text/plain")}
    assert client.post("/api/search/image", files=bad).status_code == 415
    good = {"file": ("x.png", png_bytes(), "image/png")}
    assert client.post("/api/classify", files=good, data={"labels": "only-one"}).status_code == 422
    assert client.post("/api/explain", json={"query": "x", "index": 99999}).status_code == 422
    assert client.get("/api/images/..%2Fconfig.json").status_code == 404
    assert client.get("/api/images/config.json").status_code == 404


def test_export_web(workspace):
    """The static website export: weights, embeddings, image sheets and self-check vectors."""
    import numpy as np
    from conftest import ROOT, run
    run([os.path.join(ROOT, "export_web.py"), "--run", "runs/frozen", "--out", "site", "--per_sheet", "16",
         "--allow_test_model"], workspace)
    site = workspace / "site"
    data = json.load(open(site / "data.json"))
    assert data["sheets"]["count"] == 2 and len(data["items"]) == 30          # 30 test photos, 16 per sheet
    assert sorted(f for f in os.listdir(site) if f.startswith("atlas_")) == ["atlas_00.jpg", "atlas_01.jpg"]
    buf = open(site / "model.bin", "rb").read()
    end = max(s["offset"] + int(np.prod(s["shape"])) * (4 if s["dtype"] == "f32" else 2)
              for s in data["sections"].values())
    assert end == len(buf)                                                    # sections tile the file exactly
    img = data["sections"]["image_emb"]
    emb = np.frombuffer(buf, "<f2", int(np.prod(img["shape"])), img["offset"]).reshape(img["shape"])
    assert np.allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1, atol=1e-2)
    assert len(data["checks"]) == 3 and len(data["map"]["xy"]) == 30
