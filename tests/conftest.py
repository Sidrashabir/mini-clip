"""
Shared fixture: builds a small synthetic dataset, trains a tiny model in both modes and exports it.
Uses the offline "test-tiny" backbones, so no downloads are needed (works in CI).
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run(args, cwd):
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    d = tmp_path_factory.mktemp("ws")
    from prepare import prepare_synthetic
    prepare_synthetic(str(d / "data" / "synthetic"), n_images=300)
    common = ["--data", "data/synthetic", "--image_backbone", "test-tiny", "--text_backbone", "test-tiny"]
    run([os.path.join(ROOT, "train.py"), "--preset", "test", "--epochs", "6", "--out_dir", "runs/ft", *common], d)
    run([os.path.join(ROOT, "train.py"), "--preset", "cpu", "--epochs", "3", "--aug_views", "1",
         "--out_dir", "runs/frozen", *common], d)
    run([os.path.join(ROOT, "export.py"), "--run", "runs/ft", "--out", "model"], d)
    return d
