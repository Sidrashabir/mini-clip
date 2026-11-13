"""
Production entry point: web app + REST API in one service.

    python server.py                          # model in ./model (from export.py)
    python server.py --gallery my_gallery     # search your own photos (from gallery.py)
    uvicorn server:app --host 0.0.0.0 --port 7860   # how Docker / Hugging Face run it

    Web app:  http://localhost:7860      API docs: http://localhost:7860/docs

Environment variables: MODEL_DIR, MODEL_REPO, GALLERY_DIR, MAX_CONCURRENT (default 2), TORCH_THREADS
"""
import argparse
import logging
import os
import time

import gradio as gr
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api import build_router
from app import CSS, THEME, build_ui
from inference import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mini-clip")


def create_app(model_dir=None, gallery_dir=None) -> FastAPI:
    if os.environ.get("TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
    t0 = time.time()
    engine = Engine(model_dir, gallery_dir)
    info = engine.info()
    log.info("Loaded %s (%s + %s), gallery of %d images, in %.1fs", info["name"], info["image_backbone"],
             info["text_backbone"], info["gallery_images"], time.time() - t0)

    api = FastAPI(title="Mini-CLIP API", version="1.0.0",
                  description="Text-to-image search, image search, zero-shot classification and "
                              "explanations from a CLIP-style model trained with contrastive learning.")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @api.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api") and not request.url.path.startswith("/api/images"):
            log.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code,
                     (time.perf_counter() - start) * 1000)
        return response

    api.include_router(build_router(engine))
    demo = build_ui(engine)
    demo.queue(default_concurrency_limit=int(os.environ.get("MAX_CONCURRENT", 2)), max_size=32)
    return gr.mount_gradio_app(api, demo, path="/", theme=THEME, css=CSS, footer_links=["api"],
                               allowed_paths=[os.path.abspath(engine.gallery_dir)])


app = create_app() if __name__ != "__main__" and os.environ.get("MINI_CLIP_NO_AUTOLOAD") != "1" else None

if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="exported model folder (default: model)")
    p.add_argument("--gallery", default=None, help="gallery folder (default: <model>/gallery)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = p.parse_args()
    print(f"\n  Open http://localhost:{args.port} in your browser  (API docs: http://localhost:{args.port}/docs)\n")
    uvicorn.run(create_app(args.model, args.gallery), host=args.host, port=args.port)
