"""
REST API for Mini-CLIP (interactive docs at /docs).

    GET  /api/health                 liveness check
    GET  /api/model                  model details and metrics
    POST /api/search/text            {"query": "...", "k": 12}  -> ranked images
    POST /api/search/image           multipart: file, k         -> visually/semantically similar images
    POST /api/describe               multipart: file, k         -> best-matching captions
    POST /api/classify               multipart: file, labels    -> zero-shot label probabilities
    POST /api/explain                {"query": "...", "index": 12} -> word + region importance
    GET  /api/images/{file}          gallery thumbnail
"""
import asyncio
import io
import os
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from inference import Engine

MAX_UPLOAD_MB = 10
_heavy = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", 2)))


class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, examples=["a dog running through the snow"])
    k: int = Field(12, ge=1, le=100)


class ExplainQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, examples=["a dog running through the snow"])
    index: int = Field(..., ge=0, description="gallery index from a search result")


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_MB} MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(415, "File is not a supported image")


def build_router(engine: Engine) -> APIRouter:
    r = APIRouter(prefix="/api", tags=["Mini-CLIP"])

    def with_urls(results):
        for x in results:
            x["url"] = f"/api/images/{x['file']}"
            x.pop("path", None)
        return results

    @r.get("/health")
    def health():
        return {"status": "ok"}

    @r.get("/model")
    def model_info():
        return engine.info()

    @r.post("/search/text")
    async def search_text(q: TextQuery):
        return {"query": q.query, "results": with_urls(await asyncio.to_thread(engine.search_text, q.query, q.k))}

    @r.post("/search/image")
    async def search_image(file: UploadFile = File(...), k: int = Form(12, ge=1, le=100)):
        img = await read_image(file)
        return {"results": with_urls(await asyncio.to_thread(engine.search_image, img, k))}

    @r.post("/describe")
    async def describe(file: UploadFile = File(...), k: int = Form(5, ge=1, le=20)):
        img = await read_image(file)
        return {"captions": await asyncio.to_thread(engine.describe, img, k)}

    @r.post("/classify")
    async def classify(file: UploadFile = File(...),
                       labels: str = Form(..., examples=["dog, cat, horse, car"]),
                       template: str = Form("a photo of {}")):
        names = [x for x in labels.split(",") if x.strip()][:50]
        if len(names) < 2:
            raise HTTPException(422, "Give at least two comma-separated labels")
        if "{}" not in template:
            raise HTTPException(422, "template must contain {} where the label goes")
        img = await read_image(file)
        return {"probabilities": await asyncio.to_thread(engine.classify, img, names, template)}

    @r.post("/explain")
    async def explain(q: ExplainQuery):
        if q.index >= len(engine.items):
            raise HTTPException(422, f"index must be below {len(engine.items)}")
        async with _heavy:
            return await asyncio.to_thread(engine.explain, q.query, q.index)

    @r.get("/images/{file}")
    def image(file: str):
        if not re.fullmatch(r"\d{5,7}\.jpg", file):        # no path tricks
            raise HTTPException(404)
        path = os.path.join(engine.gallery_dir, "images", file)
        if not os.path.exists(path):
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    return r
