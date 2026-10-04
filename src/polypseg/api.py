"""FastAPI service for polyp segmentation.

Run locally:
    MODEL_PATH=models/best.pt uvicorn polypseg.api:app --port 8000
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError

from polypseg import __version__
from polypseg.inference import Segmenter, encode_png, mask_to_image, overlay_mask

logger = logging.getLogger("polypseg.api")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_PIXELS = 25_000_000

UploadFileParam = Annotated[UploadFile, File(description="Colonoscopy image (JPEG or PNG)")]
ThresholdParam = Annotated[float, Query(ge=0.0, le=1.0, description="Mask probability cut-off")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.segmenter = None
    model_path = Path(os.environ.get("MODEL_PATH", "models/best.pt"))
    if model_path.is_file():
        app.state.segmenter = Segmenter.from_checkpoint(model_path, device=os.environ.get("DEVICE"))
        logger.info("Loaded model from %s", model_path)
    else:
        logger.warning("No model checkpoint at %s; /predict will return 503", model_path)
    yield


def _require_model(request: Request) -> Segmenter:
    segmenter = request.app.state.segmenter
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not loaded")
    return segmenter


def _load_upload(file: UploadFile) -> Image.Image:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Image resolution too large")
        image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image") from exc
    return image.convert("RGB")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Polyp Segmentation API",
        version=__version__,
        description=(
            "Segments polyps in colonoscopy images with a ResNet34 U-Net trained on Kvasir-SEG. "
            "Research demo only - not a medical device and not for clinical use."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        segmenter = request.app.state.segmenter
        if segmenter is None:
            return JSONResponse(status_code=503, content={"status": "model_not_loaded"})
        return {"status": "ok", "version": __version__, "img_size": segmenter.img_size}

    @app.post("/predict")
    def predict(
        request: Request,
        file: UploadFileParam,
        threshold: ThresholdParam = 0.5,
    ):
        """Return the predicted mask and an overlay as base64-encoded PNGs."""
        segmenter = _require_model(request)
        image = _load_upload(file)
        started = time.perf_counter()
        mask = segmenter.predict(image, threshold)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "width": image.width,
            "height": image.height,
            "threshold": threshold,
            "mask_area_fraction": float(mask.mean()),
            "inference_ms": round(elapsed_ms, 1),
            "mask_png_base64": base64.b64encode(encode_png(mask_to_image(mask))).decode(),
            "overlay_png_base64": base64.b64encode(encode_png(overlay_mask(image, mask))).decode(),
        }

    @app.post("/predict/overlay", response_class=Response)
    def predict_overlay(
        request: Request,
        file: UploadFileParam,
        threshold: ThresholdParam = 0.5,
    ):
        """Return the overlay directly as a PNG (handy with curl)."""
        segmenter = _require_model(request)
        image = _load_upload(file)
        mask = segmenter.predict(image, threshold)
        return Response(content=encode_png(overlay_mask(image, mask)), media_type="image/png")

    return app


app = create_app()
