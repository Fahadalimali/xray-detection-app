"""FastAPI service for X-ray object detection using the local YOLO model."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "best.pt"))
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

model: YOLO | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once when the API starts instead of on every request."""
    global model
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    yield
    model = None


app = FastAPI(
    title="X-ray Detection API",
    description="Detects trained classes in uploaded X-ray images.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return readiness information for the frontend and deployment checks."""
    return {"status": "ok", "model_loaded": model is not None, "model_path": str(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """Run inference on an uploaded image and return normalized bounding boxes."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The file is not a valid image.") from exc

    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading; try again shortly.")

    width, height = image.size
    result = model.predict(np.asarray(image), verbose=False)[0]
    class_names = result.names
    detections: list[dict[str, Any]] = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = (round(float(value), 2) for value in box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            detections.append(
                {
                    "class_id": class_id,
                    "label": str(class_names[class_id]),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

    return {
        "filename": file.filename,
        "image_width": width,
        "image_height": height,
        "detection_count": len(detections),
        "detections": detections,
    }
