from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.inference.service import FontClassifierService


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "CNN_Font_Classification.h5"))
CLASSES_DIR = Path(os.getenv("CLASSES_DIR", ROOT_DIR / "train"))
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Font Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = FontClassifierService(model_path=MODEL_PATH, classes_dir=CLASSES_DIR)
load_error: str | None = None


@app.on_event("startup")
def startup_event() -> None:
    global load_error
    try:
        service.load()
        load_error = None
    except Exception as exc:  # noqa: BLE001
        load_error = str(exc)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ready" if load_error is None else "degraded",
        "model_path": str(MODEL_PATH),
        "classes_dir": str(CLASSES_DIR),
        "error": load_error,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3) -> dict:
    if load_error is not None:
        raise HTTPException(status_code=503, detail=f"Model is unavailable: {load_error}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(payload))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image format.") from exc

    predictions = service.predict(image=image, top_k=top_k)

    return {
        "predictions": [
            {"font": pred.font, "confidence": round(pred.confidence, 6)}
            for pred in predictions
        ]
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
