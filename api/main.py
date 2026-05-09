import os
from . import app  # CORRECTED: Use relative import to find app.py in the same folder
import uvicorn
import asyncio
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Local Project Imports
from core.detector import DeepfakeDetector
from config import settings
from fastapi.middleware.cors import CORSMiddleware  # Import the middleware

# --- API Configuration ---

# CRITICAL: Update this path to your specific model file
MODEL_PATH = "models/efficientnet_b4_rwightman-23ab8bcd.pth"

# Initialize FastAPI application
app = FastAPI(
    title="Hackathon Deepfake Detector",
    description="A concurrent API for visual and audio deepfake detection.",
    version="1.0.0"
)

# 1. CORS Middleware Setup
origins = [
        "*",  # CRITICAL: Allows ALL origins (including your local file:// or null origin)
        # If you were deploying this, you would list specific domains here, e.g., "https://your-app.com"
    ]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize the detector globally (this is where the model is loaded)
try:
    detector = DeepfakeDetector(
        model_path=MODEL_PATH,
        cache=None  # Using simple in-memory cache for this demo
    )
except RuntimeError as e:
    # If model loading fails (e.g., file not found or PyTorch issue),
    # we initialize a placeholder detector but mark the system unhealthy.
    print(f"CRITICAL STARTUP ERROR: {e}")
    detector = None


# --- Pydantic Models for Response Bodies ---

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    cache_available: bool = True


class DetectionResponse(BaseModel):
    is_deepfake: bool
    confidence: float
    visual_score: float
    audio_anomalies: float
    compression_artifacts: float
    frames_analyzed: int
    has_audio: bool
    message: str


class InfoResponse(BaseModel):
    name: str
    version: str
    description: str


# --- API Endpoints ---

@app.get("/", response_model=InfoResponse)
async def root():
    """Returns basic information about the API."""
    return InfoResponse(
        name=app.title,
        version=app.version,
        description=app.description
    )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Checks the health and status of the detector components."""
    model_status = detector is not None and detector.model is not None and detector.model != "MODEL_LOADED_SUCCESSFULLY"
    return HealthResponse(
        status="healthy" if model_status else "unhealthy",
        device=settings.DEVICE,
        model_loaded=model_status,
    )


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_video(
        file: UploadFile = File(...),
        skip_cache: bool = Query(False, description="Skip the cache and force re-analysis")
):
    """
    Analyzes an uploaded video file for deepfake characteristics.
    """
    if detector is None:
        raise HTTPException(status_code=503,
                            detail="Service Unavailable: Deepfake detection model failed to load at startup.")

    # 1. Validation Checks
    # Simple check for file size (using settings.MAX_UPLOAD_SIZE)
    MAX_SIZE = settings.MAX_UPLOAD_SIZE
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {MAX_SIZE / (1024 * 1024):.0f}MB."
        )

    # 2. Read file data asynchronously
    video_data = await file.read()

    try:
        # 3. Call the main detection pipeline
        # This function handles saving the temp file, concurrent processing, and cleanup.
        result: Dict[str, Any] = await detector.detect_deepfake(video_data, skip_cache)

        # Check if the result is an error message (sent by the detector on failure)
        if "error" in result:
            raise Exception(result["error"])

        return DetectionResponse(
            is_deepfake=result.get("is_deepfake", False),
            confidence=result.get("confidence", 0.0),
            visual_score=result.get("visual_score", 0.0),
            audio_anomalies=result.get("audio_anomalies", 0.0),
            compression_artifacts=result.get("compression_artifacts", 0.0),
            frames_analyzed=result.get("num_frames_analyzed", 0),
            has_audio=result.get("has_audio", False),
            message=result.get("message", "Detection complete!")
        )

    except Exception as e:
        # Log the detailed error on the server side
        print(f"CRITICAL DETECTION ERROR: {e}")
        # Return a clean 400 error to the client
        raise HTTPException(status_code=400, detail=f"Detection failed: {str(e)}")


@app.post("/clear-cache", tags=["Admin"])
async def clear_cache():
    """Clears the internal results cache."""
    if detector is None:
        return JSONResponse(status_code=200, content={"success": False, "message": "Detector not initialized."})

    detector.cache.clear()
    return JSONResponse(status_code=200, content={"success": True, "message": "Cache cleared successfully"})


# --- Server Startup ---

if __name__ == "__main__":
    host = settings.API_HOST
    port = settings.API_PORT
    workers = settings.API_WORKERS

    print("\n" + "=" * 50)
    print(f"✅ App Name: {app.title}")
    print(f"✅ Debug Mode: {settings.DEBUG}")
    print(f"✅ Device: {settings.DEVICE}")

    model_status = "unhealthy (CRITICAL ERROR)"
    if detector and detector.model:
        model_status = f"successfully loaded ({MODEL_PATH})"

    print(f"✅ Model: {model_status}")
    print(f"✅ DeepfakeDetector initialized. Model: {MODEL_PATH}")
    print("=" * 50 + "\n")

    local_url = f"http://127.0.0.1:{port}"
    print(f" Open the documentation (Swagger UI) here:\n{local_url}/docs")
    print("-" * 50)

    # Start the Uvicorn server
    uvicorn.run(app, host=host, port=port, workers=workers)