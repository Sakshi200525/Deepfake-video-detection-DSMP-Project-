import os
import torch
from typing import Literal

# --- Device and Model Configuration ---
# Use CUDA (GPU) if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The fixed input size required by your EfficientNet-B4 model
IMAGE_SIZE = 224

# The number of frames to sample from the video for analysis
MAX_FRAMES = 32

# Deepfake threshold (if confidence > THRESHOLD, it's considered FAKE)
DEEPFAKE_THRESHOLD = 0.70

# --- API and File Configuration ---
# Host and port for the Uvicorn server
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1

# Maximum allowed video file size (10 MB in bytes)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# Set to True for verbose logging and developer features
DEBUG = True


# --- Settings Class ---
class GlobalSettings:
    """Centralized class to access all configuration settings."""

    # Device and Model
    DEVICE: Literal["cuda", "cpu"] = DEVICE
    IMAGE_SIZE: int = IMAGE_SIZE  # <--- THIS IS THE MISSING LINE
    MAX_FRAMES: int = MAX_FRAMES
    DEEPFAKE_THRESHOLD: float = DEEPFAKE_THRESHOLD

    # API and Server
    API_HOST: str = API_HOST
    API_PORT: int = API_PORT
    API_WORKERS: int = API_WORKERS

    # File Management
    MAX_UPLOAD_SIZE: int = MAX_UPLOAD_SIZE

    # Debugging
    DEBUG: bool = DEBUG


settings = GlobalSettings()