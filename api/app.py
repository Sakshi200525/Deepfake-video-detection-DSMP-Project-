from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
import uuid
import cv2
import numpy as np
from typing import Dict, Any
import json
import hashlib
from datetime import datetime

app = FastAPI(title="DeepGuard AI Detection API", version="1.0.0")

# CORS middleware - Frontend साठी allow करण्यासाठी
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production मध्ये specific origins द्या
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload folder तयार करा
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Temporary storage for file IDs
file_storage = {}

class DeepGuardModel:
    """Deepfake Detection Model (Simulated - Real model साठी replace करा)"""
    
    def __init__(self):
        # Real model weights येथे load करा
        # self.model = torch.load('model.pth') etc.
        pass
    
    def analyze_visual(self, video_path: str) -> Dict[str, Any]:
        """Visual analysis using OpenCV and AI model"""
        
        # Video frames extract करा
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        face_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Face detection (simplified - real model वापरा)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_count = max(face_count, len(faces))
            
            # Limit frames for performance
            if frame_count >= 150:
                break
        
        cap.release()
        
        # Simulated AI predictions
        # Real model नी probability काढा
        fake_probability = np.random.randint(15, 85)
        confidence = np.random.randint(65, 98)
        
        return {
            "frames_processed": frame_count,
            "faces_detected": face_count if face_count > 0 else 1,
            "fake_probability": int(fake_probability),
            "confidence": int(confidence),
            "features": {
                "face_morphing_score": round(np.random.uniform(0.1, 0.9), 2),
                "eye_blink_rate": round(np.random.uniform(0.5, 2.5), 2),
                "lip_sync_score": round(np.random.uniform(0.3, 0.95), 2)
            }
        }
    
    def analyze_audio(self, video_path: str) -> Dict[str, Any]:
        """Audio analysis for deepfake detection"""
        
        try:
            # Extract audio using ffmpeg (install ffmpeg separately)
            import subprocess
            audio_path = video_path.replace('.mp4', '_audio.wav')
            
            # Extract audio from video
            cmd = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y"
            subprocess.run(cmd, shell=True, capture_output=True)
            
            if os.path.exists(audio_path):
                # Audio features extract करा
                import librosa
                y, sr = librosa.load(audio_path, duration=10)
                
                # MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Simulated detection
                fake_probability = np.random.randint(10, 90)
                confidence = np.random.randint(60, 95)
                audio_extracted = True
                
                # Cleanup
                os.remove(audio_path)
            else:
                # No audio track
                fake_probability = 0
                confidence = 50
                audio_extracted = False
                
        except Exception as e:
            print(f"Audio analysis error: {e}")
            fake_probability = 0
            confidence = 50
            audio_extracted = False
        
        return {
            "audio_extracted": audio_extracted,
            "fake_probability": int(fake_probability),
            "confidence": int(confidence),
            "features": {
                "mfcc_mean": float(np.random.uniform(-10, 10)) if audio_extracted else 0,
                "spectral_centroid": float(np.random.uniform(100, 5000)) if audio_extracted else 0
            }
        }
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """Combined prediction from visual and audio analysis"""
        
        # Run both analyses
        visual_result = self.analyze_visual(video_path)
        audio_result = self.analyze_audio(video_path)
        
        # Weighted combination
        visual_weight = 0.7
        audio_weight = 0.3
        
        # Calculate final fake probability
        if audio_result['audio_extracted']:
            final_fake = (visual_result['fake_probability'] * visual_weight + 
                         audio_result['fake_probability'] * audio_weight)
        else:
            final_fake = visual_result['fake_probability']
            audio_weight = 0
            visual_weight = 1.0
        
        final_real = 100 - final_fake
        
        # Determine verdict
        verdict = "FAKE" if final_fake >= 50 else "REAL"
        overall_confidence = max(visual_result['confidence'], 
                                audio_result['confidence'] if audio_result['audio_extracted'] else 50)
        
        return {
            "final_result": {
                "verdict": verdict,
                "overall_confidence": int(overall_confidence),
                "real_percentage": int(final_real),
                "fake_percentage": int(final_fake)
            },
            "visual_analysis": visual_result,
            "audio_analysis": audio_result,
            "timestamp": datetime.now().isoformat()
        }

# Initialize model
model = DeepGuardModel()

@app.get("/")
async def root():
    return {
        "message": "DeepGuard AI Detection API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": ["/upload", "/detect", "/health"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for analysis"""
    
    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store file info
        file_storage[file_id] = {
            "path": file_path,
            "filename": file.filename,
            "size": os.path.getsize(file_path),
            "upload_time": datetime.now().isoformat()
        }
        
        return JSONResponse(
            content={
                "success": True,
                "file_id": file_id,
                "filename": file.filename,
                "message": "File uploaded successfully"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/detect")
async def detect_deepfake(file_id: str):
    """Run deepfake detection on uploaded video"""
    
    # Check if file exists
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = file_storage[file_id]
    video_path = file_info["path"]
    
    try:
        # Run detection
        result = model.predict(video_path)
        
        # Add file metadata
        result["file_info"] = {
            "file_id": file_id,
            "filename": file_info["filename"],
            "size_mb": round(file_info["size"] / (1024 * 1024), 2)
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    finally:
        # Optional: Cleanup uploaded file after analysis
        # Uncomment below to auto-delete files
        # if os.path.exists(video_path):
        #     os.remove(video_path)
        #     del file_storage[file_id]
        pass

@app.delete("/api/delete/{file_id}")
async def delete_video(file_id: str):
    """Delete uploaded video file"""
    
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = file_storage[file_id]["path"]
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    del file_storage[file_id]
    
    return {"success": True, "message": "File deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )