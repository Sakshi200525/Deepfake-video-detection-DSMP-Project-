import os
import torch
import torch.nn as nn
import numpy as np
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Union

# Local Imports
from config import settings
from processors.video_processor import VideoProcessor
from processors.audio_processor import AudioProcessor
from utils.cache import RedisCache  # Assuming RedisCache is used

class DeepfakeDetector:
    def __init__(self, model_path: str, cache: Optional[RedisCache] = None):
        self.model_path = model_path
        self.device = settings.DEVICE
        self.cache = cache

        image_size_arg = settings.IMAGE_SIZE
        if isinstance(image_size_arg, int):
            image_size_arg = [image_size_arg, image_size_arg]

        self.video_processor = VideoProcessor(
            target_size=image_size_arg,
            max_frames=settings.MAX_FRAMES
        )

        audio_sample_rate = getattr(settings, 'AUDIO_SAMPLE_RATE', 44100)
        max_audio_features = getattr(settings, 'MAX_AUDIO_FEATURES', 128)

        self.audio_processor = AudioProcessor(
            sample_rate=audio_sample_rate,
            max_features=max_audio_features
        )

        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.model = self._load_model()
        print(f"✅ DeepfakeDetector initialized. Model: {model_path}")

    def _load_model(self) -> Union[nn.Module, str]:
        try:
            import timm
            model = timm.create_model(
                'efficientnet_b4.ra2_in1k',
                pretrained=True,
                num_classes=1
            )
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def _run_inference(self, face_tensors: torch.Tensor) -> float:
        try:
            with torch.no_grad():
                face_tensors = face_tensors.to(self.device)
                outputs = self.model(face_tensors)
                probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
                return float(np.mean(probabilities))
        except Exception as e:
            return 0.5

    async def detect_deepfake(self, video_data: bytes, skip_cache: bool = False) -> Dict[str, Any]:
        if self.cache and not skip_cache:
            cached_result = self.cache.get_result(video_data)
            if cached_result:
                return cached_result

        loop = asyncio.get_event_loop()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video_data)
            tmp_path = tmp_file.name

        try:
            visual_future = loop.run_in_executor(self.executor, self.video_processor.process_video, tmp_path)
            audio_future = loop.run_in_executor(self.executor, self.audio_processor.process_audio, tmp_path)

            face_tensors, num_frames_analyzed = await visual_future
            audio_features, has_audio = await audio_future

            # 3. Visual Score (0.0 - 1.0)
            if face_tensors is not None and face_tensors.shape[0] > 0:
                visual_score = self._run_inference(face_tensors)
            else:
                visual_score = 0.5
                num_frames_analyzed = 0

            # 4. --- AUDIO SCORE WITH WEIGHTED LOGIC ---
            if audio_features:
                # Key features extract karke 70% weight Spectral Flatness ko diya
                mfcc_anomaly = audio_features.get("mfcc_std_anomaly", 0.5)
                flatness_anomaly = audio_features.get("spectral_flatness_score", 0.5)
                audio_score = (flatness_anomaly * 0.7) + (mfcc_anomaly * 0.3)
            else:
                audio_score = 0.5

            # 5. --- AGGREGATION & PERCENTAGE FLIP ---
            visual_weight = 0.7
            audio_weight = 0.3
            raw_confidence = (visual_score * visual_weight) + (audio_score * audio_weight)

            # Isse UI par 50% ki jagah actual authenticity dikhegi
            if raw_confidence > settings.DEEPFAKE_THRESHOLD:
                verdict = "DEEPFAKE VIDEO"
                final_percent = raw_confidence * 100
                display_visual = visual_score * 100
                display_audio = audio_score * 100
            else:
                verdict = "AUTHENTIC VIDEO"
                final_percent = (1.0 - raw_confidence) * 100
                display_visual = (1.0 - visual_score) * 100
                display_audio = (1.0 - audio_score) * 100

            result = {
                "verdict": verdict,
                "confidence": final_percent,           # UI: Final Confidence
                "visual_score": display_visual,       # UI: Visual Score
                "audio_anomalies": display_audio,     # UI: Audio Anomalies
                "compression_artifacts": 50.0,
                "frames_analyzed": num_frames_analyzed,
                "has_audio": has_audio,
                "is_deepfake": raw_confidence > settings.DEEPFAKE_THRESHOLD,
                "message": "Analysis complete."
            }

            if self.cache:
                self.cache.set_result(video_data, result)

            return result

        except Exception as e:
            return {"verdict": "ERROR", "message": str(e)}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)