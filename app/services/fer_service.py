"""
FER-2013 facial emotion recognition service.

Uses the `fer` library to classify emotions, but uses Google's ultra-lightweight 
`mediapipe` library to detect the faces first. This acts as the perfect middle-ground:
high accuracy face detection without the heavy server-crashing RAM usage of MTCNN.
"""
from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

from app.schemas import FerResult

logger = logging.getLogger("fer_service")

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 1. Initialize MediaPipe's ultra-lightweight face detector once at startup
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0,  # 0 is heavily optimized for short-range webcams
    min_detection_confidence=0.5
)

class ModelNotAvailableError(RuntimeError):
    """Raised when the FER detector/model fails to initialize."""

@lru_cache(maxsize=1)
def _get_detector():
    """Lazily creates and caches the FER detector on first use."""
    try:
        from fer import FER

        # We keep mtcnn=False to prevent Render from crashing (OOM).
        # We rely on MediaPipe to find the faces instead!
        detector = FER(mtcnn=False)
        logger.info("FER emotion classifier initialized successfully.")
        return detector
    except Exception as exc: 
        raise ModelNotAvailableError(
            "Could not initialize the FER emotion detector. Make sure the "
            "`fer`, `mediapipe`, and `tensorflow` packages are installed. "
            f"Original error: {exc}"
        ) from exc


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw upload bytes into an OpenCV BGR numpy array."""
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _predict_emotions(detector, bgr_image: np.ndarray) -> dict | None:
    """
    1. Uses MediaPipe to locate the face instantly.
    2. Crops the face out.
    3. Passes the clean, cropped face to `fer` to predict the emotion.
    """
    # Ask MediaPipe to find the face (lightning fast, high accuracy)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(rgb_image)
    
    if not results.detections:
        return None
        
    # Get the bounding box of the most prominent face
    detection = max(results.detections, key=lambda d: d.score[0])
    bbox = detection.location_data.relative_bounding_box
    
    h, w, _ = bgr_image.shape
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    box_w = int(bbox.width * w)
    box_h = int(bbox.height * h)
    
    # Add a generous 20% margin around the face. 
    # This ensures the chin/forehead aren't cut off and allows `fer` to see it clearly.
    margin_x = int(box_w * 0.2)
    margin_y = int(box_h * 0.2)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + box_w + margin_x)
    y2 = min(h, y + box_h + margin_y)
    
    cropped_face = bgr_image[y1:y2, x1:x2]
    
    # Safety check: skip if the crop is somehow empty/corrupted
    if cropped_face.size == 0:
        return None
        
    # Pass the perfectly clean, cropped face to the FER library
    fer_results = detector.detect_emotions(cropped_face)
    
    if fer_results:
        # Return the emotions dict from the first recognized face
        largest = max(fer_results, key=lambda r: r["box"][2] * r["box"][3])
        return largest["emotions"]
        
    return None


def analyze_frames(frame_bytes_list: List[bytes]) -> FerResult:
    """Runs inference across every captured frame and aggregates results."""
    detector = _get_detector()

    frames_captured = len(frame_bytes_list)
    accum = np.zeros(len(EMOTION_LABELS), dtype="float64")
    frames_analyzed = 0

    for raw in frame_bytes_list:
        try:
            bgr = _decode_image(raw)
        except Exception:
            continue  

        emotions = _predict_emotions(detector, bgr)
        if emotions is None:
            continue  

        accum += np.array([emotions[label] for label in EMOTION_LABELS])
        frames_analyzed += 1

    if frames_analyzed == 0:
        return FerResult(
            frames_captured=frames_captured,
            frames_analyzed=0,
            angry=0.0, disgust=0.0, fear=0.0,
            happy=0.0, sad=0.0, surprise=0.0, neutral=0.0,
            dominant_emotion="undetermined",
        )

    averaged = accum / frames_analyzed
    percentages = {label: round(float(val) * 100, 2) for label, val in zip(EMOTION_LABELS, averaged)}
    dominant = max(percentages, key=percentages.get)

    return FerResult(
        frames_captured=frames_captured,
        frames_analyzed=frames_analyzed,
        angry=percentages["angry"],
        disgust=percentages["disgust"],
        fear=percentages["fear"],
        happy=percentages["happy"],
        sad=percentages["sad"],
        surprise=percentages["surprise"],
        neutral=percentages["neutral"],
        dominant_emotion=dominant,
    )