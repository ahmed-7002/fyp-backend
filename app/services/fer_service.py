"""
FER-2013 facial emotion recognition service.

Uses the `fer` library (https://pypi.org/project/fer/), which bundles its
own pre-trained Keras CNN - so no training or dataset is required to run
this project. This keeps the same public function signature
(`analyze_frames`) as before, so nothing in routers/fer.py, the frontend,
or the database needs to change.

Pipeline per frame:
  1. Decode the incoming JPEG bytes into an OpenCV BGR image.
  2. Hand the frame to `fer`, which detects the face (using TensorFlow MTCNN
     for high accuracy) and returns a softmax-style probability for each of the
     7 emotions.
  3. Aggregate the per-frame predictions across the whole batch (150-160
     frames) into one averaged result.

Swapping in your own custom-trained model later (e.g. from
training/train_fer_model.py) only requires changing this file - see the
commented-out block at the bottom of _get_detector().
"""
from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import List

import cv2
import numpy as np
from PIL import Image

from app.schemas import FerResult

logger = logging.getLogger("fer_service")

# NOTE: this order matches `fer`'s own emotions dict keys - do not reorder,
# it also matches training/train_fer_model.py's EMOTION_ORDER, so a custom
# model can be dropped in later without touching this list.
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class ModelNotAvailableError(RuntimeError):
    """Raised when the FER detector/model fails to initialize."""


@lru_cache(maxsize=1)
def _get_detector():
    """
    Lazily creates and caches the FER detector on first use, so the API can
    start up even before the `fer` package has downloaded/cached its model.
    """
    try:
        from fer import FER

        # mtcnn=True uses the highly accurate TensorFlow MTCNN face detector 
        # instead of the default OpenCV Haar Cascade. This dramatically improves
        # the frames_analyzed vs frames_captured ratio.
        detector = FER(mtcnn=True)
        logger.info("FER detector initialized (pre-trained model bundled with the `fer` package using MTCNN)")
        return detector
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see error message below
        raise ModelNotAvailableError(
            "Could not initialize the FER emotion detector. Make sure the "
            "`fer`, `mtcnn`, and `tensorflow` packages are installed. "
            f"Original error: {exc}"
        ) from exc

    # --- To use your own custom-trained model instead (advanced/optional) ---
    # from tensorflow.keras.models import load_model
    # from app.config import get_settings
    # return load_model(get_settings().FER_MODEL_PATH)
    # (If you do this, also update _predict_emotions() below to run your
    # model's preprocessing/inference instead of detector.detect_emotions().)


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw upload bytes into an OpenCV BGR numpy array."""
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _predict_emotions(detector, bgr_image: np.ndarray) -> dict | None:
    """
    Runs face detection + emotion recognition on one frame.
    Returns a {emotion_label: probability} dict for the largest detected
    face, or None if no face was found in this frame.
    """
    results = detector.detect_emotions(bgr_image)
    if not results:
        return None

    # `fer` can return multiple faces; pick the largest bounding box as the
    # primary subject (same behavior as the previous Haar-cascade version).
    largest = max(results, key=lambda r: r["box"][2] * r["box"][3])
    return largest["emotions"]  # e.g. {'angry': 0.02, 'disgust': 0.0, ...}


def analyze_frames(frame_bytes_list: List[bytes]) -> FerResult:
    """
    Runs FER inference across every captured frame and aggregates results.

    Aggregation strategy: for each frame with a detected face, take the
    per-emotion probability vector. Average these vectors across all
    analyzed frames, then express each emotion as a percentage. This is
    more informative than a simple majority vote because it captures
    intensity, not just the arg-max label.
    """
    detector = _get_detector()

    frames_captured = len(frame_bytes_list)
    accum = np.zeros(len(EMOTION_LABELS), dtype="float64")
    frames_analyzed = 0

    for raw in frame_bytes_list:
        try:
            bgr = _decode_image(raw)
        except Exception:
            continue  # skip corrupt/undecodable frame, don't fail the whole batch

        emotions = _predict_emotions(detector, bgr)
        if emotions is None:
            continue  # no face detected in this frame

        accum += np.array([emotions[label] for label in EMOTION_LABELS])
        frames_analyzed += 1

    if frames_analyzed == 0:
        # No usable frames at all - return a neutral/empty result rather than
        # crashing; the frontend/UI decides how to communicate this to the user.
        return FerResult(
            frames_captured=frames_captured,
            frames_analyzed=0,
            angry=0.0,
            disgust=0.0,
            fear=0.0,
            happy=0.0,
            sad=0.0,
            surprise=0.0,
            neutral=0.0,
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