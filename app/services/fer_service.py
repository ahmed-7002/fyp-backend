"""
FER-2013 facial emotion recognition service.

Uses the `fer` library (https://pypi.org/project/fer/), which bundles its
own pre-trained Keras CNN - so no training or dataset is required to run
this project. This keeps the same public function signature
(`analyze_frames`) as before, so nothing in routers/fer.py, the frontend,
or the database needs to change.

Pipeline per frame:
  1. Decode the incoming JPEG bytes into an OpenCV BGR image.
  2. Try face detection + emotion recognition on the frame AS-IS first
     (cheapest path - most frames already have a clear, well-lit face and
     don't need any extra processing).
  3. ONLY if that first attempt finds no face: retry once on a
     CLAHE-contrast-enhanced version of the same frame. This gives frames
     with poor lighting/contrast a genuine second chance without paying
     the CLAHE cost on every frame, most of which don't need it.
  4. Aggregate the per-frame predictions across the whole batch (150-200
     frames) into one averaged result.

IMPORTANT, READ BEFORE RELYING ON THIS TO FIX THE FRAME-RATE VARIANCE:
CLAHE improves detection on frames with a face that's simply poorly lit or
low-contrast. It does NOT put a face in frame that isn't there. Per the
open investigation in the project handover doc (Combined-mode users
naturally looking down to read questions rather than facing the camera),
a meaningful share of "undetected" frames likely have no face in the
crop at all - CLAHE has no effect on those, and no preprocessing step
will. Treat this as a genuine, real improvement for the
lighting/contrast-limited subset of dropped frames, not a full fix for
the underlying variance - re-run the three-session test from the
handover doc to see the actual delta this makes.

Swapping in your own custom-trained model later (e.g. from
training/train_fer_model.py) only requires changing this file - see the
commented-out block at the bottom of _get_detector().
"""
from __future__ import annotations

import gc
import io
import logging
from functools import lru_cache
from typing import List, Optional

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

        # mtcnn=False uses OpenCV's Haar Cascade face detector (fast,
        # CPU-only, no extra native deps). Deliberately NOT switching to
        # mtcnn=True here - that reintroduces the TensorFlow memory-risk
        # category of problem already fought through on Render (see project
        # handover doc, deployment history). The CLAHE fallback below is
        # the low-RAM-safe way to improve Haar Cascade's hit rate instead.
        detector = FER(mtcnn=False)
        logger.info("FER detector initialized (pre-trained model bundled with the `fer` package)")
        return detector
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see error message below
        raise ModelNotAvailableError(
            "Could not initialize the FER emotion detector. Make sure the "
            "`fer` package is installed (pip install fer). "
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


def _apply_clahe(bgr_image: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization, applied to the L
    (lightness) channel only in LAB colour space - not to all 3 BGR
    channels directly, which would distort colour balance and could hurt
    detection rather than help it.

    CLAHE (vs. plain global histogram equalization) works in small tiles
    with a clipped contrast limit, so it boosts local contrast in dim or
    washed-out regions without blowing out already-well-lit areas of the
    same frame - relevant here since webcam frames can be unevenly lit
    (window light on one side of a face, for example).
    """
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def _predict_emotions(detector, bgr_image: np.ndarray) -> Optional[dict]:
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


def _predict_emotions_with_fallback(detector, bgr_image: np.ndarray) -> Optional[dict]:
    """
    Two-attempt detection for a single frame:
      1. Try the frame as-is (cheap - no extra CPU work for the frames
         that already detect cleanly, which is most of them).
      2. Only if that finds no face, retry once on a CLAHE-enhanced copy
         of the same frame, giving low-contrast/poorly-lit frames a real
         second chance before being counted as undetected.

    This keeps CLAHE's CPU cost paid only by the frames that need it,
    rather than on every one of the ~150-200 frames in a session.
    """
    emotions = _predict_emotions(detector, bgr_image)
    if emotions is not None:
        return emotions

    enhanced = _apply_clahe(bgr_image)
    emotions = _predict_emotions(detector, enhanced)

    # `enhanced` is a full-resolution numpy array copy - explicitly drop
    # the reference now rather than waiting for it to fall out of scope
    # naturally, since this function is called up to ~150-200 times per
    # session and each enhanced copy is otherwise the largest short-lived
    # allocation in the loop.
    del enhanced

    return emotions


def analyze_frames(frame_bytes_list: List[bytes]) -> FerResult:
    """
    Runs FER inference across every captured frame and aggregates results.

    Aggregation strategy: for each frame with a detected face, take the
    per-emotion probability vector. Average these vectors across all
    analyzed frames, then express each emotion as a percentage. This is
    more informative than a simple majority vote because it captures
    intensity, not just the arg-max label.

    Memory note: no gc.collect() calls inside the per-frame loop. Python's
    reference-counting GC already frees each frame's decoded numpy array
    the moment its loop iteration ends and nothing else references it -
    calling gc.collect() every iteration would force a full generational
    sweep 150-200 times per session, which costs more CPU time than it
    saves. Instead, a single gc.collect() runs once at the very end of the
    batch (see below), which is where reclaiming memory before the next
    request actually matters on a low-RAM server.
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

        emotions = _predict_emotions_with_fallback(detector, bgr)
        if emotions is None:
            continue  # no face detected in this frame, even after the CLAHE retry

        accum += np.array([emotions[label] for label in EMOTION_LABELS])
        frames_analyzed += 1

        # Explicitly drop the decoded frame now rather than waiting for
        # the next loop iteration to overwrite `bgr` - keeps peak memory
        # bounded to roughly one frame + one CLAHE copy at a time instead
        # of whatever the interpreter happens to still be holding.
        del bgr

    # Single end-of-batch collection: by this point every per-frame
    # numpy array from the loop above is already unreferenced, but on a
    # low-RAM server it's worth forcing that memory back to the OS now,
    # before returning control to the request handler, rather than
    # waiting for Python's normal generational thresholds to trigger it
    # organically at some later, less predictable point.
    gc.collect()

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