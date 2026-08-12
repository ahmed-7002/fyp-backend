"""
FER-2013 facial emotion recognition service.

PIPELINE (v3 - MediaPipe-assisted face detection):
  1. Decode the incoming JPEG bytes into an OpenCV BGR image.
  2. PRIMARY PATH: run Google MediaPipe's BlazeFace detector on the frame.
     MediaPipe is dramatically more tolerant of head-angle, partial
     occlusion, and motion blur than the Haar Cascade `fer` uses
     internally - this directly targets the documented root cause of the
     frame-analysis-rate variance (Combined-mode users naturally looking
     down at the questionnaire rather than facing the camera square-on).
     If MediaPipe finds a face, we crop tightly around it (with padding)
     and hand that crop to `fer` - since the crop is now mostly just a
     face, `fer`'s own internal Haar Cascade finds it almost every time.
  3. FALLBACK PATH (used when MediaPipe is unavailable, fails to load, or
     genuinely finds no face in this frame): the original two-attempt
     pipeline - try the full raw frame with `fer` directly, then retry
     once on a CLAHE-contrast-enhanced version of the same frame.
  4. Aggregate the per-frame predictions across the whole batch into one
     averaged result. Same public signature (`analyze_frames`) as before,
     so nothing in routers/fer.py, the frontend, or the database changes.

IMPORTANT - READ BEFORE DEPLOYING:

(a) MediaPipe's current Python package (0.10.x+) no longer ships the old
    `mp.solutions.face_detection` API you'll see in a lot of tutorials -
    only the newer "Tasks" API, which requires a small separately-hosted
    .tflite model file. You must download this once and commit it into
    the repo (see _MODEL_PATH below and the setup note there) - don't
    rely on downloading it at request time or container start time.

(b) libGL risk (same category of problem that killed the FastAPI Cloud
    migration - see project handover doc, deployment history): some
    MediaPipe builds pull in OpenCV/OpenGL-adjacent native dependencies
    that can throw `ImportError: libGL.so.1: cannot open shared object
    file` in a minimal headless container. Render lets you add apt
    packages via a build command (unlike FastAPI Cloud's buildpack), so
    if you hit this on deploy, add `libgl1` to Render's build step rather
    than assuming it's a dead end this time. TEST A DEPLOY ON RENDER
    BEFORE RELYING ON THIS IN PRODUCTION.

(c) Graceful degradation, on purpose: if the model file is missing or
    MediaPipe fails to initialize for ANY reason, this file logs a clear
    warning once and silently falls back to the pre-MediaPipe pipeline
    for every frame, rather than raising and breaking the whole FER
    feature. Worst case if something's misconfigured, you're back to
    current (already-working) behaviour, not a broken deploy.

Swapping in your own custom-trained emotion model later (as opposed to
the face DETECTOR swapped here) only requires changing _get_fer_detector()
- see the commented-out block at the bottom of that function.
"""
from __future__ import annotations

import gc
import io
import logging
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.schemas import FerResult

logger = logging.getLogger("fer_service")

# NOTE: this order matches `fer`'s own emotions dict keys - do not reorder,
# it also matches training/train_fer_model.py's EMOTION_ORDER, so a custom
# model can be dropped in later without touching this list.
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# --- MediaPipe face-detector model file -----------------------------------
# Download ONCE and commit this file into the repo (it's ~230KB, safe to
# check in) - do not fetch it at request time or container startup.
#   curl -L -o app/ml_models/blaze_face_short_range.tflite \
#     https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
# "short_range" is the correct model for a face filling most of the frame
# at close range (webcam / phone-near-face use case) - "full_range" is
# better for a face far from camera and would be the wrong choice here.
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ml_models",
    "blaze_face_short_range.tflite",
)

# How much padding to add around MediaPipe's tight bounding box before
# cropping, as a fraction of the box's own width/height. Some headroom
# around the face (not a pixel-perfect crop) gives `fer`'s internal
# detector a bit of context to work with, rather than an unnaturally
# tight crop that could itself be harder to detect.
_CROP_PADDING_RATIO = 0.35

# Below this confidence, treat MediaPipe as "no face found" and fall
# through to the Haar+CLAHE path rather than trusting a low-confidence box.
_MIN_DETECTION_CONFIDENCE = 0.5


class ModelNotAvailableError(RuntimeError):
    """Raised when the FER emotion-classification detector fails to
    initialize. NOT raised for MediaPipe failures - those degrade
    gracefully instead (see module docstring, point (c))."""


@lru_cache(maxsize=1)
def _get_fer_detector():
    """
    Lazily creates and caches the `fer` emotion-classification detector on
    first use. This is unchanged from the pre-MediaPipe version - MediaPipe
    only replaces the FACE-FINDING step upstream of this, not emotion
    classification itself.
    """
    try:
        from fer import FER

        # mtcnn=False uses OpenCV's Haar Cascade internally. Deliberately
        # kept False even with MediaPipe in front of it - MediaPipe already
        # does the hard face-finding work, so `fer` only needs to re-detect
        # a face that's already filling most of the cropped frame, which
        # Haar handles fine. Switching this to True would reintroduce the
        # TensorFlow memory-risk category of problem already fought
        # through on Render (see project handover doc).
        detector = FER(mtcnn=False)
        logger.info("FER emotion detector initialized (pre-trained model bundled with the `fer` package)")
        return detector
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see error message below
        raise ModelNotAvailableError(
            "Could not initialize the FER emotion detector. Make sure the "
            "`fer` package is installed (pip install fer). "
            f"Original error: {exc}"
        ) from exc

    # --- To use your own custom-trained EMOTION model instead (optional) ---
    # from tensorflow.keras.models import load_model
    # from app.config import get_settings
    # return load_model(get_settings().FER_MODEL_PATH)
    # (If you do this, also update _predict_emotions() below to run your
    # model's preprocessing/inference instead of detector.detect_emotions().)


@lru_cache(maxsize=1)
def _get_mediapipe_detector():
    """
    Lazily creates and caches the MediaPipe BlazeFace detector.

    Returns None (not an exception) if MediaPipe or its model file isn't
    available, so callers fall back to the Haar+CLAHE pipeline instead of
    the whole service breaking - see module docstring, point (c).
    lru_cache still applies to the None result, so a missing-model warning
    is logged once, not once per frame.
    """
    if not os.path.exists(_MODEL_PATH):
        logger.warning(
            "MediaPipe face-detector model not found at %s - falling back "
            "to the Haar Cascade + CLAHE pipeline for all frames. See the "
            "setup note above _MODEL_PATH in fer_service.py to enable "
            "MediaPipe-assisted detection.",
            _MODEL_PATH,
        )
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceDetector,
            FaceDetectorOptions,
            RunningMode,
        )

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=_MIN_DETECTION_CONFIDENCE,
        )
        detector = FaceDetector.create_from_options(options)
        logger.info("MediaPipe face detector initialized from %s", _MODEL_PATH)
        return detector
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see point (c)
        logger.warning(
            "MediaPipe face detector failed to initialize (%s) - falling "
            "back to the Haar Cascade + CLAHE pipeline for all frames.",
            exc,
        )
        return None


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    """Decode raw upload bytes into an OpenCV BGR numpy array."""
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _detect_face_box(mp_detector, bgr_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Runs MediaPipe face detection on one frame.
    Returns (x, y, w, h) in pixel coordinates for the largest detected
    face, or None if no face was found above the confidence threshold.
    """
    import mediapipe as mp

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    result = mp_detector.detect(mp_image)
    if not result.detections:
        return None

    def _area(detection) -> int:
        bb = detection.bounding_box
        return bb.width * bb.height

    largest = max(result.detections, key=_area)
    bb = largest.bounding_box
    return bb.origin_x, bb.origin_y, bb.width, bb.height


def _crop_with_padding(
    bgr_image: np.ndarray, box: Tuple[int, int, int, int], padding_ratio: float
) -> np.ndarray:
    """Crops `bgr_image` to `box` expanded by `padding_ratio` on each side,
    clamped to the image's actual bounds so padding near an edge doesn't
    request pixels outside the frame."""
    x, y, w, h = box
    img_h, img_w = bgr_image.shape[:2]

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return bgr_image[y1:y2, x1:x2]


def _apply_clahe(bgr_image: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization, applied to the L
    (lightness) channel only in LAB colour space - not to all 3 BGR
    channels directly, which would distort colour balance and could hurt
    detection rather than help it.
    """
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def _predict_emotions(fer_detector, bgr_image: np.ndarray) -> Optional[dict]:
    """
    Runs emotion recognition on one frame/crop via `fer`.
    Returns a {emotion_label: probability} dict for the largest detected
    face, or None if `fer`'s own internal detector found no face.
    """
    if bgr_image.size == 0:
        return None  # degenerate/empty crop, shouldn't normally happen

    results = fer_detector.detect_emotions(bgr_image)
    if not results:
        return None

    largest = max(results, key=lambda r: r["box"][2] * r["box"][3])
    return largest["emotions"]


def _analyze_single_frame(fer_detector, mp_detector, bgr_image: np.ndarray) -> Optional[dict]:
    """
    Full per-frame strategy, in priority order:
      1. MediaPipe finds the face -> crop tightly -> `fer` classifies the
         crop. This is the primary path when MediaPipe is available.
      2. MediaPipe unavailable, or found no face in this frame -> fall
         back to the original pipeline: `fer` on the raw full frame, then
         retry once on a CLAHE-enhanced version if that also fails.
    """
    if mp_detector is not None:
        box = _detect_face_box(mp_detector, bgr_image)
        if box is not None:
            crop = _crop_with_padding(bgr_image, box, _CROP_PADDING_RATIO)
            emotions = _predict_emotions(fer_detector, crop)
            del crop
            if emotions is not None:
                return emotions
            # MediaPipe found a face but `fer` still couldn't classify the
            # crop (rare - e.g. extreme motion blur) - fall through to the
            # full-frame fallback below rather than giving up immediately.

    # Fallback path (also the ONLY path when MediaPipe isn't available).
    emotions = _predict_emotions(fer_detector, bgr_image)
    if emotions is not None:
        return emotions

    enhanced = _apply_clahe(bgr_image)
    emotions = _predict_emotions(fer_detector, enhanced)
    del enhanced

    return emotions


def analyze_frames(frame_bytes_list: List[bytes]) -> FerResult:
    """
    Runs FER inference across every captured frame and aggregates results.

    Aggregation strategy: for each frame with a detected face, take the
    per-emotion probability vector. Average these vectors across all
    analyzed frames, then express each emotion as a percentage.

    Memory note: no gc.collect() calls inside the per-frame loop - Python's
    reference-counting GC already frees each frame's decoded arrays as soon
    as their loop iteration ends. A single gc.collect() runs once at the
    end of the whole batch instead, which is where reclaiming memory
    actually matters on a low-RAM server.
    """
    fer_detector = _get_fer_detector()
    mp_detector = _get_mediapipe_detector()  # None if unavailable - see point (c) above

    frames_captured = len(frame_bytes_list)
    accum = np.zeros(len(EMOTION_LABELS), dtype="float64")
    frames_analyzed = 0

    for raw in frame_bytes_list:
        try:
            bgr = _decode_image(raw)
        except Exception:
            continue  # skip corrupt/undecodable frame, don't fail the whole batch

        emotions = _analyze_single_frame(fer_detector, mp_detector, bgr)
        if emotions is None:
            del bgr
            continue  # no face detected in this frame, even after all fallbacks

        accum += np.array([emotions[label] for label in EMOTION_LABELS])
        frames_analyzed += 1
        del bgr

    gc.collect()

    if frames_analyzed == 0:
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