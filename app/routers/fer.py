"""
Endpoint that receives individually captured webcam frames and returns
aggregated FER-2013 emotion metrics. Kept separate from the final
`/assessments` submission endpoint so the frontend can run video analysis
as its own async step (with its own loading state) before/independently of
the questionnaire.

Two callers, two different frame-count shapes:
  - Video-only mode: always sends ~150-160 frames from a fixed 30-second
    capture window (see VideoAssessment.jsx).
  - Combined mode: captures one frame every ~2 seconds *while the person
    answers the questionnaire in the background*, then stops and submits
    whatever was captured the moment either 155 frames is reached or the
    questionnaire finishes - whichever comes first (see
    useBackgroundCapture.js / CombinedAssessmentFlow.jsx). A fast
    test-taker might finish in well under 150 frames' worth of time, so
    the floor here is intentionally lenient rather than assuming a fixed
    session length.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List

from app.auth import get_current_user
from app.schemas import FerResult
from app.services.fer_service import analyze_frames, ModelNotAvailableError

router = APIRouter(prefix="/api/fer", tags=["fer"])

MIN_FRAMES = 10       # lenient floor - just enough for a meaningful aggregate; Combined mode's
                        # capture duration varies with how fast someone answers the questionnaire
MAX_FRAMES = 200       # sanity ceiling to bound compute/time per request


@router.post("/analyze", response_model=FerResult)
async def analyze(
    frames: List[UploadFile] = File(..., description="Captured JPEG frames (10-200 depending on assessment mode)"),
    _user_id: str = Depends(get_current_user),
):
    if not (MIN_FRAMES <= len(frames) <= MAX_FRAMES):
        raise HTTPException(
            400,
            f"Expected between {MIN_FRAMES} and {MAX_FRAMES} frames, received {len(frames)}.",
        )

    frame_bytes = [await f.read() for f in frames]

    try:
        result = analyze_frames(frame_bytes)
    except ModelNotAvailableError as exc:
        raise HTTPException(503, str(exc)) from exc

    return result