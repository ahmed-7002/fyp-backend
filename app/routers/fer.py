"""
Endpoint that receives 150-160 individually captured webcam frames and
returns aggregated FER-2013 emotion metrics. Kept separate from the final
`/assessments` submission endpoint so the frontend can run video analysis
as its own async step (with its own loading state) before/independently of
the questionnaire.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List

from app.auth import get_current_user
from app.schemas import FerResult
from app.services.fer_service import analyze_frames, ModelNotAvailableError

router = APIRouter(prefix="/api/fer", tags=["fer"])

MIN_FRAMES = 100     # sanity floor - well under the 150-160 target, tolerates dropped frames
MAX_FRAMES = 200      # sanity ceiling to bound compute/time per request


@router.post("/analyze", response_model=FerResult)
async def analyze(
    frames: List[UploadFile] = File(..., description="150-160 JPEG frames captured during the assessment"),
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
