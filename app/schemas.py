"""
Pydantic request/response schemas.
"""
import uuid
from datetime import datetime
from typing import Optional, List, Literal

from pydantic import BaseModel, Field, conint


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------
class OnboardingIn(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=120)
    age: conint(ge=10, le=120)
    gender: Literal["male", "female", "non_binary", "prefer_not_to_say"]


# ---------------------------------------------------------------------------
# DASS-21
# ---------------------------------------------------------------------------
class DassAnswers(BaseModel):
    """21 answers, each scored 0 (Did not apply to me at all) - 3 (Applied to me very much)."""

    answers: List[conint(ge=0, le=3)] = Field(..., min_length=21, max_length=21)


class DassResult(BaseModel):
    depression_score: int
    anxiety_score: int
    stress_score: int
    depression_severity: str
    anxiety_severity: str
    stress_severity: str


# ---------------------------------------------------------------------------
# FER-2013 video analysis
# ---------------------------------------------------------------------------
class FerResult(BaseModel):
    frames_captured: int
    frames_analyzed: int
    angry: float
    disgust: float
    fear: float
    happy: float
    sad: float
    surprise: float
    neutral: float
    dominant_emotion: str


# ---------------------------------------------------------------------------
# Final submission (persists a full row in `assessments`)
# ---------------------------------------------------------------------------
class AssessmentSubmitIn(BaseModel):
    clerk_user_id: str
    full_name: str
    age: int
    gender: Literal["male", "female", "non_binary", "prefer_not_to_say"]
    assessment_mode: Literal["questionnaire", "video", "combined"]
    dass_answers: Optional[List[conint(ge=0, le=3)]] = None
    fer_result: Optional[FerResult] = None


class AssessmentSubmitOut(BaseModel):
    id: uuid.UUID
    dass_result: Optional[DassResult] = None
    fer_result: Optional[FerResult] = None
    final_risk_level: str
    final_summary: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Profile / history (list of a user's past sessions)
# ---------------------------------------------------------------------------
class AssessmentSummary(BaseModel):
    """Lightweight row for the profile page's session list - full detail is
    fetched separately (GET /api/assessments/{id}) only when a session is
    expanded, so the list itself stays fast even with many past sessions."""

    id: uuid.UUID
    assessment_mode: Literal["questionnaire", "video", "combined"]
    final_risk_level: str
    created_at: datetime

    class Config:
        from_attributes = True