"""
Pydantic request/response schemas.
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Literal

from pydantic import BaseModel, Field, conint, field_validator


def _is_valid_name(value: str) -> bool:
    """
    True if `value` contains only letters/digits (any language - .isalnum()
    is Unicode-aware) plus single spaces, hyphens, and apostrophes, with no
    leading/trailing separator and no double-separators back to back.
    Deliberately implemented as a plain character check rather than a
    \\p{L}-style regex, since Python's built-in `re` module does not support
    that Unicode property syntax (only the third-party `regex` package
    does) - using it here would raise re.error at import time.
    """
    if not value:
        return False
    allowed_separators = " '-"
    if value[0] in allowed_separators or value[-1] in allowed_separators:
        return False
    prev_was_separator = False
    for ch in value:
        if ch in allowed_separators:
            if prev_was_separator:
                return False
            prev_was_separator = True
        elif ch.isalnum():
            prev_was_separator = False
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------
class OnboardingIn(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=120)
    age: conint(ge=18, le=120)
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
    full_name: str = Field(..., min_length=1, max_length=120)
    age: conint(ge=18, le=120)
    gender: Literal["male", "female", "non_binary", "prefer_not_to_say"]
    assessment_mode: Literal["questionnaire", "video", "combined"]
    dass_answers: Optional[List[conint(ge=0, le=3)]] = None
    fer_result: Optional[FerResult] = None

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not _is_valid_name(cleaned):
            raise ValueError(
                "full_name may only contain letters, numbers, single spaces, hyphens, and apostrophes"
            )
        return cleaned


class AssessmentSubmitOut(BaseModel):
    id: uuid.UUID
    dass_result: Optional[DassResult] = None
    fer_result: Optional[FerResult] = None
    final_risk_level: str
    final_summary: str
    # Rule-based, bilingual, personalized suggestions - computed fresh on
    # every response (see app/services/risk_engine.py), never persisted to
    # the database. default_factory=list (not `= []`) is used deliberately:
    # a bare mutable default would be shared across every instance of this
    # model, a classic Python pitfall Pydantic itself guards against, but
    # being explicit here documents the intent.
    actionable_tips: List[Dict[str, str]] = Field(default_factory=list)
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