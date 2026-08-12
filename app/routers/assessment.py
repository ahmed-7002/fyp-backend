import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.data.dass21_questions import DASS21_QUESTIONS, ANSWER_LABELS
from app.database import get_db
from app.models import Assessment
from app.schemas import (
    AssessmentSubmitIn,
    AssessmentSubmitOut,
    AssessmentSummary,
    DassResult,
    FerResult,
)
from app.services.dass_scoring import score_dass21
from app.services.risk_engine import compute_overall_risk, generate_actionable_tips

router = APIRouter(prefix="/api", tags=["assessment"])


def _build_response(
    record: Assessment,
    dass_result: DassResult | None,
    fer_result: FerResult | None,
    tips: list[dict],
) -> AssessmentSubmitOut:
    """Shared response shape for both POST /api/assessments (fresh
    submission) and GET /api/assessments/{id} (re-viewed later) - was
    previously duplicated identically in both endpoints."""
    return AssessmentSubmitOut(
        id=record.id,
        dass_result=dass_result,
        fer_result=fer_result,
        final_risk_level=record.final_risk_level,
        final_summary=record.final_summary,
        actionable_tips=tips,
        created_at=record.created_at,
    )


@router.get("/dass21/questions")
def get_questions():
    return {
        "answer_labels": ANSWER_LABELS,
        "questions": [{"number": i + 1, "text": q} for i, q in enumerate(DASS21_QUESTIONS)],
    }


@router.post("/assessments", response_model=AssessmentSubmitOut)
def submit_assessment(
    payload: AssessmentSubmitIn,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    if payload.clerk_user_id != user_id:
        raise HTTPException(403, "clerk_user_id does not match authenticated user")

    needs_dass = payload.assessment_mode in ("questionnaire", "combined")
    needs_fer = payload.assessment_mode == "video"

    if needs_dass and not payload.dass_answers:
        raise HTTPException(400, "dass_answers is required for this assessment mode")
    if needs_fer and not payload.fer_result:
        raise HTTPException(400, "fer_result is required for this assessment mode")

    dass_result: DassResult | None = None
    if payload.dass_answers:
        dass_result = score_dass21(payload.dass_answers)

    fer_result: FerResult | None = payload.fer_result

    # Age and gender both come straight from the onboarding step of the
    # submission payload (the same values that already get persisted onto
    # the Assessment record below) - there's no separate lookup needed
    # since the user provides both as part of every submission. Age is cast
    # defensively in case it ever arrives as a numeric string from an older
    # client; gender is passed through as-is (risk_engine normalises it,
    # and treats "Prefer not to say" / anything unrecognised as unknown).
    age: int | None = int(payload.age) if payload.age is not None else None
    gender: str | None = payload.gender

    # compute_overall_risk now returns a 3-tuple - risk level, summary, and
    # the rule-based bilingual tips generated from these same DASS/FER
    # values plus age and gender. Tips are NOT stored on `record` below and
    # NOT part of models.py/schema.sql - they're only ever attached to the
    # API response. Neither `age` nor `gender` ever affects
    # final_risk_level/final_summary - both only affect tip WORDING (see
    # risk_engine.py docstring).
    risk_level, summary, tips = compute_overall_risk(dass_result, fer_result, age, gender)

    record = Assessment(
        id=uuid.uuid4(),
        clerk_user_id=payload.clerk_user_id,
        full_name=payload.full_name,
        # Use the same normalised `age`/`gender` locals that were just fed
        # into compute_overall_risk() above, not payload.age/payload.gender
        # directly - keeps what's persisted and what generated these tips
        # identical, including the defensive int-cast on age.
        age=age,
        gender=gender,
        assessment_mode=payload.assessment_mode,
        final_risk_level=risk_level,
        final_summary=summary,
    )

    if payload.dass_answers:
        for i, ans in enumerate(payload.dass_answers, start=1):
            setattr(record, f"dass_q{i}", ans)
        record.dass_depression_score = dass_result.depression_score
        record.dass_anxiety_score = dass_result.anxiety_score
        record.dass_stress_score = dass_result.stress_score
        record.dass_depression_severity = dass_result.depression_severity
        record.dass_anxiety_severity = dass_result.anxiety_severity
        record.dass_stress_severity = dass_result.stress_severity

    if fer_result:
        record.fer_frames_captured = fer_result.frames_captured
        record.fer_frames_analyzed = fer_result.frames_analyzed
        record.fer_angry = fer_result.angry
        record.fer_disgust = fer_result.disgust
        record.fer_fear = fer_result.fear
        record.fer_happy = fer_result.happy
        record.fer_sad = fer_result.sad
        record.fer_surprise = fer_result.surprise
        record.fer_neutral = fer_result.neutral
        record.fer_dominant_emotion = fer_result.dominant_emotion

    db.add(record)
    db.commit()
    db.refresh(record)

    return _build_response(record, dass_result, fer_result, tips)


@router.get("/assessments", response_model=list[AssessmentSummary])
def list_assessments(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """Returns every past session for the signed-in user, most recent first.

    Note: anonymized (deleted) sessions have clerk_user_id set to NULL,
    so they never match this filter and correctly stop appearing here
    for anyone, without needing any extra "is_deleted" flag."""
    records = (
        db.query(Assessment)
        .filter(Assessment.clerk_user_id == user_id)
        .order_by(Assessment.created_at.desc())
        .all()
    )
    return records


@router.get("/assessments/{assessment_id}", response_model=AssessmentSubmitOut)
def get_assessment(
    assessment_id: uuid.UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    record = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    if not record:
        raise HTTPException(404, "Assessment not found")
    if record.clerk_user_id != user_id:
        raise HTTPException(403, "Not authorized to view this assessment")

    dass_result = None
    if record.dass_depression_score is not None:
        dass_result = DassResult(
            depression_score=record.dass_depression_score,
            anxiety_score=record.dass_anxiety_score,
            stress_score=record.dass_stress_score,
            depression_severity=record.dass_depression_severity,
            anxiety_severity=record.dass_anxiety_severity,
            stress_severity=record.dass_stress_severity,
        )

    fer_result = None
    if record.fer_frames_captured is not None:
        fer_result = FerResult(
            frames_captured=record.fer_frames_captured,
            frames_analyzed=record.fer_frames_analyzed,
            angry=record.fer_angry,
            disgust=record.fer_disgust,
            fear=record.fer_fear,
            happy=record.fer_happy,
            sad=record.fer_sad,
            surprise=record.fer_surprise,
            neutral=record.fer_neutral,
            dominant_emotion=record.fer_dominant_emotion,
        )

    # Age and gender are both read directly off the already-persisted
    # record (they were stored at submission time above, straight from the
    # onboarding payload) - no extra lookup against the auth/user object is
    # needed here either.
    age: int | None = record.age
    gender: str | None = record.gender

    # Tips aren't stored - they're recomputed here from the same saved
    # DASS/FER/age/gender values, using the standalone helper directly
    # rather than compute_overall_risk() (which would needlessly recompute
    # final_risk_level/final_summary, both of which are already stored on
    # `record` and read directly below).
    tips = generate_actionable_tips(dass_result, fer_result, age, gender)

    return _build_response(record, dass_result, fer_result, tips)


@router.delete("/assessments/{assessment_id}", status_code=204)
def delete_assessment(
    assessment_id: uuid.UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """
    "Deletes" a session via anonymization, not a hard SQL delete: the two
    personally-identifying columns (clerk_user_id, full_name) are wiped to
    NULL, while age, gender, every DASS-21 answer/score, and every FER
    metric are left fully intact.

    Note: `age` and `gender` are deliberately kept (not wiped) here, same
    as before - this means tip generation for an anonymized session still
    tailors by age/gender if it's ever re-viewed, since neither is
    personally identifying in the way clerk_user_id/full_name are.
    """
    record = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    if not record:
        raise HTTPException(404, "Assessment not found")
    if record.clerk_user_id != user_id:
        raise HTTPException(403, "Not authorized to delete this assessment")

    record.clerk_user_id = None
    record.full_name = None
    db.commit()
    return None