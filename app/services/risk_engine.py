"""
Combines DASS-21 severities and FER-2013 dominant emotion into a single,
human-readable overall risk level. This is a simple rule-based heuristic,
NOT a diagnostic algorithm - it exists purely to give the results dashboard
a friendly top-line summary.
"""
from typing import Optional

from app.schemas import DassResult, FerResult

_SEVERITY_WEIGHT = {
    "Normal": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Extremely Severe": 4,
}

_NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}


def compute_overall_risk(
    dass: Optional[DassResult], fer: Optional[FerResult]
) -> tuple[str, str]:
    """Returns (risk_level, summary_text)."""
    score = 0
    parts = []

    if dass:
        max_weight = max(
            _SEVERITY_WEIGHT[dass.depression_severity],
            _SEVERITY_WEIGHT[dass.anxiety_severity],
            _SEVERITY_WEIGHT[dass.stress_severity],
        )
        score += max_weight
        parts.append(
            f"DASS-21 indicates {dass.depression_severity} depression, "
            f"{dass.anxiety_severity} anxiety, and {dass.stress_severity} stress levels."
        )

    if fer and fer.frames_analyzed > 0:
        negative_share = sum(getattr(fer, e) for e in _NEGATIVE_EMOTIONS)
        if negative_share >= 55:
            score += 2
        elif negative_share >= 30:
            score += 1
        parts.append(
            f"Facial emotion analysis across {fer.frames_analyzed} frames shows "
            f"'{fer.dominant_emotion}' as the dominant expression."
        )

    if score >= 5:
        level = "Needs Attention"
    elif score >= 3:
        level = "High"
    elif score >= 1:
        level = "Moderate"
    else:
        level = "Low"

    summary = " ".join(parts) if parts else "No assessment data was provided."
    summary += (
        " This is an automated, question-based screening tool - not a clinical "
        "diagnosis. Please consult a licensed mental health professional for a "
        "full evaluation."
    )
    return level, summary
