"""
DASS-21 scoring engine.

Official scoring procedure:
1. Sum the raw 0-3 answers belonging to each subscale (7 items per subscale).
2. Multiply the sum by 2 (this converts the DASS-21 sum onto the same scale
   as the original DASS-42, which is what the published severity cut-offs use).
3. Map the doubled score onto a severity band.
"""
from app.data.dass21_questions import DEPRESSION_ITEMS, ANXIETY_ITEMS, STRESS_ITEMS
from app.schemas import DassResult

# (max inclusive score, label) - checked in ascending order
_DEPRESSION_BANDS = [(9, "Normal"), (13, "Mild"), (20, "Moderate"), (27, "Severe"), (999, "Extremely Severe")]
_ANXIETY_BANDS = [(7, "Normal"), (9, "Mild"), (14, "Moderate"), (19, "Severe"), (999, "Extremely Severe")]
_STRESS_BANDS = [(14, "Normal"), (18, "Mild"), (25, "Moderate"), (33, "Severe"), (999, "Extremely Severe")]


def _severity(score: int, bands: list[tuple[int, str]]) -> str:
    for max_score, label in bands:
        if score <= max_score:
            return label
    return bands[-1][1]


def _subscale_sum(answers: list[int], item_numbers: list[int]) -> int:
    # item_numbers are 1-based; answers list is 0-based
    return sum(answers[n - 1] for n in item_numbers)


def score_dass21(answers: list[int]) -> DassResult:
    """
    answers: list of exactly 21 ints, each 0-3, in question order (Q1..Q21).
    """
    if len(answers) != 21:
        raise ValueError("DASS-21 requires exactly 21 answers")
    if any(a < 0 or a > 3 for a in answers):
        raise ValueError("Each DASS-21 answer must be between 0 and 3")

    depression_raw = _subscale_sum(answers, DEPRESSION_ITEMS) * 2
    anxiety_raw = _subscale_sum(answers, ANXIETY_ITEMS) * 2
    stress_raw = _subscale_sum(answers, STRESS_ITEMS) * 2

    return DassResult(
        depression_score=depression_raw,
        anxiety_score=anxiety_raw,
        stress_score=stress_raw,
        depression_severity=_severity(depression_raw, _DEPRESSION_BANDS),
        anxiety_severity=_severity(anxiety_raw, _ANXIETY_BANDS),
        stress_severity=_severity(stress_raw, _STRESS_BANDS),
    )
