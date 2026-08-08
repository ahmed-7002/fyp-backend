"""
Combines DASS-21 severities and FER-2013 dominant emotion into a single,
human-readable overall risk level, plus a set of bilingual, rule-based
"actionable tips" - short, specific suggestions derived purely from
already-computed DASS severities and FER percentages.

Nothing here is a trained model and nothing here is persisted to the
database: tips are computed fresh, on the fly, every time a result is
viewed (both right after submission and later from the Profile page),
consistent with the rest of this file's existing design philosophy -
a transparent, auditable rule engine rather than an opaque model, so an
examiner (or a user) can trace exactly why a given tip appeared.
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

# Severity tiers treated as "elevated but self-help-appropriate" vs.
# "elevated enough to specifically suggest professional support" - used by
# the tip rules below. Kept as a shared constant so the DASS-tip logic for
# all three subscales stays consistent with each other.
_MILD_TIER = ("Mild", "Moderate")
_HIGH_TIER = ("Severe", "Extremely Severe")


def compute_overall_risk(
    dass: Optional[DassResult], fer: Optional[FerResult]
) -> tuple[str, str, list[dict]]:
    """Returns (risk_level, summary_text, actionable_tips)."""
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

    tips = generate_actionable_tips(dass, fer)

    return level, summary, tips


def _dass_tips(dass: Optional[DassResult]) -> list[dict]:
    """One tip per DASS-21 subscale that is at least Mild, with distinct
    wording for the "self-help appropriate" tier vs. the "specifically
    suggest professional support" tier."""
    tips: list[dict] = []
    if not dass:
        return tips

    if dass.depression_severity in _MILD_TIER:
        tips.append({
            "en": "Try setting one small, achievable task for today, even something as "
                  "simple as a short walk - depression often makes starting anything "
                  "feel harder than it actually is.",
            "ur": "آج کے لیے ایک چھوٹا اور قابلِ حصول کام مقرر کریں، چاہے وہ صرف ایک "
                  "مختصر واک ہی کیوں نہ ہو - ڈپریشن میں کسی بھی کام کا آغاز حقیقت سے "
                  "زیادہ مشکل محسوس ہوتا ہے۔",
        })
    elif dass.depression_severity in _HIGH_TIER:
        tips.append({
            "en": "Your responses suggest a significant level of low mood. Consider "
                  "reaching out to a trusted person or a mental health professional "
                  "soon - you don't have to manage this alone.",
            "ur": "آپ کے جوابات ظاہر کرتے ہیں کہ آپ کا مزاج کافی حد تک متاثر ہے۔ کسی "
                  "قابلِ اعتماد شخص یا ذہنی صحت کے ماہر سے جلد رابطہ کرنے پر غور کریں - "
                  "آپ کو یہ اکیلے نہیں سنبھالنا۔",
        })

    if dass.anxiety_severity in _MILD_TIER:
        tips.append({
            "en": "When anxiety builds up, try the 5-4-3-2-1 grounding technique: name "
                  "5 things you can see, 4 you can touch, 3 you can hear, 2 you can "
                  "smell, and 1 you can taste.",
            "ur": "جب بے چینی بڑھے تو 5-4-3-2-1 گراؤنڈنگ تکنیک آزمائیں: 5 چیزیں جو آپ "
                  "دیکھ سکتے ہیں، 4 جنہیں چھو سکتے ہیں، 3 جو سن سکتے ہیں، 2 جن کی خوشبو "
                  "محسوس کر سکتے ہیں، اور 1 جسے چکھ سکتے ہیں، ان کے نام لیں۔",
        })
    elif dass.anxiety_severity in _HIGH_TIER:
        tips.append({
            "en": "Your anxiety responses are in a higher range. Alongside grounding "
                  "techniques, it may help to talk to a mental health professional "
                  "about what you're experiencing.",
            "ur": "آپ کی بے چینی کے جوابات زیادہ سطح پر ہیں۔ گراؤنڈنگ تکنیکوں کے ساتھ "
                  "ساتھ، ذہنی صحت کے ماہر سے اپنی صورتحال پر بات کرنا مفید ہو سکتا ہے۔",
        })

    if dass.stress_severity in _MILD_TIER:
        tips.append({
            "en": "Build in short, deliberate breaks through your day - even five "
                  "minutes away from a task can meaningfully lower stress before it "
                  "builds up further.",
            "ur": "اپنے دن میں مختصر اور جان بوجھ کر وقفے شامل کریں - کسی کام سے صرف "
                  "پانچ منٹ دور رہنا بھی تناؤ کو بڑھنے سے پہلے نمایاں طور پر کم کر سکتا ہے۔",
        })
    elif dass.stress_severity in _HIGH_TIER:
        tips.append({
            "en": "Your stress levels appear quite high. Consider what specific "
                  "demands on your time could be reduced or shared, and whether "
                  "talking to someone about your workload would help.",
            "ur": "آپ کی تناؤ کی سطح کافی زیادہ معلوم ہوتی ہے۔ غور کریں کہ آپ کے وقت پر "
                  "کون سے مخصوص دباؤ کم یا کسی کے ساتھ بانٹے جا سکتے ہیں، اور کیا کسی سے "
                  "اپنے کام کے بوجھ پر بات کرنا مددگار ہو سکتا ہے۔",
        })

    return tips


def _fer_tips(fer: Optional[FerResult]) -> list[dict]:
    """One tip if positive expression dominated, one if negative expression
    was prominent. Both can appear together (e.g. a mixed session)."""
    tips: list[dict] = []
    if not fer or fer.frames_analyzed <= 0:
        return tips

    if fer.happy > 50:
        tips.append({
            "en": "Your facial expressions leaned notably positive during this "
                  "session - that's worth acknowledging alongside your questionnaire "
                  "results.",
            "ur": "اس سیشن کے دوران آپ کے چہرے کے تاثرات نمایاں طور پر مثبت رہے - یہ ایک "
                  "اچھی بات ہے جسے آپ کے سوالنامے کے نتائج کے ساتھ نوٹ کرنا چاہیے۔",
        })

    negative_share = sum(getattr(fer, e) for e in _NEGATIVE_EMOTIONS)
    if negative_share > 40:
        tips.append({
            "en": "A notable share of your captured expressions were negative. A "
                  "short emotional regulation practice, like slow paced breathing or "
                  "briefly naming what you're feeling, can help in the moment.",
            "ur": "آپ کے حاصل کردہ تاثرات کا ایک نمایاں حصہ منفی تھا۔ سست رفتار سانس "
                  "لینے یا مختصر طور پر اپنے احساس کا نام لینے جیسی جذباتی ضبط کی مشق "
                  "فوری طور پر مددگار ہو سکتی ہے۔",
        })

    return tips


def generate_actionable_tips(
    dass: Optional[DassResult], fer: Optional[FerResult]
) -> list[dict]:
    """
    Public entry point used both by compute_overall_risk() (right after a
    fresh submission) and directly by GET /api/assessments/{id} (when
    reconstructing an older, already-saved result) - since tips are never
    persisted, they're recomputed from the same stored DASS/FER numbers
    every time a result is displayed, in either place.
    """
    return _dass_tips(dass) + _fer_tips(fer)