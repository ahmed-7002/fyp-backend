"""
Combines DASS-21 severities, FER-2013 dominant emotion, the user's
self-reported `age`, and the user's self-reported `gender` into a single,
human-readable overall risk level, plus a set of bilingual, rule-based
"actionable tips" - short, specific suggestions derived purely from
already-computed DASS severities, FER percentages, age bracket, and gender.

Nothing here is a trained model and nothing here is persisted to the
database: tips are computed fresh, on the fly, every time a result is
viewed (both right after submission and later from the Profile page),
consistent with the rest of this file's existing design philosophy -
a transparent, auditable rule engine rather than an opaque model, so an
examiner (or a user) can trace exactly why a given tip appeared.

Neither `age` nor `gender` participates in the risk_level/summary
calculation - both are forwarded only to generate_actionable_tips() so tip
WORDING can be tailored. When either is unknown, every rule falls back to
its age/gender-neutral text.

DESIGN RULE - one technique per tip, no overlap (unchanged from before):
Each DASS subscale is assigned a DIFFERENT concrete technique per severity
tier (see the age-tip helpers below), so three simultaneously-elevated
subscales read as three different things a person can do, not the same
advice worded three ways.

DESIGN RULE - gender tips are additive, not a replacement:
Gender tailoring is layered ON TOP of the existing age-tailored tip for a
subscale, as a separate supplementary tip, rather than folding gender into
the age tip's text. This keeps the two dimensions independently auditable
(you can trace a tip to exactly "age rule" or "gender rule") and means nulled
or "Prefer not to say" gender never removes/degrades the age-based tip a
user would otherwise get.

Gender content is grounded in general, well-documented psychosocial
patterns discussed in the clinical/public-health literature - e.g. gaps in
help-seeking behaviour, "mental load"/invisible-labour research, and the
minority-stress model for gender-diverse people - phrased as patterns
"many"/"some" people experience, never as a universal claim about the
individual reading it. "Prefer not to say" and unrecognised/missing values
are treated identically to "unknown": no gender-specific tip is generated,
since guessing at content for someone who explicitly declined to share
would be inappropriate.

TIP-WRITING STYLE RULE (v2): every tip is 1-2 short sentences - name the
technique, give the one concrete action, and (for High-tier DASS tips)
a brief, direct nudge toward professional support. No extended empathetic
preamble, no explanation of *why* the technique works beyond a few words.
This is a deliberate choice: someone screening for elevated depression,
anxiety, or stress is a reader who benefits from something short and clear
to act on, not a paragraph to parse.
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

# Age brackets used purely for tailoring tip WORDING (see module docstring).
# Boundaries: Youth/Students < 25, Adults 25-49, Seniors 50+.
_YOUTH_MAX_AGE = 24
_ADULT_MAX_AGE = 49


def _age_bracket(age: Optional[int]) -> Optional[str]:
    """Maps a raw age into 'youth' | 'adult' | 'senior', or None if age is
    unknown. None is a first-class outcome, not an error - callers must
    fall back to age-neutral tip text when this returns None."""
    if age is None:
        return None
    if age < 0:
        return None
    if age <= _YOUTH_MAX_AGE:
        return "youth"
    if age <= _ADULT_MAX_AGE:
        return "adult"
    return "senior"


# Accepts the exact enum values stored in the database / sent by the
# frontend (see models.py gender_enum and schemas.py's Literal:
# "male" | "female" | "non_binary" | "prefer_not_to_say") plus a few
# common human-typed variants, and normalises everything else (including
# "prefer_not_to_say", blanks, and anything unrecognised) to None.
#
# "non_binary" (underscore) is the ACTUAL value this app ever sends - it's
# the literal enum member, not free text - so it must be a dict key here,
# not just the hyphen/space variants below. Without it, every non-binary
# user's gender silently normalised to None and never got a gender tip.
_GENDER_ALIASES = {
    "male": "male",
    "m": "male",
    "man": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
    "non_binary": "nonbinary",  # the actual enum value (models.py / schemas.py)
    "non-binary": "nonbinary",
    "nonbinary": "nonbinary",
    "non binary": "nonbinary",
    "enby": "nonbinary",
    "nb": "nonbinary",
}


def _gender_bracket(gender: Optional[str]) -> Optional[str]:
    """Maps a raw gender string into 'male' | 'female' | 'nonbinary', or
    None for unknown / 'Prefer not to say' / unrecognised values. None is a
    first-class outcome here too - it means "don't add a gender-specific
    tip", not an error."""
    if not gender:
        return None
    return _GENDER_ALIASES.get(gender.strip().lower())


def compute_overall_risk(
    dass: Optional[DassResult],
    fer: Optional[FerResult],
    age: Optional[int] = None,
    gender: Optional[str] = None,
) -> tuple[str, str, list[dict]]:
    """Returns (risk_level, summary_text, actionable_tips).

    Neither `age` nor `gender` participates in the risk_level/summary
    calculation - both are forwarded only to generate_actionable_tips() so
    that tip WORDING can be tailored to life stage and gender. This keeps
    final_risk_level fully determined by DASS/FER data alone, exactly as
    before.
    """
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

    tips = generate_actionable_tips(dass, fer, age, gender)

    return level, summary, tips


# --------------------------------------------------------------------------
# DASS-21 tip rulebook
# --------------------------------------------------------------------------


def _dass_tips(
    dass: Optional[DassResult],
    age: Optional[int] = None,
    gender: Optional[str] = None,
) -> list[dict]:
    """One age-tailored tip per DASS-21 subscale that is at least Mild, plus
    (when gender is known) one supplementary gender-tailored tip for that
    same subscale. Each subscale keeps its OWN technique per tier (see
    module docstring), and the gender tip always teaches something the age
    tip doesn't, so a fully-elevated, fully-known-demographics result still
    reads as distinct, non-repetitive guidance rather than restating itself."""
    tips: list[dict] = []
    if not dass:
        return tips

    age_bracket = _age_bracket(age)
    gender_bracket = _gender_bracket(gender)

    tips.extend(_depression_tips(dass.depression_severity, age_bracket))
    gtip = _gender_depression_tip(dass.depression_severity, gender_bracket)
    if gtip:
        tips.append(gtip)

    tips.extend(_anxiety_tips(dass.anxiety_severity, age_bracket))
    gtip = _gender_anxiety_tip(dass.anxiety_severity, gender_bracket)
    if gtip:
        tips.append(gtip)

    tips.extend(_stress_tips(dass.stress_severity, age_bracket))
    gtip = _gender_stress_tip(dass.stress_severity, gender_bracket)
    if gtip:
        tips.append(gtip)

    return tips


# ---- Depression: behavioural activation (Mild) / opposite action (High) --


def _depression_tips(severity: str, bracket: Optional[str]) -> list[dict]:
    tips: list[dict] = []

    if severity in _MILD_TIER:
        if bracket == "youth":
            tips.append({
                "en": "Try 'behavioural activation': do one small action today "
                      "(message a friend, a short walk) before you feel like it - "
                      "motivation follows action, not the other way around.",
                "ur": "'رویاتی فعالیت' آزمائیں: حوصلہ آنے سے پہلے ہی آج کوئی ایک "
                      "چھوٹا کام کریں (دوست کو پیغام، مختصر واک) - عمل حوصلے سے پہلے "
                      "آتا ہے۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Try 'behavioural activation': commit to one small action "
                      "today that isn't about productivity - a walk, a real meal, "
                      "or one phone call.",
                "ur": "'رویاتی فعالیت' آزمائیں: آج کارکردگی سے ہٹ کر ایک چھوٹا کام "
                      "کریں - واک، اچھا کھانا، یا کسی کو کال۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Try a gentle 'behavioural activation' step today - morning "
                      "sunlight, tending a plant, or a favourite meal - regardless "
                      "of how you feel beforehand.",
                "ur": "آج نرم 'رویاتی فعالیت' آزمائیں - صبح کی دھوپ، پودے کی دیکھ "
                      "بھال، یا پسندیدہ کھانا - پہلے سے موڈ کیسا بھی ہو۔",
            })
        else:
            tips.append({
                "en": "Set one small, achievable task today, even a short walk - "
                      "starting is usually the hardest part.",
                "ur": "آج ایک چھوٹا اور قابلِ حصول کام مقرر کریں، چاہے صرف مختصر "
                      "واک ہو - آغاز عام طور پر سب سے مشکل ہوتا ہے۔",
            })

    elif severity in _HIGH_TIER:
        if bracket == "youth":
            tips.append({
                "en": "Try DBT's 'opposite action': when you want to isolate, do "
                      "the smaller opposite - reply to one message, join others "
                      "for ten minutes. Please also tell a trusted adult or "
                      "counsellor how you're feeling.",
                "ur": "DBT کی 'مخالف عمل' آزمائیں: تنہائی چاہنے پر چھوٹا مخالف کام "
                      "کریں - کسی پیغام کا جواب دیں، دس منٹ لوگوں کے ساتھ رہیں۔ "
                      "براہِ کرم کسی قابلِ اعتماد بڑے یا کونسلر کو بھی بتائیں۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Try 'opposite action': when you want to withdraw, keep one "
                      "small commitment instead. Please also make time this week "
                      "to speak with a mental health professional.",
                "ur": "'مخالف عمل' آزمائیں: پیچھے ہٹنے کی بجائے کوئی ایک چھوٹا "
                      "وعدہ نبھائیں۔ براہِ کرم اس ہفتے کسی ذہنی صحت کے ماہر سے بھی "
                      "بات کریں۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Try 'opposite action': when you want to stay in, choose "
                      "the smaller opposite - answer the door, return one call. "
                      "Please also speak with a doctor or mental health "
                      "professional soon.",
                "ur": "'مخالف عمل' آزمائیں: گھر میں رہنے کی بجائے چھوٹا مخالف کام "
                      "کریں - دروازہ کھولیں، کسی کال کا جواب دیں۔ براہِ کرم جلد "
                      "ڈاکٹر یا ذہنی صحت کے ماہر سے بات کریں۔",
            })
        else:
            tips.append({
                "en": "Your responses suggest significant low mood. Consider "
                      "reaching out to a trusted person or a mental health "
                      "professional soon.",
                "ur": "آپ کے جوابات کافی حد تک متاثرہ مزاج ظاہر کرتے ہیں۔ کسی "
                      "قابلِ اعتماد شخص یا ذہنی صحت کے ماہر سے جلد رابطہ کریں۔",
            })

    return tips


def _gender_depression_tip(severity: str, bracket: Optional[str]) -> Optional[dict]:
    """Supplementary, gender-informed depression tip. Technique focus is
    deliberately different from _depression_tips above (which centres on
    behavioural activation / opposite action): this one centres on
    help-seeking and social/role context, so the two tips teach different
    things rather than repeating each other."""
    if severity in _MILD_TIER:
        if bracket == "male":
            return {
                "en": "Try telling one trusted person a single honest sentence "
                      "about how you've been feeling - naming it out loud helps, "
                      "it isn't a weakness.",
                "ur": "کسی قابلِ اعتماد شخص کو اپنے احساس کے بارے میں ایک "
                      "ایماندار جملہ بتائیں - بلند آواز میں کہنا مددگار ہے، "
                      "کمزوری نہیں۔",
            }
        if bracket == "female":
            return {
                "en": "Try one small experiment this week: hand off or skip one "
                      "task you'd normally do for others, and notice how that "
                      "feels.",
                "ur": "اس ہفتے ایک چھوٹا تجربہ کریں: کوئی ایک کام جو آپ عام طور "
                      "پر دوسروں کے لیے کرتی ہیں چھوڑ دیں یا کسی اور کے سپرد "
                      "کریں، اور محسوس کریں۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Try spending a little deliberate time this week in one "
                      "space - a person, community, or group - where you feel "
                      "fully seen.",
                "ur": "اس ہفتے کچھ وقت جان بوجھ کر ایسی جگہ گزاریں - کوئی شخص، "
                      "کمیونٹی، یا گروپ - جہاں آپ مکمل طور پر تسلیم کیے جاتے ہیں۔",
            }
        return None

    if severity in _HIGH_TIER:
        if bracket == "male":
            return {
                "en": "Reaching out for support is a practical step, not a "
                      "measure of strength. Please consider contacting a mental "
                      "health professional this week.",
                "ur": "مدد مانگنا ایک عملی قدم ہے، طاقت کا پیمانہ نہیں۔ براہِ "
                      "کرم اس ہفتے کسی ذہنی صحت کے ماہر سے رابطہ کریں۔",
            }
        if bracket == "female":
            return {
                "en": "Alongside seeing a professional, try naming one specific "
                      "responsibility you need help with to someone close to "
                      "you.",
                "ur": "کسی ماہر سے ملنے کے ساتھ، اپنے قریبی شخص کو کوئی ایک "
                      "مخصوص ذمہ داری بتائیں جس میں آپ کو مدد چاہیے۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Look specifically for a mental health professional who "
                      "is identity-affirming - it's reasonable to ask about "
                      "this directly when booking.",
                "ur": "ایسا ذہنی صحت کا ماہر تلاش کریں جو شناخت کو کھلے طور پر "
                      "تسلیم کرتا ہو - بکنگ کے وقت یہ پوچھنا درست ہے۔",
            }
        return None

    return None


# ---- Anxiety: breathing/physiological (Mild) / worry window (High) -------


def _anxiety_tips(severity: str, bracket: Optional[str]) -> list[dict]:
    tips: list[dict] = []

    if severity in _MILD_TIER:
        if bracket == "youth":
            tips.append({
                "en": "Try the 'physiological sigh': two short inhales through "
                      "the nose, then one long exhale through the mouth. Repeat "
                      "3-4 times.",
                "ur": "'فزیولوجیکل سائی' آزمائیں: ناک سے دو مختصر سانسیں، پھر "
                      "منہ سے ایک لمبی سانس باہر۔ 3-4 بار دہرائیں۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Try box breathing: inhale for 4 counts, hold 4, exhale "
                      "4, hold 4. Repeat for a minute - discreet enough for a "
                      "meeting or your desk.",
                "ur": "'باکس بریدنگ' آزمائیں: 4 گنتی سانس اندر، 4 روکیں، 4 "
                      "باہر، 4 روکیں۔ ایک منٹ دہرائیں - یہ میٹنگ یا میز پر بھی "
                      "کی جا سکتی ہے۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Pair slow breathing with gentle movement: inhale for 4 "
                      "counts raising your arms, exhale for 6 lowering them. "
                      "Ten rounds, seated or standing.",
                "ur": "سست سانس کو نرم حرکت کے ساتھ ملائیں: 4 گنتی سانس اندر "
                      "لیتے ہوئے بازو اٹھائیں، 6 گنتی باہر چھوڑتے ہوئے نیچے "
                      "لائیں۔ دس دور کریں۔",
            })
        else:
            tips.append({
                "en": "Try the 5-4-3-2-1 grounding technique: name 5 things "
                      "you see, 4 you touch, 3 you hear, 2 you smell, 1 you "
                      "taste.",
                "ur": "5-4-3-2-1 گراؤنڈنگ تکنیک آزمائیں: 5 چیزیں دیکھیں، 4 "
                      "چھوئیں، 3 سنیں، 2 سونگھیں، 1 چکھیں۔",
            })

    elif severity in _HIGH_TIER:
        if bracket == "youth":
            tips.append({
                "en": "Try a daily 10-minute 'worry window': write worries down "
                      "and revisit them at a set time instead of engaging right "
                      "away. Please also talk to a counsellor or mental health "
                      "professional.",
                "ur": "روزانہ 10 منٹ کی 'فکر کی کھڑکی' آزمائیں: فکریں لکھ لیں "
                      "اور مقررہ وقت پر دیکھیں، فوراً نہیں۔ براہِ کرم کونسلر یا "
                      "ذہنی صحت کے ماہر سے بھی بات کریں۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Try a daily 'worry window': set aside 10-15 minutes each "
                      "evening to think through worries and note one next step "
                      "for each. Please also speak with a mental health "
                      "professional.",
                "ur": "روزانہ 'فکر کی کھڑکی' آزمائیں: شام کو 10-15 منٹ فکروں پر "
                      "سوچنے اور اگلا قدم لکھنے کے لیے رکھیں۔ براہِ کرم ذہنی "
                      "صحت کے ماہر سے بھی بات کریں۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Try a short daily 'worry window': set aside 10 minutes "
                      "to sit with your worries on purpose, then set them "
                      "aside. This also deserves a conversation with a doctor "
                      "or mental health professional.",
                "ur": "روزانہ مختصر 'فکر کی کھڑکی' آزمائیں: 10 منٹ جان بوجھ کر "
                      "فکروں کے ساتھ گزاریں، پھر الگ رکھ دیں۔ اس پر ڈاکٹر یا "
                      "ذہنی صحت کے ماہر سے بھی بات کریں۔",
            })
        else:
            tips.append({
                "en": "Your anxiety responses are in a higher range. Alongside "
                      "grounding techniques, consider talking to a mental "
                      "health professional about what you're experiencing.",
                "ur": "آپ کی بے چینی زیادہ سطح پر ہے۔ گراؤنڈنگ تکنیکوں کے "
                      "ساتھ، ذہنی صحت کے ماہر سے بات کرنے پر غور کریں۔",
            })

    return tips


def _gender_anxiety_tip(severity: str, bracket: Optional[str]) -> Optional[dict]:
    """Supplementary, gender-informed anxiety tip. Technique focus is
    deliberately different from _anxiety_tips above (breathing / worry
    windows): this one centres on the underlying social pressure often
    feeding the anxiety, so the two tips stay non-overlapping."""
    if severity in _MILD_TIER:
        if bracket == "male":
            return {
                "en": "Restless energy can be a sign of anxiety too. Try 5 "
                      "minutes of physical activity - push-ups, a fast walk, "
                      "stairs - when you notice it building.",
                "ur": "بے چین توانائی بھی بے چینی کی علامت ہو سکتی ہے۔ جب یہ "
                      "بڑھے تو 5 منٹ جسمانی سرگرمی کریں - پش اپس، تیز واک، "
                      "سیڑھیاں۔",
            }
        if bracket == "female":
            return {
                "en": "Try a 'brain dump' before bed: write down everything "
                      "you're holding in mind for other people, so your mind "
                      "doesn't rehearse it overnight.",
                "ur": "سونے سے پہلے 'دماغ خالی کرنا' آزمائیں: دوسروں کے لیے یاد "
                      "رکھی گئی باتیں لکھ لیں تاکہ ذہن رات بھر نہ دہرائے۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Before entering an unfamiliar or unaffirming space, "
                      "plan one small anchor in advance - a person you can "
                      "text, or a phrase to use if you need to step away.",
                "ur": "کسی نامانوس یا غیر تسلیم کرنے والی جگہ جانے سے پہلے ایک "
                      "سہارا طے کریں - کوئی شخص جسے پیغام بھیج سکیں، یا نکلنے "
                      "کا جملہ۔",
            }
        return None

    if severity in _HIGH_TIER:
        if bracket == "male":
            return {
                "en": "Please raise this specifically with a doctor or mental "
                      "health professional, using concrete words like 'racing "
                      "thoughts' rather than calling it 'just stress'.",
                "ur": "براہِ کرم ڈاکٹر یا ذہنی صحت کے ماہر کو واضح الفاظ میں "
                      "بتائیں، جیسے 'خیالات کا تیزی سے دوڑنا'، اسے محض 'تناؤ' "
                      "نہ کہیں۔",
            }
        if bracket == "female":
            return {
                "en": "Please talk to a mental health professional, and "
                      "separately, identify one recurring 'invisible task' you "
                      "could hand to someone else.",
                "ur": "براہِ کرم کسی ذہنی صحت کے ماہر سے بات کریں، اور ایک بار "
                      "بار آنے والا 'غیر مرئی کام' کسی اور کے سپرد کریں۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "When looking for professional support, it's reasonable "
                      "to specifically ask about identity-affirming care - you "
                      "shouldn't have to explain the basics of who you are "
                      "first.",
                "ur": "پیشہ ورانہ مدد تلاش کرتے وقت شناخت کو تسلیم کرنے والی "
                      "دیکھ بھال کے بارے میں پوچھنا درست ہے - آپ کو پہلے خود کو "
                      "سمجھانا نہیں پڑنا چاہیے۔",
            }
        return None

    return None


# ---- Stress: pacing (Mild) / triage & prioritisation (High) --------------


def _stress_tips(severity: str, bracket: Optional[str]) -> list[dict]:
    tips: list[dict] = []

    if severity in _MILD_TIER:
        if bracket == "youth":
            tips.append({
                "en": "Try timeboxing: 25 minutes of focused work, then a real "
                      "5-minute break away from screens before the next block.",
                "ur": "ٹائم باکسنگ آزمائیں: 25 منٹ مرکوز کام، پھر اسکرین سے دور "
                      "5 منٹ کا حقیقی وقفہ۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Try 'task chunking': break one large task into its "
                      "smallest next step and do only that, before "
                      "reassessing.",
                "ur": "'ٹاسک چنکنگ' آزمائیں: بڑے کام کو اگلے چھوٹے قدم میں "
                      "تقسیم کریں اور صرف وہی کریں، پھر دوبارہ جائزہ لیں۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Set a fixed time each day, even ten minutes, for "
                      "gentle movement - a short walk, light stretching, or "
                      "fresh air.",
                "ur": "ہر روز ایک مقررہ وقت رکھیں، دس منٹ ہی سہی، نرم حرکت کے "
                      "لیے - مختصر واک، ہلکی سٹریچنگ، یا کھلی ہوا۔",
            })
        else:
            tips.append({
                "en": "Build in short, deliberate breaks through your day - "
                      "even five minutes away from a task can lower stress "
                      "before it builds up.",
                "ur": "اپنے دن میں مختصر وقفے شامل کریں - کسی کام سے پانچ منٹ "
                      "دور رہنا بھی تناؤ کو بڑھنے سے پہلے کم کر سکتا ہے۔",
            })

    elif severity in _HIGH_TIER:
        if bracket == "youth":
            tips.append({
                "en": "Try a quick triage: sort everything due this week into "
                      "'must do', 'can ask for an extension on', or 'can "
                      "drop'. Please also talk to a teacher, counsellor, or "
                      "mental health professional.",
                "ur": "فوری ترجیح بندی آزمائیں: ہر ذمہ داری کو 'لازمی'، "
                      "'توسیع مانگی جا سکتی ہے'، یا 'چھوڑی جا سکتی ہے' میں "
                      "تقسیم کریں۔ براہِ کرم استاد، کونسلر، یا ذہنی صحت کے "
                      "ماہر سے بھی بات کریں۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Try a triage exercise: list every current demand, then "
                      "mark each 'only I can do this', 'could be delegated', "
                      "or 'could wait'. Please also raise this with a mental "
                      "health professional.",
                "ur": "ترجیح بندی آزمائیں: ہر موجودہ دباؤ کی فہرست بنائیں، "
                      "پھر نشان لگائیں 'صرف میں کر سکتا ہوں'، 'سپرد ہو سکتا "
                      "ہے'، یا 'انتظار کر سکتا ہے'۔ براہِ کرم ذہنی صحت کے ماہر "
                      "سے بھی بات کریں۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Try simplifying: pick one commitment you're holding "
                      "onto out of habit rather than necessity, and let it go "
                      "or share it. This also deserves a doctor or mental "
                      "health professional's attention.",
                "ur": "آسان بنانے کی کوشش کریں: کوئی ایک وعدہ جو عادت کے تحت "
                      "نبھا رہے ہیں چھوڑ دیں یا بانٹ لیں۔ اس پر ڈاکٹر یا ذہنی "
                      "صحت کے ماہر سے بھی بات کریں۔",
            })
        else:
            tips.append({
                "en": "Your stress levels appear quite high. Consider what "
                      "demands on your time could be reduced or shared, and "
                      "talk to someone about your workload.",
                "ur": "آپ کی تناؤ کی سطح کافی زیادہ ہے۔ غور کریں کہ کون سے "
                      "دباؤ کم یا بانٹے جا سکتے ہیں، اور اپنے کام کے بوجھ پر "
                      "کسی سے بات کریں۔",
            })

    return tips


def _gender_stress_tip(severity: str, bracket: Optional[str]) -> Optional[dict]:
    """Supplementary, gender-informed stress tip. Technique focus is
    deliberately different from _stress_tips above (timeboxing / task
    chunking / triage): this one centres on redistributing or renegotiating
    role-based load, so the two tips stay non-overlapping."""
    if severity in _MILD_TIER:
        if bracket == "male":
            return {
                "en": "Try asking for help with one specific thing this week "
                      "- not as a last resort, but as a normal, early step.",
                "ur": "اس ہفتے کسی ایک چیز میں مدد مانگیں - آخری حل کے طور پر "
                      "نہیں، بلکہ ایک معمول کے قدم کے طور پر۔",
            }
        if bracket == "female":
            return {
                "en": "Try making one invisible task visible: write it down "
                      "and hand it to someone else explicitly, rather than "
                      "quietly doing it again.",
                "ur": "کوئی ایک غیر مرئی کام نظر آنے والا بنائیں: اسے لکھ کر "
                      "واضح طور پر کسی اور کے سپرد کریں، خاموشی سے دوبارہ خود "
                      "نہ کریں۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Identify which spaces in your week are genuinely "
                      "low-friction, and protect more time there.",
                "ur": "اپنے ہفتے میں وہ جگہیں پہچانیں جو حقیقتاً کم دباؤ والی "
                      "ہیں، اور وہاں مزید وقت محفوظ رکھیں۔",
            }
        return None

    if severity in _HIGH_TIER:
        if bracket == "male":
            return {
                "en": "Please talk to a mental health professional about your "
                      "workload, and consider telling one close person you're "
                      "stretched thin - it's useful information, not an "
                      "admission of failure.",
                "ur": "براہِ کرم اپنے کام کے بوجھ پر ذہنی صحت کے ماہر سے بات "
                      "کریں، اور کسی قریبی شخص کو بتائیں کہ آپ زیادہ دباؤ میں "
                      "ہیں۔",
            }
        if bracket == "female":
            return {
                "en": "Please talk to a mental health professional, and "
                      "separately, list three recurring responsibilities and "
                      "identify who else could take over at least one "
                      "permanently.",
                "ur": "براہِ کرم ذہنی صحت کے ماہر سے بات کریں، اور تین بار بار "
                      "آنے والی ذمہ داریاں فہرست کر کے دیکھیں کہ کون ایک "
                      "ذمہ داری مستقل سنبھال سکتا ہے۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Please talk to a mental health professional, ideally "
                      "one with experience supporting gender-diverse clients, "
                      "and consider whether one recurring source of friction "
                      "could be addressed directly.",
                "ur": "براہِ کرم ذہنی صحت کے ماہر سے بات کریں، ترجیحاً صنفی "
                      "طور پر متنوع مؤکلوں کا تجربہ رکھنے والے سے، اور کسی بار "
                      "بار آنے والی رگڑ کو براہِ راست حل کرنے پر غور کریں۔",
            }
        return None

    return None


# --------------------------------------------------------------------------
# FER-2013 tip rulebook
# --------------------------------------------------------------------------


def _fer_tips(
    fer: Optional[FerResult],
    age: Optional[int] = None,
    gender: Optional[str] = None,
) -> list[dict]:
    """One tip if positive expression dominated, one if negative expression
    was prominent, one additional tip specifically when sadness is the
    driving factor at higher levels, and (when gender is known) one further
    supplementary gender-tailored tip when negative expression is
    prominent. Each helper teaches a distinct technique so the set doesn't
    repeat itself when more than one fires."""
    tips: list[dict] = []
    if not fer or fer.frames_analyzed <= 0:
        return tips

    age_bracket = _age_bracket(age)
    gender_bracket = _gender_bracket(gender)

    if fer.happy > 50:
        tips.append(_fer_happy_tip(age_bracket))

    negative_share = sum(getattr(fer, e) for e in _NEGATIVE_EMOTIONS)
    if negative_share > 40:
        tips.append(_fer_negative_tip(age_bracket))
        gtip = _gender_fer_negative_tip(gender_bracket)
        if gtip:
            tips.append(gtip)

    # Sadness-specific rule: intersects a single emotion (sad) with age, per
    # spec, since sadness in particular tends to call for different coping
    # strategies depending on life stage (isolation risk in older adults vs.
    # social/identity pressure in younger users). Uses a distinct technique
    # (behavioural, not breathing-based) from _fer_negative_tip above so the
    # two don't overlap when both fire together.
    if fer.sad > 30:
        tips.append(_fer_sad_tip(age_bracket))

    return tips


def _fer_happy_tip(bracket: Optional[str]) -> dict:
    """Technique: identify-and-repeat - name the context behind the positive
    reading so it can be intentionally reproduced."""
    if bracket == "youth":
        return {
            "en": "Your expressions leaned notably positive this session. "
                  "Try to identify what was present beforehand (a person, "
                  "activity, a break from screens) and build more of it "
                  "into your week.",
            "ur": "اس سیشن میں آپ کے تاثرات نمایاں مثبت رہے۔ پہچانیں کہ اس سے "
                  "پہلے کیا موجود تھا اور اسے اپنے ہفتے میں مزید شامل کریں۔",
        }
    if bracket == "adult":
        return {
            "en": "Your expressions leaned notably positive this session - "
                  "worth noting alongside your questionnaire results, "
                  "especially if it coincided with time away from work.",
            "ur": "اس سیشن میں آپ کے تاثرات نمایاں مثبت رہے - یہ سوالنامے کے "
                  "نتائج کے ساتھ نوٹ کرنے کے قابل ہے، خاص طور پر اگر یہ کام "
                  "سے وقفے میں ہوا۔",
        }
    if bracket == "senior":
        return {
            "en": "Your expressions leaned notably positive this session. "
                  "Whatever surrounded this moment - company, routine, time "
                  "outdoors - is worth protecting and repeating.",
            "ur": "اس سیشن میں آپ کے تاثرات نمایاں مثبت رہے۔ جو کچھ اس لمحے "
                  "کے ارد گرد تھا اسے محفوظ رکھنا اور دہرانا مفید ہے۔",
        }
    return {
        "en": "Your expressions leaned notably positive this session - "
              "worth acknowledging alongside your questionnaire results.",
        "ur": "اس سیشن میں آپ کے تاثرات نمایاں مثبت رہے - یہ سوالنامے کے "
              "نتائج کے ساتھ نوٹ کرنے کے قابل ہے۔",
    }


def _fer_negative_tip(bracket: Optional[str]) -> dict:
    """Technique: affect labelling - naming the feeling in one sentence to
    reduce its intensity. Distinct from the sadness-specific tip below,
    which focuses on a behavioural response rather than a naming exercise."""
    if bracket == "youth":
        return {
            "en": "A notable share of your expressions were negative. Try "
                  "'affect labelling': name the feeling in one plain "
                  "sentence - naming it this specifically can ease its "
                  "intensity within minutes.",
            "ur": "آپ کے تاثرات کا ایک نمایاں حصہ منفی تھا۔ 'احساس کا نام "
                  "لینا' آزمائیں: ایک سادہ جملے میں احساس کا نام لیں - یہ "
                  "چند منٹوں میں شدت کم کر سکتا ہے۔",
        }
    if bracket == "adult":
        return {
            "en": "A notable share of your expressions were negative. Try "
                  "'affect labelling': name what you're feeling and why in "
                  "one sentence, rather than leaving it vague.",
            "ur": "آپ کے تاثرات کا ایک نمایاں حصہ منفی تھا۔ 'احساس کا نام "
                  "لینا' آزمائیں: ایک جملے میں نام لیں کہ کیا محسوس ہو رہا "
                  "ہے اور کیوں۔",
        }
    if bracket == "senior":
        return {
            "en": "A notable share of your expressions were negative. Try "
                  "'affect labelling': quietly name what you're feeling in "
                  "a single sentence rather than letting it sit unnamed.",
            "ur": "آپ کے تاثرات کا ایک نمایاں حصہ منفی تھا۔ 'احساس کا نام "
                  "لینا' آزمائیں: خاموشی سے ایک جملے میں احساس کا نام لیں۔",
        }
    return {
        "en": "A notable share of your expressions were negative. A short "
              "practice, like slow breathing or naming the feeling, can "
              "help in the moment.",
        "ur": "آپ کے تاثرات کا ایک نمایاں حصہ منفی تھا۔ سست سانس یا احساس "
              "کا نام لینا فوری مددگار ہو سکتا ہے۔",
    }


def _gender_fer_negative_tip(bracket: Optional[str]) -> Optional[dict]:
    """Supplementary, gender-informed tip when negative expression is
    prominent. Technique focus is deliberately different from
    _fer_negative_tip above (affect labelling): this one centres on where
    the feeling is directed (inward suppression vs. outward validation),
    so the two stay non-overlapping."""
    if bracket == "male":
        return {
            "en": "Negative feeling can hide behind a neutral face, "
                  "especially under pressure not to react. Check in with "
                  "your body - jaw, shoulders, hands - for tension it may "
                  "be masking.",
            "ur": "منفی احساس غیر جانبدار چہرے کے پیچھے چھپ سکتا ہے۔ اپنے "
                  "جسم کو جانچیں - جبڑا، کندھے، ہاتھ - جو تناؤ چھپا رہا "
                  "ہو۔",
        }
    if bracket == "female":
        return {
            "en": "It's easy to keep functioning through a hard feeling "
                  "without addressing it. Give yourself explicit permission "
                  "for five minutes today that belong only to you.",
            "ur": "مشکل احساس کے ساتھ کام جاری رکھنا آسان ہے، لیکن اسے حل "
                  "کیے بغیر۔ آج پانچ منٹ صرف اپنے لیے مقرر کریں۔",
        }
    if bracket == "nonbinary":
        return {
            "en": "Negative feelings after unaffirming spaces are a valid "
                  "response, not oversensitivity. Spend a few minutes "
                  "afterward somewhere explicitly affirming to help reset.",
            "ur": "غیر تسلیم کرنے والی جگہوں کے بعد منفی احساس ایک درست "
                  "ردِعمل ہے۔ اس کے بعد کچھ منٹ کسی تسلیم کرنے والی جگہ "
                  "گزاریں۔",
        }
    return None


def _fer_sad_tip(bracket: Optional[str]) -> dict:
    """Technique: a concrete behavioural/connection step, kept deliberately
    different from the naming exercise in _fer_negative_tip so the two read
    as distinct actions when both fire in the same result."""
    if bracket == "youth":
        return {
            "en": "Sadness stood out clearly this session. As one concrete "
                  "step, message someone in your trusted circle and suggest "
                  "meeting up or calling, even briefly.",
            "ur": "اس سیشن میں اداسی واضح طور پر نمایاں تھی۔ ایک ٹھوس قدم "
                  "کے طور پر کسی قابلِ اعتماد شخص کو پیغام بھیجیں اور ملنے یا "
                  "کال کی تجویز دیں۔",
        }
    if bracket == "adult":
        return {
            "en": "Sadness stood out clearly this session. Take five "
                  "minutes to identify one relationship or responsibility "
                  "asking more of you than it gives back, and note one "
                  "small change.",
            "ur": "اس سیشن میں اداسی واضح طور پر نمایاں تھی۔ پانچ منٹ لے کر "
                  "پہچانیں کہ کون سا تعلق یا ذمہ داری آپ سے زیادہ مانگ رہی "
                  "ہے، اور ایک چھوٹی تبدیلی نوٹ کریں۔",
        }
    if bracket == "senior":
        return {
            "en": "Sadness stood out clearly this session. Pick one person "
                  "- family, an old friend, a neighbour - and reach out "
                  "today, even briefly.",
            "ur": "اس سیشن میں اداسی واضح طور پر نمایاں تھی۔ ایک شخص چنیں - "
                  "خاندان، پرانا دوست، پڑوسی - اور آج ہی رابطہ کریں۔",
        }
    return {
        "en": "Sadness stood out clearly this session. Taking a few "
              "minutes to acknowledge it and connecting with someone you "
              "trust can help before it builds further.",
        "ur": "اس سیشن میں اداسی واضح طور پر نمایاں تھی۔ چند منٹ رک کر اسے "
              "تسلیم کرنا اور کسی قابلِ اعتماد شخص سے رابطہ کرنا مددگار ہو "
              "سکتا ہے۔",
    }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def generate_actionable_tips(
    dass: Optional[DassResult],
    fer: Optional[FerResult],
    age: Optional[int] = None,
    gender: Optional[str] = None,
) -> list[dict]:
    """
    Public entry point used both by compute_overall_risk() (right after a
    fresh submission) and directly by GET /api/assessments/{id} (when
    reconstructing an older, already-saved result) - since tips are never
    persisted, they're recomputed from the same stored DASS/FER/age/gender
    values every time a result is displayed, in either place.

    `age` and `gender` are both optional and purely additive: passing None
    for either (or both) reproduces the original, neutral tip set for that
    dimension exactly. 'Prefer not to say' is treated the same as None for
    gender - no gender-specific tip is generated in that case.
    """
    return _dass_tips(dass, age, gender) + _fer_tips(fer, age, gender)