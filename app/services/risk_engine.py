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


# Accepts the exact labels used on the onboarding form ("Female", "Male",
# "Non-binary", "Prefer not to say") plus a few common variants, and
# normalises everything else (including "Prefer not to say", blanks, and
# anything unrecognised) to None.
_GENDER_ALIASES = {
    "male": "male",
    "m": "male",
    "man": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
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
                "en": "It's completely understandable to feel weighed down right now - "
                      "many students carry this quietly while comparing themselves to "
                      "curated highlight reels online. Try a CBT technique called "
                      "'behavioural activation': pick one tiny, concrete action today "
                      "(texting one friend, a 10-minute walk between classes) and do "
                      "it BEFORE you feel motivated, not after - motivation usually "
                      "follows action, not the other way around.",
                "ur": "ابھی خود کو بوجھل محسوس کرنا بالکل قابلِ فہم ہے - بہت سے طلبہ "
                      "خاموشی سے یہ بوجھ اٹھاتے ہیں جبکہ سوشل میڈیا پر دوسروں کی سجی "
                      "سنوری زندگیوں سے اپنا موازنہ کرتے رہتے ہیں۔ CBT کی ایک تکنیک "
                      "'رویاتی فعالیت' آزمائیں: آج کے لیے ایک نہایت چھوٹا اور ٹھوس "
                      "قدم چنیں (کسی دوست کو پیغام بھیجنا، کلاسوں کے درمیان 10 منٹ کی "
                      "واک) اور حوصلہ آنے سے پہلے ہی وہ کام کریں - عام طور پر پہلے "
                      "عمل ہوتا ہے، پھر حوصلہ آتا ہے، الٹا نہیں۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Low mood alongside a full plate of work and relationship "
                      "responsibilities is exhausting, and it's not a personal "
                      "failing. Try 'behavioural activation': write down everything "
                      "pulling at you, then commit to just ONE small physical action "
                      "today, unrelated to output or productivity - a short walk, "
                      "cooking a real meal, calling one person. Completing it gives "
                      "your brain real evidence that you can still act, even when "
                      "motivation is low.",
                "ur": "کام اور تعلقات کی مکمل ذمہ داریوں کے ساتھ اداسی محسوس کرنا "
                      "تھکا دینے والا ہے، اور یہ آپ کی کوئی ذاتی کمزوری نہیں۔ "
                      "'رویاتی فعالیت' آزمائیں: جو کچھ بھی آپ کو پریشان کر رہا ہے اسے "
                      "لکھ لیں، پھر آج کے لیے صرف ایک چھوٹے جسمانی عمل کا عہد کریں، "
                      "جس کا کارکردگی یا پیداوار سے تعلق نہ ہو - مختصر واک، حقیقی کھانا "
                      "پکانا، کسی ایک شخص کو کال کرنا۔ اسے مکمل کرنا دماغ کو یہ ثبوت "
                      "دیتا ہے کہ حوصلہ کم ہونے پر بھی آپ عمل کر سکتے ہیں۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Mood and the body are closely linked, and this becomes more "
                      "noticeable with age. Try 'behavioural activation' in its "
                      "gentlest form: choose one small, pleasurable action for today "
                      "- ten minutes of morning sunlight, tending a plant, or "
                      "preparing a favourite small meal - and do it regardless of "
                      "how you feel beforehand. Small completed actions rebuild a "
                      "sense of agency that low mood tends to erode.",
                "ur": "مزاج اور جسم کا آپس میں گہرا تعلق ہے، اور عمر بڑھنے کے ساتھ یہ "
                      "زیادہ نمایاں ہو جاتا ہے۔ 'رویاتی فعالیت' کو اس کی نرم ترین شکل "
                      "میں آزمائیں: آج کے لیے ایک چھوٹا، خوشگوار عمل چنیں - صبح کی "
                      "دس منٹ کی دھوپ، پودے کی دیکھ بھال، یا اپنا پسندیدہ چھوٹا کھانا "
                      "بنانا - اور اسے پہلے سے کیسا محسوس کر رہے ہیں اس سے قطع نظر "
                      "کریں۔ چھوٹے مکمل شدہ اعمال وہ خودمختاری کا احساس دوبارہ تعمیر "
                      "کرتے ہیں جسے اداسی کمزور کر دیتی ہے۔",
            })
        else:
            tips.append({
                "en": "Try setting one small, achievable task for today, even something as "
                      "simple as a short walk - depression often makes starting anything "
                      "feel harder than it actually is.",
                "ur": "آج کے لیے ایک چھوٹا اور قابلِ حصول کام مقرر کریں، چاہے وہ صرف ایک "
                      "مختصر واک ہی کیوں نہ ہو - ڈپریشن میں کسی بھی کام کا آغاز حقیقت سے "
                      "زیادہ مشکل محسوس ہوتا ہے۔",
            })

    elif severity in _HIGH_TIER:
        if bracket == "youth":
            tips.append({
                "en": "What you're carrying sounds heavy. On the coping side, try "
                      "'opposite action' from DBT: when low mood pulls you to "
                      "isolate, deliberately do the small opposite thing - sit in a "
                      "shared space, reply to one message, join a group activity for "
                      "just ten minutes, even without wanting to. Separately, and "
                      "just as importantly: please tell a trusted adult, campus "
                      "counsellor, or a mental health professional what you shared "
                      "here - you don't have to carry this alone.",
                "ur": "آپ جو بوجھ اٹھا رہے ہیں وہ واقعی بھاری معلوم ہوتا ہے۔ نمٹنے کے "
                      "لیے DBT کی 'مخالف عمل' تکنیک آزمائیں: جب اداسی آپ کو تنہائی کی "
                      "طرف کھینچے، جان بوجھ کر اس کے چھوٹے مخالف کام کریں - کسی مشترکہ "
                      "جگہ میں بیٹھیں، ایک پیغام کا جواب دیں، صرف دس منٹ کے لیے کسی "
                      "گروپ سرگرمی میں شامل ہوں، چاہے دل نہ چاہے۔ الگ سے، اور اتنا ہی "
                      "اہم: براہِ کرم جو کچھ آپ نے یہاں بتایا وہ کسی قابلِ اعتماد بڑے، "
                      "کیمپس کونسلر، یا ذہنی صحت کے ماہر کو بھی بتائیں - آپ کو یہ "
                      "اکیلے نہیں سنبھالنا۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "This level of low mood, layered on top of career and "
                      "relationship demands, deserves real support. As an "
                      "in-the-moment technique, try 'opposite action': when the pull "
                      "is to withdraw and cancel plans, do the smaller opposite - "
                      "keep one commitment, even briefly. This isn't about pushing "
                      "through everything; it's one counter-move against the "
                      "isolation that deepens low mood. Please also make time this "
                      "week to speak with a mental health professional directly.",
                "ur": "کیریئر اور تعلقات کے دباؤ کے ساتھ اس سطح کی اداسی کو حقیقی مدد "
                      "کی ضرورت ہے۔ فوری تکنیک کے طور پر 'مخالف عمل' آزمائیں: جب دل "
                      "پیچھے ہٹنے اور منصوبے منسوخ کرنے کو چاہے، اس کے چھوٹے مخالف کام "
                      "کریں - کوئی ایک وعدہ نبھائیں، مختصر ہی سہی۔ یہ سب کچھ زبردستی "
                      "کرنے کی بات نہیں؛ یہ تنہائی کے خلاف ایک جوابی قدم ہے جو اداسی کو "
                      "مزید گہرا کرتی ہے۔ براہِ کرم اس ہفتے کسی ذہنی صحت کے ماہر سے بھی "
                      "براہِ راست بات کرنے کے لیے وقت نکالیں۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Please know that what you're feeling is real and treatable, "
                      "not something to simply endure - a conversation with a doctor "
                      "or mental health professional soon matters here. As a daily "
                      "practice alongside that: try 'opposite action' - when low mood "
                      "makes you want to stay in and skip contact, choose the smaller "
                      "opposite instead, like answering the door, sitting on the "
                      "porch, or returning one call, even for a few minutes.",
                "ur": "براہِ کرم جان لیں کہ آپ کا یہ احساس حقیقی ہے اور اس کا علاج "
                      "ممکن ہے، اسے یونہی برداشت کرنے کی ضرورت نہیں - یہاں جلد کسی "
                      "ڈاکٹر یا ذہنی صحت کے ماہر سے بات کرنا اہم ہے۔ اس کے ساتھ روزانہ "
                      "کی مشق کے طور پر: 'مخالف عمل' آزمائیں - جب اداسی آپ کو گھر میں "
                      "رہنے اور رابطے سے بچنے پر مجبور کرے، اس کے بجائے چھوٹا مخالف کام "
                      "کریں، جیسے دروازہ کھولنا، برآمدے میں بیٹھنا، یا کسی ایک کال کا "
                      "جواب دینا، چاہے چند منٹ کے لیے ہی سہی۔",
            })
        else:
            tips.append({
                "en": "Your responses suggest a significant level of low mood. Consider "
                      "reaching out to a trusted person or a mental health professional "
                      "soon - you don't have to manage this alone.",
                "ur": "آپ کے جوابات ظاہر کرتے ہیں کہ آپ کا مزاج کافی حد تک متاثر ہے۔ کسی "
                      "قابلِ اعتماد شخص یا ذہنی صحت کے ماہر سے جلد رابطہ کرنے پر غور کریں - "
                      "آپ کو یہ اکیلے نہیں سنبھالنا۔",
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
                "en": "Many men are socialised to push low mood aside rather than "
                      "name it, which can make it build quietly. Try telling one "
                      "person you trust a single honest sentence about how you've "
                      "been feeling this week - not to solve it, just to say it out "
                      "loud. Naming it to someone else is itself a form of relief, "
                      "not a sign anything is wrong with you.",
                "ur": "بہت سے مردوں کو یہ سکھایا جاتا ہے کہ اداسی کو نظرانداز کریں "
                      "بجائے اسے تسلیم کرنے کے، جس سے یہ خاموشی سے بڑھ سکتی ہے۔ کسی "
                      "قابلِ اعتماد شخص کو اس ہفتے اپنے احساس کے بارے میں ایک ایماندار "
                      "جملہ بتانے کی کوشش کریں - اسے حل کرنے کے لیے نہیں، بس بلند "
                      "آواز میں کہنے کے لیے۔ کسی اور کو بتانا خود ایک طرح کا سکون ہے، "
                      "اس بات کی علامت نہیں کہ آپ میں کچھ غلط ہے۔",
            }
        if bracket == "female":
            return {
                "en": "Low mood often arrives alongside an invisible 'mental load' - "
                      "remembering, planning, and caretaking for others - that many "
                      "women carry by default. Try one small experiment this week: "
                      "consciously hand off or skip one task you'd normally do "
                      "automatically for someone else, and notice how that feels "
                      "without immediately filling the space with something else.",
                "ur": "اداسی اکثر ایک غیر مرئی 'ذہنی بوجھ' کے ساتھ آتی ہے - یاد "
                      "رکھنا، منصوبہ بندی کرنا، اور دوسروں کی دیکھ بھال کرنا - جو بہت "
                      "سی خواتین بطور معمول اٹھاتی ہیں۔ اس ہفتے ایک چھوٹا تجربہ "
                      "کریں: جان بوجھ کر کوئی ایک کام کسی اور کے سپرد کریں یا چھوڑ "
                      "دیں جو آپ عام طور پر خودکار طریقے سے کرتی ہیں، اور دیکھیں کہ "
                      "اس خالی جگہ کو فوراً کسی اور چیز سے بھرے بغیر آپ کیسا محسوس "
                      "کرتی ہیں۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Low mood connected to navigating a world that doesn't always "
                      "see or affirm your identity is real, and it isn't something "
                      "you're overreacting to. Try to identify one space - a person, "
                      "a community, an online group - where you feel fully seen, and "
                      "spend a little deliberate time there this week rather than "
                      "waiting for it to happen on its own.",
                "ur": "ایک ایسی دنیا میں رہنے سے جڑی اداسی حقیقی ہے جو ہمیشہ آپ کی "
                      "شناخت کو نہیں دیکھتی یا تسلیم نہیں کرتی، اور یہ کوئی ایسی چیز "
                      "نہیں جس پر آپ ضرورت سے زیادہ ردِعمل ظاہر کر رہے ہیں۔ ایک ایسی "
                      "جگہ پہچاننے کی کوشش کریں - کوئی شخص، کمیونٹی، یا آن لائن گروپ "
                      "- جہاں آپ مکمل طور پر تسلیم کیے جاتے ہیں، اور اس ہفتے وہاں "
                      "جان بوجھ کر تھوڑا وقت گزاریں، بجائے اس کے کہ یہ خود بخود ہونے "
                      "کا انتظار کریں۔",
            }
        return None

    if severity in _HIGH_TIER:
        if bracket == "male":
            return {
                "en": "Men are statistically less likely to seek help for low mood, "
                      "often because of pressure to appear self-sufficient - but "
                      "reaching out is a practical decision, not a measure of "
                      "strength or weakness. Please consider making contact with a "
                      "mental health professional this week, framed simply as "
                      "getting information, if that feels easier than 'getting "
                      "help'.",
                "ur": "اداسی کے لیے مدد مانگنے کا امکان اعداد و شمار کے مطابق مردوں "
                      "میں کم ہوتا ہے، اکثر خودکفیل دکھنے کے دباؤ کی وجہ سے - لیکن "
                      "مدد مانگنا ایک عملی فیصلہ ہے، طاقت یا کمزوری کا پیمانہ نہیں۔ "
                      "براہِ کرم اس ہفتے کسی ذہنی صحت کے ماہر سے رابطہ کرنے پر غور "
                      "کریں، اگر 'مدد لینا' مشکل لگے تو اسے صرف معلومات حاصل کرنا "
                      "سمجھ لیں۔",
            }
        if bracket == "female":
            return {
                "en": "This level of low mood deserves professional support, and it "
                      "also deserves a serious look at what's being carried day to "
                      "day - many women reach this point while also managing a "
                      "disproportionate share of household or caregiving load. "
                      "Alongside seeing a professional, try naming out loud to "
                      "someone close to you one specific responsibility you need "
                      "help with this week.",
                "ur": "اداسی کی یہ سطح پیشہ ورانہ مدد کی مستحق ہے، اور اسے سنجیدگی سے "
                      "دیکھنے کی بھی ضرورت ہے کہ روزانہ کیا بوجھ اٹھایا جا رہا ہے - "
                      "بہت سی خواتین اس مقام تک پہنچتی ہیں جبکہ گھریلو یا نگہداشت کے "
                      "کام کا غیر متناسب بوجھ بھی اٹھا رہی ہوتی ہیں۔ کسی ماہر سے ملنے "
                      "کے ساتھ ساتھ، اپنے قریبی شخص کے سامنے کھل کر ایک مخصوص ذمہ "
                      "داری کا نام لیں جس میں آپ کو اس ہفتے مدد چاہیے۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "This level of low mood, especially when tied to navigating "
                      "identity-related stress, deserves real support - and it "
                      "matters to find a mental health professional who is "
                      "explicitly identity-affirming, rather than one you have to "
                      "first educate. It is worth asking directly about this when "
                      "booking, and it does not make your experience any less "
                      "valid to need that.",
                "ur": "اداسی کی یہ سطح، خاص طور پر جب شناخت سے جڑے دباؤ کے ساتھ ہو، "
                      "حقیقی مدد کی مستحق ہے - اور ایسا ذہنی صحت کا ماہر تلاش کرنا "
                      "اہم ہے جو کھلے طور پر شناخت کو تسلیم کرتا ہو، نہ کہ وہ جسے آپ "
                      "کو پہلے سمجھانا پڑے۔ بکنگ کے وقت اس بارے میں براہِ راست پوچھنا "
                      "درست ہے، اور اس کی ضرورت آپ کے تجربے کو کسی بھی طرح کم اہم "
                      "نہیں بناتی۔",
            }
        return None

    return None


# ---- Anxiety: breathing/physiological (Mild) / worry window (High) -------


def _anxiety_tips(severity: str, bracket: Optional[str]) -> list[dict]:
    tips: list[dict] = []

    if severity in _MILD_TIER:
        if bracket == "youth":
            tips.append({
                "en": "When exam stress or notifications spike your heart rate, try "
                      "the 'physiological sigh': two short inhales through your nose "
                      "back-to-back, then one long, slow exhale through your mouth. "
                      "Repeat it 3-4 times - it's one of the fastest known ways to "
                      "calm the nervous system, and you can do it silently at your "
                      "desk between classes.",
                "ur": "جب امتحان کا دباؤ یا نوٹیفیکیشنز آپ کی دل کی دھڑکن بڑھا دیں تو "
                      "'فزیولوجیکل سائی' آزمائیں: ناک سے لگاتار دو مختصر سانسیں "
                      "اندر کھینچیں، پھر منہ سے ایک لمبی، سست سانس باہر چھوڑیں۔ اسے 3 "
                      "سے 4 بار دہرائیں - یہ اعصابی نظام کو پرسکون کرنے کے تیز ترین "
                      "طریقوں میں سے ایک ہے، اور آپ اسے کلاسوں کے درمیان خاموشی سے "
                      "اپنی میز پر کر سکتے ہیں۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "When anxiety spikes mid-workday, try box breathing: inhale for "
                      "4 counts, hold for 4, exhale for 4, hold for 4, and repeat for "
                      "a minute. It's discreet enough to do in a meeting or at your "
                      "desk, and it directly lowers the physical arousal that racing "
                      "thoughts feed on.",
                "ur": "جب کام کے دوران بے چینی بڑھے تو 'باکس بریدنگ' آزمائیں: 4 گنتی "
                      "تک سانس اندر لیں، 4 گنتی تک روکیں، 4 گنتی تک باہر چھوڑیں، 4 "
                      "گنتی تک روکیں، اور اسے ایک منٹ تک دہرائیں۔ یہ اتنی خاموش تکنیک "
                      "ہے کہ آپ اسے میٹنگ میں یا اپنی میز پر بھی کر سکتے ہیں، اور یہ "
                      "اس جسمانی تحریک کو براہِ راست کم کرتی ہے جس سے تیز خیالات "
                      "پرورش پاتے ہیں۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "When worry builds, try pairing slow breathing with gentle "
                      "movement: breathe in for a count of 4 while raising your arms "
                      "slowly, breathe out for a count of 6 while lowering them. Ten "
                      "rounds of this, done seated or standing, is a low-strain way "
                      "to settle the body before the mind follows.",
                "ur": "جب پریشانی بڑھے تو سست سانس کو نرم حرکت کے ساتھ ملائیں: 4 "
                      "گنتی تک سانس اندر لیتے ہوئے آہستہ آہستہ بازو اوپر اٹھائیں، 6 "
                      "گنتی تک سانس باہر چھوڑتے ہوئے انہیں نیچے لائیں۔ بیٹھ کر یا کھڑے "
                      "ہو کر اس کے دس دور کرنا جسم کو کم دباؤ کے ساتھ پرسکون کرنے کا "
                      "طریقہ ہے، اور ذہن بعد میں اس کی پیروی کرتا ہے۔",
            })
        else:
            tips.append({
                "en": "When anxiety builds up, try the 5-4-3-2-1 grounding technique: name "
                      "5 things you can see, 4 you can touch, 3 you can hear, 2 you can "
                      "smell, and 1 you can taste.",
                "ur": "جب بے چینی بڑھے تو 5-4-3-2-1 گراؤنڈنگ تکنیک آزمائیں: 5 چیزیں جو آپ "
                      "دیکھ سکتے ہیں، 4 جنہیں چھو سکتے ہیں، 3 جو سن سکتے ہیں، 2 جن کی خوشبو "
                      "محسوس کر سکتے ہیں، اور 1 جسے چکھ سکتے ہیں، ان کے نام لیں۔",
            })

    elif severity in _HIGH_TIER:
        if bracket == "youth":
            tips.append({
                "en": "This level of anxiety is a lot to carry through classes and "
                      "revision. Try scheduling a daily 10-minute 'worry window' - "
                      "when a worry pops up outside that window, jot it down and "
                      "tell yourself 'I'll think about this at 6pm' instead of "
                      "engaging with it immediately. This CBT technique trains your "
                      "brain that worries don't need instant attention. Please pair "
                      "this with talking to a school counsellor or mental health "
                      "professional about what's underneath the worry.",
                "ur": "بے چینی کی یہ سطح کلاسوں اور امتحانی تیاری کے دوران بہت زیادہ "
                      "ہے۔ روزانہ 10 منٹ کی 'فکر کی کھڑکی' مقرر کرنے کی کوشش کریں - "
                      "جب کوئی فکر اس وقت سے باہر ابھرے، اسے لکھ لیں اور خود سے کہیں "
                      "'میں اس پر شام 6 بجے سوچوں گا' بجائے اس کے کہ فوراً اس میں الجھ "
                      "جائیں۔ یہ CBT تکنیک آپ کے دماغ کو سکھاتی ہے کہ فکروں کو فوری "
                      "توجہ کی ضرورت نہیں۔ براہِ کرم اسے کسی اسکول کونسلر یا ذہنی صحت "
                      "کے ماہر سے فکر کی اصل وجہ پر بات کرنے کے ساتھ ملائیں۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Your anxiety responses are in a higher range, likely fed by "
                      "career and relationship demands running in the background all "
                      "day. Try a daily 'worry window': set aside 10-15 minutes in "
                      "the evening to deliberately think through what's worrying you "
                      "and write down one next step for each - then, when the same "
                      "worry resurfaces during the day, remind yourself it already "
                      "has a slot. Please also bring this to a mental health "
                      "professional, since sustained high anxiety affects sleep and "
                      "physical health too.",
                "ur": "آپ کی بے چینی کے جوابات زیادہ سطح پر ہیں، غالباً کیریئر اور "
                      "تعلقات کے تقاضے دن بھر پس منظر میں چلتے رہنے کی وجہ سے۔ روزانہ "
                      "کی 'فکر کی کھڑکی' آزمائیں: شام کو 10 سے 15 منٹ الگ رکھیں تاکہ "
                      "جان بوجھ کر اپنی پریشانیوں کے بارے میں سوچیں اور ہر ایک کے لیے "
                      "اگلا قدم لکھیں - پھر جب دن میں وہی فکر دوبارہ ابھرے، خود کو "
                      "یاد دلائیں کہ اس کا وقت پہلے سے مقرر ہے۔ براہِ کرم یہ بات کسی "
                      "ذہنی صحت کے ماہر تک بھی پہنچائیں، کیونکہ مسلسل شدید بے چینی نیند "
                      "اور جسمانی صحت کو بھی متاثر کرتی ہے۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "This level of anxiety deserves a conversation with a doctor "
                      "or mental health professional - it's common and treatable, "
                      "not something to manage silently. As a daily practice, try a "
                      "short 'worry window': pick a fixed 10 minutes each day to sit "
                      "with your worries on purpose, write them down, and set them "
                      "aside for the rest of the day - this can stop worry from "
                      "quietly running in the background for hours.",
                "ur": "بے چینی کی یہ سطح کسی ڈاکٹر یا ذہنی صحت کے ماہر سے بات کرنے کی "
                      "مستحق ہے - یہ عام اور قابلِ علاج ہے، اسے خاموشی سے سنبھالنے کی "
                      "ضرورت نہیں۔ روزانہ کی مشق کے طور پر مختصر 'فکر کی کھڑکی' "
                      "آزمائیں: ہر روز جان بوجھ کر 10 منٹ کا مقررہ وقت اپنی پریشانیوں "
                      "کے ساتھ گزارنے کے لیے چنیں، انہیں لکھ لیں، اور باقی دن کے لیے "
                      "الگ رکھ دیں - یہ فکر کو گھنٹوں تک خاموشی سے پس منظر میں چلنے "
                      "سے روک سکتا ہے۔",
            })
        else:
            tips.append({
                "en": "Your anxiety responses are in a higher range. Alongside grounding "
                      "techniques, it may help to talk to a mental health professional "
                      "about what you're experiencing.",
                "ur": "آپ کی بے چینی کے جوابات زیادہ سطح پر ہیں۔ گراؤنڈنگ تکنیکوں کے ساتھ "
                      "ساتھ، ذہنی صحت کے ماہر سے اپنی صورتحال پر بات کرنا مفید ہو سکتا ہے۔",
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
                "en": "Anxiety can show up as restlessness or irritability rather "
                      "than obvious worry, especially when there's pressure to "
                      "'have it handled'. Try channelling that restless energy into "
                      "a short burst of physical activity - push-ups, a fast walk, "
                      "stairs - for 5 minutes when you notice it building; it gives "
                      "the energy somewhere real to go.",
                "ur": "بے چینی اکثر واضح فکر کی بجائے بے چینی یا چڑچڑاپن کی صورت میں "
                      "ظاہر ہوتی ہے، خاص طور پر جب 'سب کچھ سنبھالا ہوا' دکھنے کا دباؤ "
                      "ہو۔ اس بے چین توانائی کو جسمانی سرگرمی کے مختصر پھٹنے میں "
                      "استعمال کرنے کی کوشش کریں - پش اپس، تیز واک، سیڑھیاں - جب "
                      "محسوس ہو کہ یہ بڑھ رہی ہے، 5 منٹ کے لیے؛ یہ توانائی کو ایک "
                      "حقیقی راستہ دیتا ہے۔",
            }
        if bracket == "female":
            return {
                "en": "Anxiety can intensify when you're mentally tracking many "
                      "people's needs at once, not just your own. Try a 'brain dump' "
                      "before bed: write down everything you're holding in mind for "
                      "other people (appointments, reminders, things to arrange) so "
                      "your mind doesn't have to keep rehearsing it overnight.",
                "ur": "بے چینی اس وقت بڑھ سکتی ہے جب آپ ذہنی طور پر کئی لوگوں کی "
                      "ضروریات کو ٹریک کر رہی ہوں، نہ صرف اپنی۔ سونے سے پہلے 'دماغ "
                      "خالی کرنا' آزمائیں: وہ سب کچھ لکھ لیں جو آپ دوسروں کے لیے ذہن "
                      "میں رکھی ہوئی ہیں (ملاقاتیں، یاد دہانیاں، کرنے والے کام) تاکہ "
                      "آپ کے ذہن کو رات بھر اسے دہرانا نہ پڑے۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Anxiety tied to unfamiliar spaces - new people, forms that "
                      "don't fit, unclear reactions - is a real, specific kind of "
                      "vigilance, not just general nervousness. Before entering a "
                      "situation like this, it can help to plan one small anchor in "
                      "advance: a person you can text, or a phrase you'll use if you "
                      "need to step away.",
                "ur": "نامانوس جگہوں سے جڑی بے چینی - نئے لوگ، وہ فارم جو فٹ نہیں "
                      "بیٹھتے، غیر واضح ردِعمل - ایک حقیقی، مخصوص قسم کی چوکسی ہے، "
                      "محض عمومی گھبراہٹ نہیں۔ ایسی صورتحال میں جانے سے پہلے، پہلے "
                      "سے ایک چھوٹا سہارا طے کرنا مددگار ہو سکتا ہے: کوئی شخص جسے آپ "
                      "پیغام بھیج سکیں، یا ایک جملہ جو آپ باہر نکلنے کی ضرورت پر "
                      "استعمال کریں۔",
            }
        return None

    if severity in _HIGH_TIER:
        if bracket == "male":
            return {
                "en": "Persistently high anxiety often gets minimised as 'just "
                      "stress' rather than named and addressed, particularly by men. "
                      "Please consider raising this specifically and directly with a "
                      "doctor or mental health professional, using concrete words "
                      "like 'racing thoughts' or 'can't switch off' rather than "
                      "downplaying it.",
                "ur": "مسلسل شدید بے چینی کو اکثر نام دینے اور حل کرنے کی بجائے "
                      "'محض تناؤ' کہہ کر نظرانداز کر دیا جاتا ہے، خاص طور پر مردوں "
                      "کی طرف سے۔ براہِ کرم اسے واضح طور پر کسی ڈاکٹر یا ذہنی صحت کے "
                      "ماہر کے سامنے بیان کرنے پر غور کریں، ٹھوس الفاظ استعمال کرتے "
                      "ہوئے جیسے 'خیالات تیزی سے دوڑتے ہیں' یا 'ذہن بند نہیں ہوتا'، "
                      "اسے کم اہم بنائے بغیر۔",
            }
        if bracket == "female":
            return {
                "en": "This level of anxiety deserves professional attention, and "
                      "it's also worth asking whether you're being asked to hold too "
                      "many people's logistics and emotions at once. Please talk to "
                      "a mental health professional, and separately, try identifying "
                      "one recurring 'invisible task' you could ask a partner, "
                      "family member, or colleague to take over.",
                "ur": "بے چینی کی یہ سطح پیشہ ورانہ توجہ کی مستحق ہے، اور یہ سوچنا "
                      "بھی ضروری ہے کہ کیا آپ سے ایک وقت میں بہت سے لوگوں کی منصوبہ "
                      "بندی اور جذبات سنبھالنے کو کہا جا رہا ہے۔ براہِ کرم کسی ذہنی "
                      "صحت کے ماہر سے بات کریں، اور الگ سے، کوئی ایک بار بار آنے "
                      "والا 'غیر مرئی کام' پہچانیں جسے آپ کسی ساتھی، خاندان کے فرد، "
                      "یا ساتھی کارکن کے سپرد کر سکتی ہیں۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Constant vigilance in spaces that may not affirm your "
                      "identity is a documented driver of anxiety, not an "
                      "overreaction. When looking for professional support, it is "
                      "reasonable to specifically look for, or ask about, "
                      "identity-affirming care - you shouldn't have to spend therapy "
                      "time explaining the basics of who you are.",
                "ur": "ایسی جگہوں پر مسلسل چوکسی جو آپ کی شناخت کو تسلیم نہیں کر "
                      "سکتیں، بے چینی کی ایک مصدقہ وجہ ہے، ضرورت سے زیادہ ردِعمل "
                      "نہیں۔ پیشہ ورانہ مدد تلاش کرتے وقت، خاص طور پر شناخت کو "
                      "تسلیم کرنے والی دیکھ بھال تلاش کرنا یا اس کے بارے میں پوچھنا "
                      "درست ہے - آپ کو تھراپی کا وقت اپنی بنیادی شناخت سمجھانے میں "
                      "صرف نہیں کرنا چاہیے۔",
            }
        return None

    return None


# ---- Stress: pacing (Mild) / triage & prioritisation (High) --------------


def _stress_tips(severity: str, bracket: Optional[str]) -> list[dict]:
    tips: list[dict] = []

    if severity in _MILD_TIER:
        if bracket == "youth":
            tips.append({
                "en": "Try pacing study sessions the way athletes pace effort: 25 "
                      "minutes of focused work, then a real 5-minute break away from "
                      "screens (stand up, look out a window, get water) before the "
                      "next block. This rhythm - a form of timeboxing - prevents the "
                      "slow build-up of stress that back-to-back cramming causes.",
                "ur": "پڑھائی کے سیشنز کو اسی طرح رفتار دیں جیسے کھلاڑی اپنی توانائی "
                      "بچا کر رکھتے ہیں: 25 منٹ مرکوز کام، پھر اسکرین سے دور 5 منٹ کا "
                      "حقیقی وقفہ (کھڑے ہوں، کھڑکی سے باہر دیکھیں، پانی پئیں) اگلے "
                      "حصے سے پہلے۔ یہ تال - ٹائم باکسنگ کی ایک شکل - مسلسل پڑھائی سے "
                      "پیدا ہونے والے تناؤ کو آہستہ آہستہ بڑھنے سے روکتی ہے۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "At this stage, stress often builds from an ever-growing "
                      "mental to-do list spanning work and home. Try 'task "
                      "chunking': break one large task into its smallest next "
                      "physical step and do only that step - not the whole thing - "
                      "before reassessing. Finishing a small, well-defined piece "
                      "resets the overwhelm that a huge, vague task creates.",
                "ur": "اس عمر میں تناؤ اکثر کام اور گھر کی مسلسل بڑھتی ہوئی ذہنی فہرست "
                      "سے پیدا ہوتا ہے۔ 'ٹاسک چنکنگ' آزمائیں: کسی ایک بڑے کام کو اس "
                      "کے سب سے چھوٹے اگلے عملی قدم میں تقسیم کریں اور دوبارہ جائزہ "
                      "لینے سے پہلے صرف وہی قدم کریں - پورا کام نہیں۔ ایک چھوٹے، واضح "
                      "حصے کو مکمل کرنا اس مغلوبیت کو ختم کرتا ہے جو ایک بڑا، غیر "
                      "واضح کام پیدا کرتا ہے۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "Gentle, low-impact movement - a short walk, light stretching, "
                      "or time in fresh air - lowers stress hormones in a way that's "
                      "kind to the body. Try setting a fixed time each day for this, "
                      "even ten minutes, so it becomes a reliable pressure-release "
                      "rather than something that only happens if the day allows it.",
                "ur": "نرم اور کم دباؤ والی حرکت - مختصر واک، ہلکی سٹریچنگ، یا کھلی "
                      "ہوا میں وقت گزارنا - جسم کے لیے نرمی کے ساتھ تناؤ کے ہارمونز "
                      "کم کرتی ہے۔ اس کے لیے ہر روز ایک مقررہ وقت رکھنے کی کوشش کریں، "
                      "دس منٹ ہی سہی، تاکہ یہ ایک قابلِ اعتماد راحت بن جائے، نہ کہ صرف "
                      "اس وقت ہو جب دن اجازت دے۔",
            })
        else:
            tips.append({
                "en": "Build in short, deliberate breaks through your day - even five "
                      "minutes away from a task can meaningfully lower stress before it "
                      "builds up further.",
                "ur": "اپنے دن میں مختصر اور جان بوجھ کر وقفے شامل کریں - کسی کام سے صرف "
                      "پانچ منٹ دور رہنا بھی تناؤ کو بڑھنے سے پہلے نمایاں طور پر کم کر سکتا ہے۔",
            })

    elif severity in _HIGH_TIER:
        if bracket == "youth":
            tips.append({
                "en": "This level of stress under academic pressure is not "
                      "sustainable long-term. Try a quick triage: list everything "
                      "due this week, then sort each item into 'must do', 'can ask "
                      "for an extension on', or 'can drop/delegate to a study "
                      "partner' - most students overestimate how much is truly "
                      "fixed. Please also talk to a teacher, counsellor, or mental "
                      "health professional about your workload directly.",
                "ur": "تعلیمی دباؤ کے تحت تناؤ کی یہ سطح طویل مدت تک برداشت کرنا ممکن "
                      "نہیں۔ فوری ترجیح بندی آزمائیں: اس ہفتے کی ہر ذمہ داری فہرست "
                      "کریں، پھر ہر چیز کو 'لازمی کرنا ہے'، 'توسیع مانگی جا سکتی ہے'، "
                      "یا 'چھوڑا یا اسٹڈی پارٹنر کے سپرد کیا جا سکتا ہے' میں تقسیم "
                      "کریں - زیادہ تر طلبہ یہ اندازہ زیادہ لگا لیتے ہیں کہ کتنا حقیقتاً "
                      "طے شدہ ہے۔ براہِ کرم اپنے کام کے بوجھ پر کسی استاد، کونسلر، یا "
                      "ذہنی صحت کے ماہر سے براہِ راست بھی بات کریں۔",
            })
        elif bracket == "adult":
            tips.append({
                "en": "Your stress levels appear quite high, likely from carrying "
                      "career and relationship demands simultaneously. Try a simple "
                      "triage exercise: list every current demand on your time, then "
                      "mark each as 'only I can do this', 'could be delegated', or "
                      "'could wait'. Acting on even one item in the second or third "
                      "category this week reduces real load, not just the feeling "
                      "of it. Chronic high stress affects the body too, so please "
                      "also raise this with a mental health professional.",
                "ur": "آپ کی تناؤ کی سطح کافی زیادہ معلوم ہوتی ہے، غالباً کیریئر اور "
                      "تعلقات کے تقاضے بیک وقت اٹھانے کی وجہ سے۔ ایک سادہ ترجیح بندی "
                      "کی مشق آزمائیں: اپنے وقت پر ہر موجودہ دباؤ کی فہرست بنائیں، پھر "
                      "ہر ایک کو 'صرف میں یہ کر سکتا ہوں'، 'کسی اور کے سپرد ہو سکتا "
                      "ہے'، یا 'انتظار کر سکتا ہے' کا نشان لگائیں۔ اس ہفتے دوسری یا "
                      "تیسری قسم میں سے کسی ایک پر عمل کرنا اصل بوجھ کم کرتا ہے، نہ "
                      "کہ صرف اس کا احساس۔ مسلسل شدید تناؤ جسم کو بھی متاثر کرتا "
                      "ہے، اس لیے براہِ کرم یہ بات ذہنی صحت کے ماہر سے بھی بیان کریں۔",
            })
        elif bracket == "senior":
            tips.append({
                "en": "This level of stress deserves attention from a doctor or "
                      "mental health professional, both for your mind and your body. "
                      "As a practical step, try simplifying: pick one commitment or "
                      "obligation you're currently holding onto out of habit rather "
                      "than necessity, and consider letting it go or asking someone "
                      "to share it - fewer, more manageable commitments often ease "
                      "stress more than trying to do everything more efficiently.",
                "ur": "تناؤ کی یہ سطح کسی ڈاکٹر یا ذہنی صحت کے ماہر کی توجہ کی مستحق "
                      "ہے، آپ کے ذہن اور جسم دونوں کے لیے۔ عملی قدم کے طور پر، آسان "
                      "بنانے کی کوشش کریں: کوئی ایک وعدہ یا ذمہ داری چنیں جسے آپ فی "
                      "الحال ضرورت کی بجائے عادت کے تحت نبھا رہے ہیں، اور اسے چھوڑنے "
                      "یا کسی کے ساتھ بانٹنے پر غور کریں - کم مگر قابلِ انتظام ذمہ "
                      "داریاں اکثر تناؤ کو ہر چیز زیادہ مؤثر طریقے سے کرنے کی کوشش سے "
                      "زیادہ کم کرتی ہیں۔",
            })
        else:
            tips.append({
                "en": "Your stress levels appear quite high. Consider what specific "
                      "demands on your time could be reduced or shared, and whether "
                      "talking to someone about your workload would help.",
                "ur": "آپ کی تناؤ کی سطح کافی زیادہ معلوم ہوتی ہے۔ غور کریں کہ آپ کے وقت پر "
                      "کون سے مخصوص دباؤ کم یا کسی کے ساتھ بانٹے جا سکتے ہیں، اور کیا کسی سے "
                      "اپنے کام کے بوجھ پر بات کرنا مددگار ہو سکتا ہے۔",
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
                "en": "There's often social pressure to be the one who 'handles it "
                      "all' without asking for help, which quietly raises stress "
                      "over time. Try asking for help with one specific thing this "
                      "week - not as a last resort, but as a normal, early step.",
                "ur": "اکثر یہ سماجی دباؤ ہوتا ہے کہ بغیر مدد مانگے 'سب کچھ سنبھالا "
                      "جائے'، جو وقت کے ساتھ خاموشی سے تناؤ بڑھاتا ہے۔ اس ہفتے کسی ایک "
                      "مخصوص چیز میں مدد مانگنے کی کوشش کریں - آخری حل کے طور پر نہیں، "
                      "بلکہ ایک معمول کے، ابتدائی قدم کے طور پر۔",
            }
        if bracket == "female":
            return {
                "en": "A large share of everyday stress can come from invisible "
                      "labour - remembering birthdays, restocking supplies, tracking "
                      "everyone's schedules - that rarely gets acknowledged as real "
                      "work. Try making one of these tasks visible: write it down "
                      "and hand it to someone else explicitly, rather than just "
                      "doing it quietly again.",
                "ur": "روزمرہ کے تناؤ کا ایک بڑا حصہ غیر مرئی محنت سے آ سکتا ہے - "
                      "سالگرہیں یاد رکھنا، سامان کا خیال رکھنا، سب کا شیڈول ٹریک "
                      "کرنا - جسے شاذ و نادر ہی حقیقی کام تسلیم کیا جاتا ہے۔ ان میں "
                      "سے کسی ایک کام کو نظر آنے والا بنانے کی کوشش کریں: اسے لکھ کر "
                      "واضح طور پر کسی اور کے سپرد کریں، بجائے اس کے کہ خاموشی سے "
                      "دوبارہ خود کریں۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Stress from navigating unsupportive systems - paperwork, "
                      "misgendering, having to explain yourself repeatedly - is "
                      "real, cumulative load, even when each incident seems small on "
                      "its own. Where possible, try identifying which spaces in your "
                      "week are genuinely low-friction, and protect more time there.",
                "ur": "غیر معاون نظاموں سے نمٹنے کا تناؤ - کاغذی کارروائی، غلط جنس "
                      "سے پکارا جانا، بار بار خود کو سمجھانا پڑنا - ایک حقیقی، جمع "
                      "ہونے والا بوجھ ہے، چاہے ہر واقعہ اکیلا چھوٹا لگے۔ جہاں ممکن "
                      "ہو، اپنے ہفتے میں یہ پہچاننے کی کوشش کریں کہ کون سی جگہیں "
                      "حقیقتاً کم دباؤ والی ہیں، اور وہاں مزید وقت محفوظ رکھیں۔",
            }
        return None

    if severity in _HIGH_TIER:
        if bracket == "male":
            return {
                "en": "Carrying high stress silently, especially around being seen "
                      "as a reliable provider or problem-solver, takes a real toll. "
                      "Please talk to a mental health professional about your "
                      "workload, and consider naming to one person close to you that "
                      "you're stretched thin - not as an admission of failure, but "
                      "as useful information they can act on.",
                "ur": "خاموشی سے شدید تناؤ اٹھانا، خاص طور پر ایک قابلِ اعتماد سہارا "
                      "یا مسئلہ حل کرنے والے کے طور پر دیکھے جانے کے دباؤ میں، حقیقی "
                      "نقصان پہنچاتا ہے۔ براہِ کرم اپنے کام کے بوجھ پر کسی ذہنی صحت "
                      "کے ماہر سے بات کریں، اور اپنے کسی قریبی شخص کو یہ بتانے پر "
                      "غور کریں کہ آپ زیادہ دباؤ میں ہیں - ناکامی کے اعتراف کے طور "
                      "پر نہیں، بلکہ ایک مفید معلومات کے طور پر جس پر وہ عمل کر "
                      "سکیں۔",
            }
        if bracket == "female":
            return {
                "en": "This level of stress, especially if it comes from managing "
                      "everyone else's needs alongside your own, deserves both "
                      "professional support and a real conversation about "
                      "redistributing load. Please talk to a mental health "
                      "professional, and separately, list three recurring "
                      "responsibilities and identify who else could take over at "
                      "least one of them permanently.",
                "ur": "تناؤ کی یہ سطح، خاص طور پر اگر یہ اپنے ساتھ ساتھ سب کی "
                      "ضروریات سنبھالنے سے آ رہی ہو، پیشہ ورانہ مدد اور بوجھ دوبارہ "
                      "تقسیم کرنے پر حقیقی بات چیت دونوں کی مستحق ہے۔ براہِ کرم کسی "
                      "ذہنی صحت کے ماہر سے بات کریں، اور الگ سے، تین بار بار آنے "
                      "والی ذمہ داریاں فہرست کریں اور پہچانیں کہ ان میں سے کم از کم "
                      "ایک کو مستقل طور پر کون اور سنبھال سکتا ہے۔",
            }
        if bracket == "nonbinary":
            return {
                "en": "Sustained high stress from constantly navigating unsupportive "
                      "environments is a well-documented cumulative burden. Please "
                      "talk to a mental health professional, ideally one with "
                      "specific experience supporting gender-diverse clients, and "
                      "consider whether any single recurring source of friction - a "
                      "form, a policy, a relationship - could be addressed directly "
                      "rather than absorbed repeatedly.",
                "ur": "غیر معاون ماحول سے مسلسل نمٹنے کی وجہ سے شدید تناؤ ایک اچھی "
                      "طرح دستاویزی جمع ہونے والا بوجھ ہے۔ براہِ کرم کسی ذہنی صحت کے "
                      "ماہر سے بات کریں، ترجیحاً ایسے جسے صنفی طور پر متنوع مؤکلوں کی "
                      "مدد کا مخصوص تجربہ ہو، اور غور کریں کہ کیا رگڑ کا کوئی ایک "
                      "بار بار آنے والا ذریعہ - کوئی فارم، پالیسی، یا تعلق - براہِ "
                      "راست حل کیا جا سکتا ہے بجائے اسے بار بار برداشت کرنے کے۔",
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
            "en": "Your facial expressions leaned notably positive during this "
                  "session - that's genuinely worth noticing. Try to identify what "
                  "was present right before or during this session (a person, an "
                  "activity, a break from screens) so you can intentionally build "
                  "more of it into your week.",
            "ur": "اس سیشن کے دوران آپ کے چہرے کے تاثرات نمایاں طور پر مثبت رہے - یہ "
                  "واقعی نوٹ کرنے کے قابل بات ہے۔ یہ پہچاننے کی کوشش کریں کہ اس سیشن "
                  "سے پہلے یا دوران کیا موجود تھا (کوئی شخص، سرگرمی، اسکرین سے وقفہ) "
                  "تاکہ آپ اسے جان بوجھ کر اپنے ہفتے میں مزید شامل کر سکیں۔",
        }
    if bracket == "adult":
        return {
            "en": "Your facial expressions leaned notably positive during this "
                  "session - that's worth acknowledging alongside your questionnaire "
                  "results. If this coincided with time away from work demands, it's "
                  "a useful signal for how you might structure future breaks.",
            "ur": "اس سیشن کے دوران آپ کے چہرے کے تاثرات نمایاں طور پر مثبت رہے - یہ "
                  "ایک اچھی بات ہے جسے آپ کے سوالنامے کے نتائج کے ساتھ نوٹ کرنا "
                  "چاہیے۔ اگر یہ کام کے دباؤ سے دور وقت کے ساتھ ہوا، تو یہ ایک مفید "
                  "اشارہ ہے کہ آپ آئندہ وقفے کیسے ترتیب دے سکتے ہیں۔",
        }
    if bracket == "senior":
        return {
            "en": "Your facial expressions leaned notably positive during this "
                  "session - that's genuinely good to see. Whatever surrounded this "
                  "moment - company, a familiar routine, time outdoors - is worth "
                  "protecting and repeating regularly.",
            "ur": "اس سیشن کے دوران آپ کے چہرے کے تاثرات نمایاں طور پر مثبت رہے - یہ "
                  "دیکھ کر واقعی خوشی ہوئی۔ جو کچھ بھی اس لمحے کے ارد گرد تھا - "
                  "ساتھی، جانی پہچانی روٹین، باہر کا وقت - اسے محفوظ رکھنا اور "
                  "باقاعدگی سے دہرانا قابلِ قدر ہے۔",
        }
    return {
        "en": "Your facial expressions leaned notably positive during this "
              "session - that's worth acknowledging alongside your questionnaire "
              "results.",
        "ur": "اس سیشن کے دوران آپ کے چہرے کے تاثرات نمایاں طور پر مثبت رہے - یہ ایک "
              "اچھی بات ہے جسے آپ کے سوالنامے کے نتائج کے ساتھ نوٹ کرنا چاہیے۔",
    }


def _fer_negative_tip(bracket: Optional[str]) -> dict:
    """Technique: affect labelling - naming the feeling in one sentence to
    reduce its intensity. Distinct from the sadness-specific tip below,
    which focuses on a behavioural response rather than a naming exercise."""
    if bracket == "youth":
        return {
            "en": "A notable share of your captured expressions were negative. Try "
                  "'affect labelling': say the feeling to yourself in one plain "
                  "sentence, like 'I'm feeling overwhelmed about the exam' - "
                  "research shows simply naming an emotion this specifically can "
                  "measurably reduce its intensity within minutes.",
            "ur": "آپ کے حاصل کردہ تاثرات کا ایک نمایاں حصہ منفی تھا۔ 'احساس کا نام "
                  "لینا' آزمائیں: اپنے احساس کو ایک سادہ جملے میں کہیں، جیسے 'میں "
                  "امتحان کے بارے میں مغلوب محسوس کر رہا ہوں' - تحقیق بتاتی ہے کہ "
                  "کسی جذبے کو اس طرح واضح طور پر نام دینا چند منٹوں میں اس کی شدت "
                  "کو نمایاں طور پر کم کر سکتا ہے۔",
        }
    if bracket == "adult":
        return {
            "en": "A notable share of your captured expressions were negative. Try "
                  "'affect labelling': in one sentence, name what you're actually "
                  "feeling and what it's about ('I'm frustrated about the deadline "
                  "shifting again') rather than letting it stay vague. Putting a "
                  "specific label on it, even silently, tends to lower its grip "
                  "faster than trying to push through it unnamed.",
            "ur": "آپ کے حاصل کردہ تاثرات کا ایک نمایاں حصہ منفی تھا۔ 'احساس کا نام "
                  "لینا' آزمائیں: ایک جملے میں نام لیں کہ آپ اصل میں کیا محسوس کر رہے "
                  "ہیں اور کیوں ('مجھے ڈیڈلائن دوبارہ بدلنے پر مایوسی ہو رہی ہے') "
                  "بجائے اسے غیر واضح رہنے دینے کے۔ اسے مخصوص نام دینا، خاموشی سے ہی "
                  "سہی، اسے بغیر نام لیے برداشت کرنے کی نسبت تیزی سے ہلکا کر دیتا ہے۔",
        }
    if bracket == "senior":
        return {
            "en": "A notable share of your captured expressions were negative. Try "
                  "'affect labelling': quietly name what you're feeling in a single "
                  "sentence - 'I'm feeling lonely this afternoon' - rather than "
                  "letting it sit unnamed in the background. This simple naming step "
                  "is a well-studied way to ease an emotion's intensity in the "
                  "moment.",
            "ur": "آپ کے حاصل کردہ تاثرات کا ایک نمایاں حصہ منفی تھا۔ 'احساس کا نام "
                  "لینا' آزمائیں: خاموشی سے ایک جملے میں نام لیں کہ آپ کیا محسوس کر "
                  "رہے ہیں - 'آج دوپہر مجھے تنہائی محسوس ہو رہی ہے' - بجائے اسے "
                  "پس منظر میں بغیر نام کے چھوڑنے کے۔ یہ سادہ نام لینے کا عمل کسی "
                  "جذبے کی شدت کو فوری طور پر کم کرنے کا ایک اچھی طرح مطالعہ شدہ "
                  "طریقہ ہے۔",
        }
    return {
        "en": "A notable share of your captured expressions were negative. A "
              "short emotional regulation practice, like slow paced breathing or "
              "briefly naming what you're feeling, can help in the moment.",
        "ur": "آپ کے حاصل کردہ تاثرات کا ایک نمایاں حصہ منفی تھا۔ سست رفتار سانس "
              "لینے یا مختصر طور پر اپنے احساس کا نام لینے جیسی جذباتی ضبط کی مشق "
              "فوری طور پر مددگار ہو سکتی ہے۔",
    }


def _gender_fer_negative_tip(bracket: Optional[str]) -> Optional[dict]:
    """Supplementary, gender-informed tip when negative expression is
    prominent. Technique focus is deliberately different from
    _fer_negative_tip above (affect labelling): this one centres on where
    the feeling is directed (inward suppression vs. outward validation),
    so the two stay non-overlapping."""
    if bracket == "male":
        return {
            "en": "Negative expressions can sometimes get masked as neutral or "
                  "flat, especially when there's pressure not to visibly react. It "
                  "can help to check in with your body specifically - jaw, "
                  "shoulders, hands - since physical tension often reveals feeling "
                  "that a neutral face is hiding.",
            "ur": "منفی تاثرات کبھی کبھار غیر جانبدار یا سپاٹ ظاہر ہو سکتے ہیں، خاص "
                  "طور پر جب واضح ردِعمل نہ دکھانے کا دباؤ ہو۔ اپنے جسم کو خاص طور "
                  "پر جانچنا مددگار ہو سکتا ہے - جبڑا، کندھے، ہاتھ - کیونکہ جسمانی "
                  "تناؤ اکثر وہ احساس ظاہر کرتا ہے جسے غیر جانبدار چہرہ چھپا رہا "
                  "ہوتا ہے۔",
        }
    if bracket == "female":
        return {
            "en": "When negative expressions show up alongside a busy caretaking "
                  "load, it can be easy to keep functioning through it without "
                  "actually addressing it. Try giving yourself explicit permission "
                  "for five minutes today that belong only to you, with no one "
                  "else's needs attached.",
            "ur": "جب منفی تاثرات مصروف نگہداشت کے بوجھ کے ساتھ سامنے آئیں، تو "
                  "اصل میں اسے حل کیے بغیر کام جاری رکھنا آسان ہو سکتا ہے۔ آج اپنے "
                  "لیے واضح طور پر پانچ منٹ کی اجازت دینے کی کوشش کریں جو صرف آپ "
                  "کے ہوں، جن سے کسی اور کی ضروریات وابستہ نہ ہوں۔",
        }
    if bracket == "nonbinary":
        return {
            "en": "Negative expressions after navigating spaces that don't fully "
                  "see you are a valid, real response, not an oversensitivity. "
                  "Consider spending a few minutes afterward with someone or "
                  "something explicitly affirming - a supportive friend, a "
                  "community space - to help reset before moving on with your day.",
            "ur": "ایسی جگہوں سے گزرنے کے بعد منفی تاثرات جو آپ کو مکمل طور پر "
                  "تسلیم نہیں کرتیں، ایک درست، حقیقی ردِعمل ہے، ضرورت سے زیادہ "
                  "حساسیت نہیں۔ اس کے بعد کچھ منٹ کسی ایسے شخص یا چیز کے ساتھ گزارنے "
                  "پر غور کریں جو واضح طور پر تسلیم کرے - کوئی معاون دوست، کمیونٹی "
                  "کی جگہ - تاکہ دن جاری رکھنے سے پہلے دوبارہ سکون مل سکے۔",
        }
    return None


def _fer_sad_tip(bracket: Optional[str]) -> dict:
    """Technique: a concrete behavioural/connection step, kept deliberately
    different from the naming exercise in _fer_negative_tip so the two read
    as distinct actions when both fire in the same result."""
    if bracket == "youth":
        return {
            "en": "Sadness showed up as a clear signal in your expressions during "
                  "this session. Rather than just sitting with it, try one concrete "
                  "step: message one person in your peer group you trust and suggest "
                  "meeting up or calling, even briefly - social comparison online "
                  "can quietly deepen sadness, while one real conversation tends to "
                  "counter it.",
            "ur": "اس سیشن کے دوران اداسی آپ کے تاثرات میں ایک واضح اشارے کے طور پر "
                  "سامنے آئی۔ صرف اس کے ساتھ بیٹھنے کی بجائے، ایک ٹھوس قدم اٹھائیں: "
                  "اپنے قابلِ اعتماد ہم عمر گروپ میں کسی ایک شخص کو پیغام بھیجیں اور "
                  "ملنے یا کال کرنے کی تجویز دیں، مختصر ہی سہی - سوشل میڈیا پر موازنہ "
                  "خاموشی سے اداسی گہری کر سکتا ہے، جبکہ ایک حقیقی گفتگو اس کا مقابلہ "
                  "کرتی ہے۔",
        }
    if bracket == "adult":
        return {
            "en": "Sadness showed up as a clear signal in your expressions during "
                  "this session. As a concrete step, before returning to your task "
                  "list, take five minutes to identify one relationship or "
                  "responsibility that's currently asking more of you than it's "
                  "giving back, and write down one small change you could make to "
                  "it this week.",
            "ur": "اس سیشن کے دوران اداسی آپ کے تاثرات میں ایک واضح اشارے کے طور پر "
                  "سامنے آئی۔ ایک ٹھوس قدم کے طور پر، اپنی فہرست کی طرف واپس جانے سے "
                  "پہلے، پانچ منٹ لیں یہ پہچاننے کے لیے کہ کون سا تعلق یا ذمہ داری "
                  "فی الحال آپ سے واپس ملنے سے کہیں زیادہ مانگ رہی ہے، اور اس میں ایک "
                  "چھوٹی تبدیلی لکھیں جو آپ اس ہفتے کر سکتے ہیں۔",
        }
    if bracket == "senior":
        return {
            "en": "Sadness showed up as a clear signal in your expressions during "
                  "this session. As a concrete step, pick one specific person - "
                  "family member, old friend, or neighbour - and reach out to them "
                  "today, even with a short message. Sadness in later life is often "
                  "connected to reduced daily contact, and rebuilding that contact "
                  "directly tends to help more than waiting for it to happen.",
            "ur": "اس سیشن کے دوران اداسی آپ کے تاثرات میں ایک واضح اشارے کے طور پر "
                  "سامنے آئی۔ ایک ٹھوس قدم کے طور پر، ایک مخصوص شخص چنیں - خاندان کا "
                  "فرد، پرانا دوست، یا پڑوسی - اور آج ہی ان سے رابطہ کریں، مختصر پیغام "
                  "ہی سہی۔ بڑھاپے میں اداسی اکثر روزمرہ رابطے میں کمی سے جڑی ہوتی ہے، "
                  "اور اس رابطے کو براہِ راست دوبارہ تعمیر کرنا اس کے خود بخود ہونے کا "
                  "انتظار کرنے سے زیادہ مددگار ثابت ہوتا ہے۔",
        }
    return {
        "en": "Sadness showed up as a clear signal in your expressions during "
              "this session. Taking a few minutes to acknowledge the feeling "
              "directly, and connecting with someone you trust, can help before "
              "it builds further.",
        "ur": "اس سیشن کے دوران اداسی آپ کے تاثرات میں ایک واضح اشارے کے طور پر "
              "سامنے آئی۔ چند منٹ رک کر اس احساس کو سیدھا تسلیم کرنا، اور کسی "
              "قابلِ اعتماد شخص سے رابطہ کرنا، اسے مزید بڑھنے سے پہلے مددگار ہو "
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