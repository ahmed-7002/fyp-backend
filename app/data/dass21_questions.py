"""
The official 21-item DASS-21 scale.
Index positions below are 1-based to match the `dass_qN` naming convention
used everywhere else (question 1 -> answers[0], etc.).
"""

DASS21_QUESTIONS = [
    "I found it hard to wind down",
    "I was aware of dryness of my mouth",
    "I couldn't seem to experience any positive feeling at all",
    "I experienced breathing difficulty (e.g. excessively rapid breathing, "
    "breathlessness in the absence of physical exertion)",
    "I found it difficult to work up the initiative to do things",
    "I tended to over-react to situations",
    "I experienced trembling (e.g. in the hands)",
    "I felt that I was using a lot of nervous energy",
    "I was worried about situations in which I might panic and make a fool of myself",
    "I felt that I had nothing to look forward to",
    "I found myself getting agitated",
    "I found it difficult to relax",
    "I felt down-hearted and blue",
    "I was intolerant of anything that kept me from getting on with what I was doing",
    "I felt I was close to panic",
    "I was unable to become enthusiastic about anything",
    "I felt I wasn't worth much as a person",
    "I felt that I was rather touchy",
    "I was aware of the action of my heart in the absence of physical exertion "
    "(e.g. sense of heart rate increase, heart missing a beat)",
    "I felt scared without any good reason",
    "I felt that life was meaningless",
]

# 1-based question numbers belonging to each DASS-21 subscale (official scoring key)
DEPRESSION_ITEMS = [3, 5, 10, 13, 16, 17, 21]
ANXIETY_ITEMS = [2, 4, 7, 9, 15, 19, 20]
STRESS_ITEMS = [1, 6, 8, 11, 12, 14, 18]

ANSWER_LABELS = [
    "Did not apply to me at all",
    "Applied to me to some degree, or some of the time",
    "Applied to me to a considerable degree, or a good part of time",
    "Applied to me very much, or most of the time",
]
