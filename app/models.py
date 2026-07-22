"""
ORM models. The `assessments` table is intentionally flat and wide -
one row per completed assessment - so it can be exported directly as a
clean, ready-to-use training dataset for a future model.
"""
import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Enum,
    func,
)
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Assessment(Base):
    __tablename__ = "assessments"

    # --- Identity ------------------------------------------------------
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    clerk_user_id = Column(String(64), nullable=False, index=True)

    # --- Demographics ----------------------------------------------------
    full_name = Column(String(120), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(
        Enum("male", "female", "non_binary", "prefer_not_to_say", name="gender_enum"),
        nullable=False,
    )

    # --- Assessment mode ---------------------------------------------------
    assessment_mode = Column(
        Enum("questionnaire", "video", "combined", name="assessment_mode_enum"),
        nullable=False,
    )

    # --- DASS-21 raw responses (each 0-3, per official DASS-21 scale) ------
    dass_q1 = Column(Integer)
    dass_q2 = Column(Integer)
    dass_q3 = Column(Integer)
    dass_q4 = Column(Integer)
    dass_q5 = Column(Integer)
    dass_q6 = Column(Integer)
    dass_q7 = Column(Integer)
    dass_q8 = Column(Integer)
    dass_q9 = Column(Integer)
    dass_q10 = Column(Integer)
    dass_q11 = Column(Integer)
    dass_q12 = Column(Integer)
    dass_q13 = Column(Integer)
    dass_q14 = Column(Integer)
    dass_q15 = Column(Integer)
    dass_q16 = Column(Integer)
    dass_q17 = Column(Integer)
    dass_q18 = Column(Integer)
    dass_q19 = Column(Integer)
    dass_q20 = Column(Integer)
    dass_q21 = Column(Integer)

    # --- DASS-21 computed sub-scores (raw score x2 per official scoring) ---
    dass_depression_score = Column(Integer)
    dass_anxiety_score = Column(Integer)
    dass_stress_score = Column(Integer)
    dass_depression_severity = Column(String(20))   # Normal/Mild/Moderate/Severe/Extremely Severe
    dass_anxiety_severity = Column(String(20))
    dass_stress_severity = Column(String(20))

    # --- FER-2013 aggregated metrics (% of analyzed frames per emotion) ----
    fer_frames_captured = Column(Integer)
    fer_frames_analyzed = Column(Integer)     # frames where a face was detected
    fer_angry = Column(Float)
    fer_disgust = Column(Float)
    fer_fear = Column(Float)
    fer_happy = Column(Float)
    fer_sad = Column(Float)
    fer_surprise = Column(Float)
    fer_neutral = Column(Float)
    fer_dominant_emotion = Column(String(20))

    # --- Final combined result -------------------------------------------
    final_risk_level = Column(String(20))     # Low / Moderate / High / Needs Attention
    final_summary = Column(String(1000))

    # --- Timestamps -----------------------------------------------------
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
