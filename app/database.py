"""
SQLAlchemy engine + session management.
Optimized for serverless Postgres (Neon): small pool, pre-ping enabled so
stale connections dropped by the serverless proxy are transparently recycled.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,   # validate connections before using them (Neon idles/closes conns)
    pool_size=5,
    max_overflow=10,
    pool_recycle=300,     # recycle every 5 minutes
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
