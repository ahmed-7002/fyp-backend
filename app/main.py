from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import Base, engine
from app.routers import assessment, fer

settings = get_settings()

app = FastAPI(
    title="Mental Health Assessment API",
    description=(
        "Backend for a DASS-21 + FER-2013 mental health screening tool. "
        "Not a diagnostic instrument - see disclaimer on every response."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessment.router)
app.include_router(fer.router)


@app.on_event("startup")
def on_startup():
    # Creates tables if they don't exist yet. For production, prefer Alembic
    # migrations (see backend/README.md) instead of relying on this.
    Base.metadata.create_all(bind=engine)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "disclaimer": (
            "This is a question-based screening tool, not a professional "
            "diagnostic instrument. For a professional assessment, please "
            "consult a medical doctor."
        ),
    }
