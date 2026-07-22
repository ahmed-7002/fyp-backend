"""
Centralized application configuration.
Loads values from environment variables / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # Clerk auth
    CLERK_SECRET_KEY: str
    CLERK_PUBLISHABLE_KEY: str
    CLERK_JWKS_URL: str
    CLERK_ISSUER: str

    # CORS
    FRONTEND_ORIGIN: str = "http://localhost:5173"

    # ML (only used if you swap in a custom-trained model - see fer_service.py)
    FER_MODEL_PATH: str = "./app/ml_models/fer2013_model.h5"

    # App
    ENV: str = "development"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton so the .env file is only parsed once."""
    return Settings()
