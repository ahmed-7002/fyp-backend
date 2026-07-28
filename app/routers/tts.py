"""
Server-side text-to-speech fallback, used only when the visitor's own
browser has no native voice available for the selected language (the
frontend's Web Speech API check handles that detection - see
DassQuestionnaire.jsx). This endpoint exists specifically to cover cases
like Urdu, which many Windows machines have no built-in voice for at all.

Uses gTTS (Google Text-to-Speech) - a lightweight wrapper around Google
Translate's TTS functionality. Two things worth knowing:
  - It requires live internet access from wherever this backend runs.
  - It uses an undocumented endpoint (not the official paid Google Cloud
    TTS API), so treat it as "very reliable in practice for a project like
    this" rather than a formally guaranteed contract.
"""
import io
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from gtts import gTTS

from app.auth import get_current_user

router = APIRouter(prefix="/api", tags=["tts"])

# Only the languages this project's questionnaire actually supports.
SUPPORTED_LANGS = {"en", "ur"}
MAX_TEXT_LENGTH = 500  # comfortably longer than any single DASS-21 item


@lru_cache(maxsize=128)
def _synthesize(text: str, lang: str) -> bytes:
    """
    Generates and caches spoken audio for one (text, lang) pair.

    Caching matters here specifically: the DASS-21 has only 21 fixed
    question strings per language, so once each has been spoken once,
    every later request for that same question - by this user or anyone
    else - is served instantly from memory instead of calling Google's
    servers again.
    """
    buffer = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buffer)
    return buffer.getvalue()


@router.get("/tts")
def synthesize_speech(
    text: str = Query(..., min_length=1, max_length=MAX_TEXT_LENGTH),
    lang: str = Query(...),
    _user_id: str = Depends(get_current_user),
):
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            400, f"Unsupported language '{lang}'. Supported: {sorted(SUPPORTED_LANGS)}"
        )

    try:
        audio_bytes = _synthesize(text, lang)
    except Exception as exc:  # noqa: BLE001 - most commonly: no internet, or Google's
        # undocumented endpoint temporarily rate-limiting/unavailable
        raise HTTPException(
            503,
            "Couldn't generate speech right now - please check your internet "
            f"connection and try again. ({exc})",
        ) from exc

    return Response(content=audio_bytes, media_type="audio/mpeg")