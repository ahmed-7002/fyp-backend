"""
Verifies Clerk-issued session JWTs sent from the React frontend
(Authorization: Bearer <token>) using Clerk's published JWKS.
"""
from functools import lru_cache

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError

from app.config import get_settings

bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=1)
def _get_jwks() -> dict:
    settings = get_settings()
    resp = httpx.get(settings.CLERK_JWKS_URL, timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def _find_signing_key(jwks: dict, kid: str) -> dict | None:
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    return None


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """
    FastAPI dependency: validates the Clerk JWT and returns the Clerk user id.
    Raises 401 if the token is missing/invalid/expired.
    """
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")

    token = credentials.credentials
    settings = get_settings()

    try:
        unverified_header = jwt.get_unverified_header(token)
        jwks = _get_jwks()
        signing_key = _find_signing_key(jwks, unverified_header.get("kid"))
        if signing_key is None:
            # JWKS may have rotated - refresh once and retry
            _get_jwks.cache_clear()
            jwks = _get_jwks()
            signing_key = _find_signing_key(jwks, unverified_header.get("kid"))
        if signing_key is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown signing key")

        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=settings.CLERK_ISSUER,
            options={"verify_aud": False},
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token missing subject")
        return user_id

    except JWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {exc}") from exc
