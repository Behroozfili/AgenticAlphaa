"""Fail-closed Supabase-compatible JWT authentication boundary."""

from __future__ import annotations

from typing import Any

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse

from api.config import Settings, get_settings


_PUBLIC_PATHS = frozenset({"/", "/health", "/readiness", "/docs", "/redoc", "/openapi.json"})


def _runtime(request: Request) -> Settings:
    configured = getattr(request.app.state, "runtime_settings", None)
    return configured if isinstance(configured, Settings) else get_settings()


def _bearer(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    scheme, separator, token = value.strip().partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def decode_user_token(token: str, settings: Settings) -> str:
    """Verify signature/standard claims and return only the canonical subject."""
    secret = settings.SUPABASE_JWT_SECRET.strip()
    if not secret:
        raise ValueError("JWT verifier is not configured")
    kwargs: dict[str, Any] = {
        "algorithms": ["HS256"],
        "options": {"require": ["exp", "iat", "sub"]},
    }
    audience = settings.JWT_AUDIENCE.strip()
    if audience:
        kwargs["audience"] = audience
    else:
        kwargs["options"]["verify_aud"] = False
    issuer = settings.JWT_ISSUER.strip()
    if issuer:
        kwargs["issuer"] = issuer
    claims = jwt.decode(token, secret, **kwargs)
    subject = claims.get("sub")
    if not isinstance(subject, str) or subject != subject.strip() or not (1 <= len(subject) <= 160):
        raise ValueError("JWT subject is invalid")
    return subject


async def jwt_auth_middleware(request: Request, call_next):
    settings = _runtime(request)
    if request.url.path in _PUBLIC_PATHS or not request.url.path.startswith("/api/"):
        request.state.user_id = settings.DEFAULT_USER_ID
        return await call_next(request)
    token = _bearer(request.headers.get("Authorization"))
    if token is None:
        if settings.AUTH_REQUIRED:
            return JSONResponse(
                status_code=401,
                content={"error": {"code": "AUTH_REQUIRED", "message": "Bearer authentication is required."}},
                headers={"WWW-Authenticate": "Bearer"},
            )
        request.state.user_id = settings.DEFAULT_USER_ID
        return await call_next(request)
    try:
        request.state.user_id = decode_user_token(token, settings)
    except Exception:
        return JSONResponse(
            status_code=401,
            content={"error": {"code": "AUTH_INVALID", "message": "Bearer token is invalid."}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


__all__ = ["decode_user_token", "jwt_auth_middleware"]
