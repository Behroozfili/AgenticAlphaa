from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from api.auth import jwt_auth_middleware
from api.config import Settings, validate_settings
from api.dependencies import get_user_id
from api.routes.analyze import AnalyzeRequest


SECRET = "test-secret-at-least-32-characters-long"


def _client(*, required=True):
    settings = Settings(
        _env_file=None, AUTH_REQUIRED=required,
        SUPABASE_JWT_SECRET=SECRET, JWT_AUDIENCE="authenticated",
    )
    app = FastAPI()
    app.state.runtime_settings = settings
    app.middleware("http")(jwt_auth_middleware)

    @app.get("/api/private")
    async def private(request: Request):
        return {"user_id": get_user_id(request)}

    return TestClient(app, raise_server_exceptions=False)


def _token(**overrides):
    now = datetime.now(UTC)
    claims = {
        "sub": "verified-user-123", "aud": "authenticated",
        "iat": now, "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return jwt.encode(claims, SECRET, algorithm="HS256")


def test_valid_bearer_subject_is_the_only_request_identity():
    response = _client().get(
        "/api/private",
        headers={"Authorization": f"Bearer {_token()}", "X-User-Id": "forged-user"},
    )
    assert response.status_code == 200
    assert response.json() == {"user_id": "verified-user-123"}


def test_missing_or_invalid_bearer_fails_closed_when_required():
    missing = _client().get("/api/private", headers={"X-User-Id": "forged-user"})
    invalid = _client().get("/api/private", headers={"Authorization": "Bearer invalid"})
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "AUTH_REQUIRED"
    assert invalid.status_code == 401
    assert invalid.json()["error"]["code"] == "AUTH_INVALID"


def test_body_user_id_is_not_part_of_analyze_request_contract():
    request = AnalyzeRequest(query="A sufficiently long market question", user_id="forged-user")
    assert not hasattr(request, "user_id")


def test_get_user_id_ignores_unverified_identity_header():
    request = SimpleNamespace(
        state=SimpleNamespace(), headers={"X-User-Id": "forged-user"},
    )
    assert get_user_id(request) != "forged-user"


def test_production_requires_auth_quant_artifact_and_rag_v2():
    base = {
        "_env_file": None, "APP_ENV": "production",
        "ANTHROPIC_API_KEY": "key", "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "service", "NEO4J_URI": "bolt://example",
        "NEO4J_USER": "neo4j", "NEO4J_PASSWORD": "password",
        "QUANT_ARTIFACT_PATH": "", "QUANT_RUNTIME_REQUIRED": False,
        "RAG_VECTOR_SCHEMA": "legacy",
    }
    with pytest.raises(Exception) as raised:
        validate_settings(Settings(**base))
    message = str(raised.value)
    assert "AUTH_REQUIRED=true" in message
    assert "QUANT_RUNTIME_REQUIRED/QUANT_ARTIFACT_PATH" in message
    assert "RAG_VECTOR_SCHEMA=v2" in message

