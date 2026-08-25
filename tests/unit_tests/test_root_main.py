"""Unit tests for the repository-level ``python main.py`` entry point."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import call

import pytest

import main as root_main


def test_main_runs_seed_before_server(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(root_main.os, "chdir", lambda _path: events.append("chdir"))
    monkeypatch.setattr(root_main, "run_rag_seed", lambda: events.append("seed"))
    monkeypatch.setattr(root_main, "run_server", lambda: events.append("server"))

    root_main.main()

    assert events == ["chdir", "seed", "server"]


def test_main_does_not_start_server_when_seed_raises(monkeypatch):
    monkeypatch.setattr(root_main.os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        root_main,
        "run_rag_seed",
        lambda: (_ for _ in ()).throw(RuntimeError("seed failed")),
    )
    server_calls: list[str] = []
    monkeypatch.setattr(root_main, "run_server", lambda: server_calls.append("server"))

    with pytest.raises(RuntimeError, match="seed failed"):
        root_main.main()

    assert server_calls == []


def test_seed_tickers_are_normalized_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_SEED_TICKERS", " msft, nvda ,AMD ")

    assert root_main._seed_tickers() == ["MSFT", "NVDA", "AMD"]


def test_run_rag_seed_waits_for_async_seed(monkeypatch):
    received: list[list[str]] = []

    async def fake_seed(tickers: list[str]) -> None:
        received.append(tickers)

    monkeypatch.setenv("RAG_SEED_TICKERS", "MSFT")
    monkeypatch.setattr(root_main, "seed_main", fake_seed)

    root_main.run_rag_seed()

    assert received == [["MSFT"]]


def test_run_server_uses_environment_configuration(monkeypatch):
    calls: list[tuple[tuple, dict]] = []
    fake_uvicorn = SimpleNamespace(
        run=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(root_main, "uvicorn", fake_uvicorn)
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("API_RELOAD", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    root_main.run_server()

    assert calls == [
        call(
            "api.main:app",
            host="0.0.0.0",
            port=9000,
            reload=False,
            log_level="debug",
        )
    ]


@pytest.mark.parametrize("value", ["sometimes", "2", "enabled"])
def test_invalid_reload_value_is_rejected(monkeypatch, value):
    monkeypatch.setenv("API_RELOAD", value)

    with pytest.raises(ValueError, match="API_RELOAD"):
        root_main.run_server()
