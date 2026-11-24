"""Unit tests for the repository-level ``python main.py`` entry point."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import call

import pytest

import main as root_main


def test_main_starts_server_without_auto_seed(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(root_main.os, "chdir", lambda _path: events.append("chdir"))
    monkeypatch.setattr(root_main, "run_rag_seed", lambda: events.append("seed"))
    monkeypatch.setattr(root_main, "run_server", lambda: events.append("server"))

    root_main.main()

    assert events == ["chdir", "server"]


def test_main_does_not_invoke_seed(monkeypatch):
    monkeypatch.setattr(root_main.os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        root_main,
        "run_rag_seed",
        lambda: (_ for _ in ()).throw(RuntimeError("seed failed")),
    )
    server_calls: list[str] = []
    monkeypatch.setattr(root_main, "run_server", lambda: server_calls.append("server"))

    root_main.main()
    assert server_calls == ["server"]


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


def test_seed_logging_does_not_expose_ticker_values(monkeypatch, caplog):
    private = "PRIVATE_SENTINEL"

    async def fake_seed(_tickers):
        return None

    monkeypatch.setenv("RAG_SEED_TICKERS", private)
    monkeypatch.setattr(root_main, "seed_main", fake_seed)
    with caplog.at_level("INFO", logger="agenticalpha.entrypoint"):
        root_main.run_rag_seed()
    assert private not in caplog.text


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


def test_cli_defaults_to_server_and_does_not_seed(monkeypatch):
    events = []
    monkeypatch.setattr(root_main, "_configure_process", lambda: None)
    monkeypatch.setattr(root_main, "run_server", lambda: events.append("server"))
    monkeypatch.setattr(root_main, "run_rag_seed", lambda: events.append("seed"))
    assert root_main.cli([]) == 0
    assert events == ["server"]


def test_seed_command_is_explicit_opt_in(monkeypatch):
    events = []
    monkeypatch.setattr(root_main, "_configure_process", lambda: None)
    monkeypatch.setattr(root_main, "run_server", lambda: events.append("server"))
    monkeypatch.setattr(root_main, "run_rag_seed", lambda: events.append("seed"))
    assert root_main.cli(["seed-rag"]) == 0
    assert events == ["seed"]


@pytest.mark.parametrize("value", ["sometimes", "2", "enabled"])
def test_invalid_reload_value_is_rejected(monkeypatch, value):
    monkeypatch.setenv("API_RELOAD", value)

    with pytest.raises(ValueError, match="API_RELOAD"):
        root_main.run_server()


def test_seed_cli_exposes_temporal_version_and_resume_options():
    from rag.seed import build_parser

    args = build_parser().parse_args([
        "NVDA", "--start", "2024-01-01", "--end", "2024-01-31",
        "--as-of", "2024-02-01", "--dataset-version", "snapshot-v1",
        "--archive-json", "archive.json", "--checkpoint", "checkpoint.json", "--skip-graph",
    ])
    assert args.tickers == ["NVDA"]
    assert args.as_of == "2024-02-01"
    assert args.dataset_version == "snapshot-v1"
    assert args.archive_json == "archive.json"
    assert args.checkpoint == "checkpoint.json"
    assert args.skip_graph is True


def test_json_archive_provider_has_content_addressed_identity_and_fresh_rows(tmp_path):
    import json
    from rag.seed import JsonArchiveProvider

    archive = tmp_path / "archive.json"
    archive.write_text(json.dumps({"documents": [{"ticker": "NVDA", "title": "archived", "metadata": {"nested": "original"}}]}), encoding="utf-8")
    provider = JsonArchiveProvider(archive)
    first = list(provider())
    first[0]["ticker"] = "MUTATED"
    first[0]["metadata"]["nested"] = "MUTATED"
    second = list(provider())

    assert provider.provider_id.startswith("json-archive-sha256:")
    assert len(provider.provider_id.removeprefix("json-archive-sha256:")) == 64
    assert second[0]["ticker"] == "NVDA"
    assert second[0]["metadata"]["nested"] == "original"


def test_json_archive_provider_filters_requested_tickers(tmp_path):
    import json
    from rag.seed import JsonArchiveProvider

    archive = tmp_path / "archive.json"
    archive.write_text(json.dumps({"documents": [
        {"ticker": "NVDA", "title": "requested"},
        {"ticker": "AMD", "title": "unrequested"},
    ]}), encoding="utf-8")

    rows = list(JsonArchiveProvider(archive)(tickers=["nvda"]))

    assert [row["ticker"] for row in rows] == ["NVDA"]


def test_historical_seed_cli_requires_explicit_archive():
    from rag.seed import cli

    with pytest.raises(SystemExit, match="--archive-json"):
        cli([
            "NVDA", "--historical", "--start", "2024-01-01",
            "--end", "2024-01-31", "--as-of", "2024-02-01",
        ])


def test_seed_cli_injects_configured_archive_provider(monkeypatch, tmp_path, capsys):
    import json
    from rag.ingestion import IngestionManifest, StageStatus
    from rag import seed

    archive = tmp_path / "archive.json"
    archive.write_text(json.dumps({"documents": []}), encoding="utf-8")
    received = {}

    async def fake_main(tickers, **kwargs):
        received.update({"tickers": tickers, **kwargs})
        return IngestionManifest(
            run_id="fixture-run", tickers=tickers, dataset_version=kwargs["dataset_version"],
            historical=True, start=kwargs["start"], end=kwargs["end"], as_of=kwargs["as_of"],
            vector=StageStatus("complete", count=0), graph=StageStatus("skipped"),
        )

    monkeypatch.setattr(seed, "main", fake_main)
    exit_code = seed.cli([
        "nvda", "--historical", "--start", "2024-01-01",
        "--end", "2024-01-31", "--as-of", "2024-02-01",
        "--dataset-version", "snapshot-v1", "--archive-json", str(archive), "--skip-graph",
    ])

    assert exit_code == 0
    assert received["tickers"] == ["nvda"]
    assert callable(received["archive_provider"])
    assert received["archive_provider_id"].startswith("json-archive-sha256:")
    assert received["historical"] is True
    assert json.loads(capsys.readouterr().out)["vector"]["status"] == "complete"
