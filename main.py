

from __future__ import annotations

import asyncio
import argparse
import logging
import os
from pathlib import Path
from rag.seed import DEFAULT_TICKERS
from rag.seed import main as seed_main
import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3000

log = logging.getLogger("agenticalpha.entrypoint")


def _env_bool(name: str, *, default: bool) -> bool:
    """Read a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of: true/false, yes/no, on/off, or 1/0"
    )


def _seed_tickers() -> list[str]:

    configured = os.getenv("RAG_SEED_TICKERS", "")
    if not configured.strip():
        return list(DEFAULT_TICKERS)

    tickers = [ticker.strip().upper() for ticker in configured.split(",")]
    return [ticker for ticker in tickers if ticker]


def run_rag_seed() -> None:
    tickers = _seed_tickers()
    if not tickers:
        raise ValueError("RAG_SEED_TICKERS did not contain any valid ticker")

    log.info("rag_seed_started")
    asyncio.run(seed_main(tickers))
    log.info("rag_seed_completed")


def run_server() -> None:
    host = os.getenv("API_HOST", DEFAULT_HOST)
    try:
        port = int(os.getenv("API_PORT", str(DEFAULT_PORT)))
    except ValueError as exc:
        raise ValueError("API_PORT must be an integer") from exc

    reload_enabled = _env_bool("API_RELOAD", default=True)
    log.info(
        "Starting AgenticAlpha API at http://%s:%d (reload=%s)",
        host,
        port,
        reload_enabled,
    )
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
    )


def _configure_process() -> None:
    os.chdir(PROJECT_ROOT)
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    """Backward-compatible server entry point; seeding is never implicit."""
    _configure_process()
    run_server()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenticalpha")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=("serve", "seed-rag"),
        help="start the API or explicitly seed RAG",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    """Run an explicit process command; tests can pass argv without side effects."""
    args = build_parser().parse_args(argv)
    _configure_process()
    if args.command == "seed-rag":
        run_rag_seed()
    else:
        run_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
