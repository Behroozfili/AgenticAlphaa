"""Explicit CLI for live or reproducible local-archive RAG ingestion."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from rag.ingestion import IngestionManifest, run_ingestion_pipeline
from rag.loader import ArchiveProvider, RawDocument

DEFAULT_TICKERS = [ "MSFT"]


class JsonArchiveProvider:
    """Deterministic, offline archive provider backed by one immutable JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        payload_bytes = self.path.read_bytes()
        payload = json.loads(payload_bytes.decode("utf-8"))
        documents = payload.get("documents") if isinstance(payload, dict) else payload
        if not isinstance(documents, list) or not all(isinstance(item, dict) for item in documents):
            raise ValueError("archive JSON must be a list or an object with a documents list")
        self._documents: tuple[dict[str, Any], ...] = tuple(dict(item) for item in documents)
        self.provider_id = f"json-archive-sha256:{hashlib.sha256(payload_bytes).hexdigest()}"

    def __call__(self, **boundaries: Any) -> Iterable[RawDocument | Mapping[str, Any]]:
        # AlphaLoader independently enforces ticker/time/dataset boundaries.
        # Filtering here is defense in depth and avoids materializing unrelated
        # archive rows; fresh mappings prevent callers mutating the snapshot.
        requested = {
            str(ticker).strip().upper()
            for ticker in boundaries.get("tickers", [])
            if str(ticker).strip()
        }
        return [
            copy.deepcopy(document)
            for document in self._documents
            if not requested or str(document.get("ticker", "")).strip().upper() in requested
        ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.seed", description="Run explicit, resumable RAG ingestion")
    parser.add_argument("tickers", nargs="*", help="Ticker symbols (default: curated live set)")
    parser.add_argument("--start", help="Inclusive historical publication boundary")
    parser.add_argument("--end", help="Inclusive historical publication boundary")
    parser.add_argument("--as-of", dest="as_of", help="Point-in-time knowledge boundary")
    parser.add_argument("--dataset-version", default="live-dataset-v1")
    parser.add_argument("--archive-json", help="Immutable local JSON archive used in historical mode")
    parser.add_argument("--checkpoint", help="Local manifest/checkpoint JSON path")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--historical", action="store_true", help="Require temporal boundaries and --archive-json; live fallback stays disabled")
    return parser


async def main(
    tickers: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    as_of: str | None = None,
    dataset_version: str = "live-dataset-v1",
    checkpoint_path: str | None = None,
    skip_graph: bool = False,
    historical: bool | None = None,
    archive_provider: ArchiveProvider | None = None,
    archive_provider_id: str | None = None,
) -> IngestionManifest:
    return await run_ingestion_pipeline(
        tickers=[ticker.upper() for ticker in tickers],
        skip_graph=skip_graph,
        start=start,
        end=end,
        as_of=as_of,
        dataset_version=dataset_version,
        historical=historical,
        checkpoint_path=checkpoint_path,
        archive_provider=archive_provider,
        archive_provider_id=archive_provider_id,
    )


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    historical = args.historical or args.archive_json is not None or any(value is not None for value in (args.start, args.end, args.as_of))
    if historical and not all((args.start, args.end, args.as_of)):
        raise SystemExit("historical ingestion requires --start, --end, and --as-of")
    if historical and not args.archive_json:
        raise SystemExit("historical ingestion requires --archive-json; live fallback is disabled")
    provider = JsonArchiveProvider(args.archive_json) if args.archive_json else None
    manifest = asyncio.run(main(
        args.tickers or DEFAULT_TICKERS,
        start=args.start,
        end=args.end,
        as_of=args.as_of,
        dataset_version=args.dataset_version,
        checkpoint_path=args.checkpoint,
        skip_graph=args.skip_graph,
        historical=historical,
        archive_provider=provider,
        archive_provider_id=provider.provider_id if provider else None,
    ))
    print(json.dumps(asdict(manifest), sort_keys=True, indent=2))
    return 0 if manifest.vector.status in {"complete", "pending"} and manifest.graph.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(cli())


__all__ = ["DEFAULT_TICKERS", "JsonArchiveProvider", "build_parser", "cli", "main"]
