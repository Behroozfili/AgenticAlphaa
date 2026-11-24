"""Offline CLI for the deterministic Stage 09 walk-forward backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.monthly_backtest import run_monthly_backtest
from evaluation.point_in_time_dataset import Stage09EventSink, load_point_in_time_fixture


def build_report(
    fixture: Path, *, run_id: str, event_sink: Stage09EventSink | None = None,
) -> dict:
    dataset = load_point_in_time_fixture(fixture, event_sink=event_sink, run_id=run_id)
    return run_monthly_backtest(dataset, run_id=run_id, event_sink=event_sink).to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline point-in-time walk-forward evaluation")
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--run-id", default="stage09-r1-a1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.fixture, run_id=args.run_id)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_report", "main"]
