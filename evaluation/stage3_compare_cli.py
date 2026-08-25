"""Compare a baseline trace with a candidate financial-agent artifact.

The quality gate intentionally counts only valid metrics that expose both a
clear period and a clear source.  It does not reward a report merely for
containing more numeric values.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluation.valid_availability import compare_valid_availability


def _load_artifact(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
        if not match:
            raise ValueError(f"No fenced JSON object found in {path}")
        payload = json.loads(match.group(1))
    summary = payload.get("financial_metrics_summary")
    return summary if isinstance(summary, dict) else payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--label", default="comparison")
    args = parser.parse_args()
    result = compare_valid_availability(
        _load_artifact(args.baseline), _load_artifact(args.candidate)
    )
    compact = {
        "label": args.label,
        "baseline_available": result["before"]["available_metric_count"],
        "baseline_valid_with_period_and_source": result["before"][
            "valid_with_period_and_source_count"
        ],
        "candidate_available": result["after"]["available_metric_count"],
        "candidate_valid_with_period_and_source": result["after"][
            "valid_with_period_and_source_count"
        ],
        "valid_delta": result["valid_with_period_and_source_delta"],
        "lost_valid_metrics": result["lost_valid_with_period_and_source"],
        "passes_non_regression": result["passes_non_regression"],
    }
    print(json.dumps(compact, indent=2))
    return 0 if result["passes_non_regression"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
