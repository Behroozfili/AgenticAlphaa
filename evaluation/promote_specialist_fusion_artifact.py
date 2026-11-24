"""Wrap an existing immutable Quant model with the audited specialist fusion contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.market_tools.quant_artifacts import (
    build_quant_artifact_bundle,
    load_quant_artifacts,
    save_quant_artifact_bundle,
)
from tools.market_tools.specialist_fusion import SPECIALIST_FUSION_VARIANT_TOKEN


def promote(source: Path, output: Path) -> Path:
    artifacts = load_quant_artifacts(source)
    base_variant = artifacts.model_variant.split(f"+{SPECIALIST_FUSION_VARIANT_TOKEN}", 1)[0]
    payload = build_quant_artifact_bundle(
        model=artifacts.model,
        readiness=artifacts.readiness,
        universe_manifest=artifacts.universe_manifest,
        model_variant=f"{base_variant}+{SPECIALIST_FUSION_VARIANT_TOKEN}",
    )
    save_quant_artifact_bundle(output, payload)
    print(f"artifact={output}")
    print(f"base_model_version={artifacts.model.model_version}")
    print(f"model_variant={payload['model_variant']}")
    print("specialists=research,financial,sentiment")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    promote(args.source, args.output)


if __name__ == "__main__":
    main()
