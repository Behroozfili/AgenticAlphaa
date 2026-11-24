"""Versioned, fail-closed persistence for Quant runtime artifacts."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from core.error_handler import VersionMismatchError
from core.forecast_contract import HORIZON_TRADING_DAYS, parse_utc_datetime
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION
from tools.market_tools.forecast_model import (
    ACTION_POLICY_VERSION,
    MINIMUM_TRAINING_CUTOFFS,
    MODEL_FAMILY_VERSION,
    DeterministicModel,
)
from tools.market_tools.historical_dataset_builder import DataReadiness, validate_universe_manifest
from tools.market_tools.market_data import MARKET_DATA_VERSION, canonical_hash
from tools.market_tools.market_server import MarketForecastService


QUANT_ARTIFACT_SCHEMA_VERSION = "quant-runtime-artifact-v1"
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class QuantArtifactError(VersionMismatchError):
    """Stable fail-closed exception for an unusable runtime artifact."""

    error_code = "QUANT_ARTIFACT_REJECTED"


@dataclass(frozen=True, slots=True)
class QuantRuntimeArtifacts:
    model: DeterministicModel
    readiness: DataReadiness
    universe_manifest: Mapping[str, Any]
    model_variant: str
    artifact_hash: str

    def service(self) -> MarketForecastService:
        return MarketForecastService(
            model=self.model,
            readiness=self.readiness,
            universe_manifest_hash=str(self.universe_manifest["manifest_hash"]),
            universe_constituents=tuple(self.universe_manifest["constituents"]),
            model_variant=self.model_variant,
        )


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QuantArtifactError(f"{field} must be an object")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise QuantArtifactError(f"{field} must be canonical non-empty text")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise QuantArtifactError(f"{field} must be an integer >= {minimum}")
    return value


def _floats(value: Any, *, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise QuantArtifactError(f"{field} must contain exactly {length} values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise QuantArtifactError(f"{field} values must be finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise QuantArtifactError(f"{field} values must be finite numbers") from exc
        if not math.isfinite(number):
            raise QuantArtifactError(f"{field} values must be finite numbers")
        result.append(number)
    return tuple(result)


def _sha256(value: Any, *, field: str) -> str:
    result = _text(value, field=field)
    if not _SHA256.fullmatch(result):
        raise QuantArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _readiness_from_mapping(value: Any) -> DataReadiness:
    source = dict(_mapping(value, field="readiness"))
    claimed = _sha256(source.pop("manifest_hash", None), field="readiness.manifest_hash")
    if canonical_hash(source) != claimed:
        raise QuantArtifactError("readiness manifest hash mismatch")
    source["manifest_hash"] = claimed
    readiness = DataReadiness(MappingProxyType(source), claimed)
    try:
        readiness.require_pass()
    except Exception as exc:
        raise QuantArtifactError("runtime readiness status must be PASS") from exc
    if source.get("dataset_version") != MARKET_DATA_VERSION:
        raise QuantArtifactError("readiness dataset version is unsupported")
    if source.get("feature_version") != MARKET_FEATURE_VERSION:
        raise QuantArtifactError("readiness feature version is unsupported")
    return readiness


def _model_from_mapping(value: Any) -> DeterministicModel:
    source = _mapping(value, field="model")
    if source.get("model_family_version") != MODEL_FAMILY_VERSION:
        raise QuantArtifactError("model family version is unsupported")
    if source.get("action_policy_version") != ACTION_POLICY_VERSION:
        raise QuantArtifactError("action policy version is unsupported")
    if source.get("horizon_trading_days") != HORIZON_TRADING_DAYS:
        raise QuantArtifactError("model horizon must be 21 trading sessions")
    raw_names = source.get("feature_names")
    if not isinstance(raw_names, list) or not raw_names:
        raise QuantArtifactError("model.feature_names must be a non-empty list")
    names = tuple(_text(item, field="model.feature_names") for item in raw_names)
    if names != tuple(sorted(set(names))):
        raise QuantArtifactError("model.feature_names must be unique and sorted")
    width = len(names)
    imputer = _mapping(source.get("imputer"), field="model.imputer")
    scaler = _mapping(source.get("scaler"), field="model.scaler")
    logistic = _mapping(source.get("logistic"), field="model.logistic")
    ridge_return = _mapping(source.get("ridge_return"), field="model.ridge_return")
    ridge_excess = _mapping(source.get("ridge_excess_return"), field="model.ridge_excess_return")
    if imputer.get("strategy") != "training_fold_median" or scaler.get("strategy") != "training_fold_standard":
        raise QuantArtifactError("model preprocessing strategy mismatch")
    medians = _floats(imputer.get("statistics"), field="model.imputer.statistics", length=width)
    means = _floats(scaler.get("means"), field="model.scaler.means", length=width)
    scales = _floats(scaler.get("scales"), field="model.scaler.scales", length=width)
    if any(scale <= 0 for scale in scales):
        raise QuantArtifactError("model scales must be positive")
    logistic_coefficients = _floats(
        logistic.get("coefficients_with_intercept"), field="model.logistic.coefficients", length=width + 1,
    )
    return_coefficients = _floats(
        ridge_return.get("coefficients_with_intercept"), field="model.ridge_return.coefficients", length=width + 1,
    )
    excess_coefficients = _floats(
        ridge_excess.get("coefficients_with_intercept"), field="model.ridge_excess.coefficients", length=width + 1,
    )
    interval = _floats(source.get("residual_interval"), field="model.residual_interval", length=2)
    if interval[0] > interval[1]:
        raise QuantArtifactError("model residual interval is reversed")
    training_start = _text(source.get("training_start"), field="model.training_start")
    training_end = _text(source.get("training_end"), field="model.training_end")
    if parse_utc_datetime(training_start) > parse_utc_datetime(training_end):
        raise QuantArtifactError("model training range is reversed")
    training_rows = _integer(source.get("training_rows"), field="model.training_rows", minimum=1)
    training_cutoffs = _integer(
        source.get("training_cutoffs"), field="model.training_cutoffs", minimum=MINIMUM_TRAINING_CUTOFFS,
    )
    if training_rows < training_cutoffs:
        raise QuantArtifactError("model training rows cannot be fewer than cutoffs")
    training_hash = _sha256(source.get("training_data_hash"), field="model.training_data_hash")
    readiness_hash = _sha256(source.get("readiness_manifest_hash"), field="model.readiness_manifest_hash")
    eligible_hash = _sha256(source.get("eligible_rows_hash"), field="model.eligible_rows_hash")

    seed = {
        "family": MODEL_FAMILY_VERSION, "features": names,
        "medians": [round(item, 14) for item in medians],
        "means": [round(item, 14) for item in means],
        "scales": [round(item, 14) for item in scales],
        "logistic": [round(item, 14) for item in logistic_coefficients],
        "ridge_return": [round(item, 14) for item in return_coefficients],
        "ridge_excess": [round(item, 14) for item in excess_coefficients],
        "residual_interval": list(interval), "training_hash": training_hash,
        "readiness_manifest_hash": readiness_hash,
    }
    expected_version = f"{MODEL_FAMILY_VERSION}+sha256:{canonical_hash(seed)}"
    model_version = _text(source.get("model_version"), field="model.model_version")
    if model_version != expected_version:
        raise QuantArtifactError("model version does not match its coefficients and provenance")
    return DeterministicModel(
        feature_names=names, medians=medians, means=means, scales=scales,
        logistic_coefficients=logistic_coefficients,
        return_coefficients=return_coefficients,
        excess_coefficients=excess_coefficients,
        residual_interval=(interval[0], interval[1]), model_version=model_version,
        feature_version=_text(source.get("feature_version"), field="model.feature_version"),
        dataset_version=_text(source.get("dataset_version"), field="model.dataset_version"),
        action_policy_version=ACTION_POLICY_VERSION,
        training_start=training_start, training_end=training_end,
        training_rows=training_rows, training_cutoffs=training_cutoffs,
        training_data_hash=training_hash, readiness_manifest_hash=readiness_hash,
        eligible_rows_hash=eligible_hash,
    )


def build_quant_artifact_bundle(
    *, model: DeterministicModel, readiness: DataReadiness,
    universe_manifest: Mapping[str, Any], model_variant: str = "full_model_v1",
) -> dict[str, Any]:
    """Build canonical bytes for output by an offline training job."""
    universe = validate_universe_manifest(universe_manifest)
    variant = _text(model_variant, field="model_variant")
    MarketForecastService(
        model=model, readiness=readiness,
        universe_manifest_hash=str(universe["manifest_hash"]),
        universe_constituents=tuple(universe["constituents"]), model_variant=variant,
    )
    readiness_payload = dict(readiness.manifest)
    readiness_payload["manifest_hash"] = readiness.manifest_hash
    payload: dict[str, Any] = {
        "artifact_schema_version": QUANT_ARTIFACT_SCHEMA_VERSION,
        "model_variant": variant,
        "model": model.artifact(),
        "readiness": readiness_payload,
        "universe_manifest": universe,
    }
    payload["artifact_hash"] = canonical_hash(payload)
    return payload


def _load_payload(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    try:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise QuantArtifactError("Quant artifact exceeds the size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except QuantArtifactError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise QuantArtifactError("Quant artifact cannot be read") from exc
    return _mapping(payload, field="artifact")


def load_quant_artifacts(
    source: str | Path | Mapping[str, Any], *,
    expected_dataset_version: str | None = None,
    expected_feature_version: str | None = None,
    expected_model_version: str | None = None,
) -> QuantRuntimeArtifacts:
    """Load and cross-check the complete startup artifact or fail closed."""
    try:
        payload = dict(_load_payload(source))
        claimed_hash = _sha256(payload.pop("artifact_hash", None), field="artifact_hash")
        if canonical_hash(payload) != claimed_hash:
            raise QuantArtifactError("Quant artifact content hash mismatch")
        if payload.get("artifact_schema_version") != QUANT_ARTIFACT_SCHEMA_VERSION:
            raise QuantArtifactError("Quant artifact schema version is unsupported")
        readiness = _readiness_from_mapping(payload.get("readiness"))
        model = _model_from_mapping(payload.get("model"))
        universe = validate_universe_manifest(
            _mapping(payload.get("universe_manifest"), field="universe_manifest")
        )
        variant = _text(payload.get("model_variant"), field="model_variant")
        loaded = QuantRuntimeArtifacts(
            model=model, readiness=readiness,
            universe_manifest=MappingProxyType(universe),
            model_variant=variant, artifact_hash=claimed_hash,
        )
        loaded.service()  # performs all model/readiness/universe cross-checks
        expectations = (
            (expected_dataset_version, model.dataset_version, "dataset"),
            (expected_feature_version, model.feature_version, "feature"),
            (expected_model_version, model.model_version, "model"),
        )
        for expected, actual, name in expectations:
            if expected is not None and expected != actual:
                raise QuantArtifactError(f"configured {name} version does not match artifact")
        return loaded
    except QuantArtifactError:
        raise
    except Exception as exc:
        raise QuantArtifactError("Quant runtime artifact validation failed") from exc


def save_quant_artifact_bundle(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Persist once; published artifacts are immutable and never overwritten."""
    # Validate before writing even if the caller assembled the mapping manually.
    load_quant_artifacts(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise QuantArtifactError("Quant artifact path already exists") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


__all__ = [
    "MAX_ARTIFACT_BYTES", "QUANT_ARTIFACT_SCHEMA_VERSION", "QuantArtifactError",
    "QuantRuntimeArtifacts", "build_quant_artifact_bundle", "load_quant_artifacts",
    "save_quant_artifact_bundle",
]
