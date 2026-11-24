"""Immutable point-in-time snapshots and the pre-fit DATA_READINESS gate."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError, VersionMismatchError
from core.forecast_contract import (
    CALENDAR_VERSION,
    ForecastContext,
    HORIZON_TRADING_DAYS,
    compute_forward_return_labels,
    nth_trading_session_after,
    parse_utc_datetime,
    regular_session_close_utc,
)
from tools.market_tools import (quant_lifecycle_event, validate_evidence_references,
                                validate_run_id)
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION, MarketFeatures
from tools.market_tools.market_data import MARKET_DATA_VERSION, MarketDataBundle, canonical_hash


DATASET_BUILDER_VERSION = "point-in-time-dataset-v1"
MONTH_END_CUTOFF_POLICY_VERSION = "monthly_last_session_v1"
MONTH_TO_DATE_CUTOFF_POLICY_VERSION = "monthly_to_date_v1"
CUTOFF_POLICY_VERSION = MONTH_END_CUTOFF_POLICY_VERSION
SUPPORTED_CUTOFF_POLICY_VERSIONS = frozenset({
    MONTH_END_CUTOFF_POLICY_VERSION,
    MONTH_TO_DATE_CUTOFF_POLICY_VERSION,
})
LABEL_CONTRACT_VERSION = "forward_21d_excess_return-v1"
# Adjusted-close division/subtraction can differ by a few binary floating-point
# ulps after JSON round trips. This absolute-only tolerance is deterministic;
# relative tolerance is deliberately zero so material discrepancies fail closed.
LABEL_RETURN_ABS_TOLERANCE = 1e-12
FULL_FEATURE_MINIMUM_TICKERS = 4
FULL_FEATURE_MINIMUM_CUTOFFS_PER_TICKER = 36
FULL_FEATURE_MINIMUM_FAMILY_COVERAGE = 0.80
FULL_FEATURE_MINIMUM_GRAPH_COVERAGE = 0.80
FULL_FEATURE_FAMILIES = MappingProxyType({
    "research": (
        "research_source_diversity", "research_vector_coverage",
        "research_graph_coverage", "research_catalyst_count", "research_risk_count",
    ),
    "financial": (
        "financial_valuation_gap", "financial_revenue_growth",
        "financial_earnings_growth", "financial_roe_pct", "financial_net_margin_pct",
        "financial_roic_pct", "financial_net_debt_to_ebitda", "financial_debt_ratio",
        "financial_current_ratio", "financial_quick_ratio", "financial_fcf_margin_pct",
        "financial_fcf_yield_pct", "financial_ocf_ratio", "financial_filing_age_days",
    ),
    "sentiment": (
        "sentiment_aggregate_7d", "sentiment_aggregate_14d",
        "sentiment_aggregate_30d", "sentiment_momentum_7d_vs_30d",
        "sentiment_dispersion_30d", "sentiment_volume_7d", "sentiment_volume_14d",
        "sentiment_volume_30d", "sentiment_source_diversity_30d",
        "sentiment_confidence",
    ),
})


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_universe_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a real versioned universe manifest and return its canonical copy."""
    version = str(payload.get("universe_version") or "").strip()
    source = str(payload.get("source") or "").strip()
    if not version or version == "UNSET" or not source or source == "UNSET":
        raise ValueError("an approved universe version and source are required")
    constituents = payload.get("constituents")
    if not isinstance(constituents, list) or not constituents:
        raise ValueError("universe constituents must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for item in constituents:
        if not isinstance(item, Mapping):
            raise TypeError("universe constituents must be objects")
        ticker = str(item.get("ticker") or "").strip().upper()
        benchmark = str(item.get("benchmark") or "").strip().upper()
        source_ref = str(item.get("source_ref") or "").strip()
        if not ticker or not benchmark or not source_ref or source_ref == "UNSET":
            raise ValueError("each universe constituent requires ticker, benchmark and source_ref")
        try:
            effective_from = date.fromisoformat(str(item.get("effective_from")))
            effective_to = date.fromisoformat(str(item["effective_to"])) if item.get("effective_to") else None
        except ValueError as exc:
            raise ValueError("universe effective intervals must use ISO-8601 dates") from exc
        if effective_to is not None and effective_to < effective_from:
            raise ValueError("universe effective_to cannot precede effective_from")
        normalized.append({
            "ticker": ticker, "effective_from": effective_from.isoformat(),
            "effective_to": effective_to.isoformat() if effective_to else None, "benchmark": benchmark,
            "source_ref": source_ref,
        })
    normalized.sort(key=lambda item: (item["ticker"], str(item["effective_from"])))
    result = {
        "schema_version": str(payload.get("schema_version") or "1.0"),
        "universe_version": version, "source": source,
        "created_at": payload.get("created_at"), "constituents": normalized,
        "limitations": list(payload.get("limitations") or []),
    }
    computed = _json_hash(result)
    claimed = str(payload.get("manifest_hash") or "")
    if claimed and claimed != computed:
        raise ValueError("universe manifest hash mismatch")
    result["manifest_hash"] = computed
    return result


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    ticker: str
    as_of: str
    target_date: str
    horizon_trading_days: int
    benchmark: str
    analysis_mode: str
    dataset_version: str
    feature_version: str
    values: Mapping[str, float | None]
    evidence_ids: tuple[str, ...]
    missing_indicators: Mapping[str, bool]
    degraded: bool
    finalized_at: str
    snapshot_hash: str
    schema_version: str = "forecast-v1"
    calendar_version: str = CALENDAR_VERSION
    feature_materialized_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", validate_evidence_references(self.evidence_ids))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker, "as_of": self.as_of,
            "target_date": self.target_date,
            "horizon_trading_days": self.horizon_trading_days,
            "benchmark": self.benchmark, "analysis_mode": self.analysis_mode,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "values": dict(self.values), "evidence_ids": list(self.evidence_ids),
            "missing_indicators": dict(self.missing_indicators),
            "degraded": self.degraded, "finalized_at": self.finalized_at,
            "schema_version": self.schema_version,
            "calendar_version": self.calendar_version,
            "feature_materialized_at": self.feature_materialized_at,
        }


@dataclass(frozen=True, slots=True)
class LabeledSnapshot:
    snapshot: FeatureSnapshot
    ticker_total_return: float
    benchmark_total_return: float
    forward_21d_excess_return: float
    direction_positive: int
    outcome_materialized_at: str


def _canonical_label_identity_reason(row: LabeledSnapshot) -> str | None:
    """Return the fail-closed reason when a materialized label is noncanonical."""
    label_values = (
        row.ticker_total_return,
        row.benchmark_total_return,
        row.forward_21d_excess_return,
    )
    if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) for value in label_values)
            or type(row.direction_positive) is not int
            or row.direction_positive not in (0, 1)):
        return "missing_price_label"
    canonical_excess_return = (
        float(row.ticker_total_return) - float(row.benchmark_total_return)
    )
    if not math.isclose(
        float(row.forward_21d_excess_return),
        canonical_excess_return,
        rel_tol=0.0,
        abs_tol=LABEL_RETURN_ABS_TOLERANCE,
    ):
        return "inconsistent_excess_return"
    canonical_direction = 1 if canonical_excess_return > 0.0 else 0
    if row.direction_positive != canonical_direction:
        return "inconsistent_direction_label"
    return None


def _require_canonical_label_identity(row: LabeledSnapshot) -> None:
    reason = _canonical_label_identity_reason(row)
    if reason is not None:
        raise ValueError(f"canonical label identity violation: {reason}")


@dataclass(frozen=True, slots=True)
class DataReadiness:
    manifest: Mapping[str, Any]
    manifest_hash: str

    @property
    def status(self) -> str:
        return str(self.manifest["status"])

    def require_pass(self) -> None:
        if self.status != "PASS":
            raise DataUnavailableAsOfError("DATA_READINESS must be PASS before model fitting")


def _build_feature_snapshot(
    *, context: ForecastContext, features: MarketFeatures, finalized_at: datetime | str,
) -> FeatureSnapshot:
    expected_as_of = _iso(context.as_of)
    if (features.ticker != context.ticker or features.as_of != expected_as_of
            or features.dataset_version != context.dataset_version
            or features.feature_version != context.feature_version
            or features.schema_version != context.schema_version
            or features.calendar_version != CALENDAR_VERSION
            or features.benchmark != context.benchmark
            or features.analysis_mode != context.analysis_mode.value):
        raise VersionMismatchError("market features do not match ForecastContext identity")
    if features.feature_version != MARKET_FEATURE_VERSION:
        raise VersionMismatchError("market feature version mismatch")
    finalized = parse_utc_datetime(finalized_at, field_name="finalized_at")
    if finalized > context.as_of:
        raise TemporalLeakageError("feature snapshot must be finalized no later than as_of")
    numeric = features.numeric_vector()
    payload = {
        "ticker": context.ticker, "as_of": _iso(context.as_of),
        "target_date": context.target_date.isoformat(),
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "benchmark": context.benchmark, "analysis_mode": context.analysis_mode.value,
        "dataset_version": context.dataset_version,
        "feature_version": features.feature_version,
        "values": dict(sorted(numeric.items())),
        "evidence_ids": list(features.evidence_ids),
        "missing_indicators": dict(features.missing_indicators),
        "degraded": features.degraded, "finalized_at": _iso(finalized),
        "schema_version": context.schema_version,
        "calendar_version": CALENDAR_VERSION,
        "feature_materialized_at": features.materialized_at,
    }
    return FeatureSnapshot(
        ticker=context.ticker, as_of=payload["as_of"], target_date=payload["target_date"],
        horizon_trading_days=HORIZON_TRADING_DAYS, benchmark=context.benchmark,
        analysis_mode=context.analysis_mode.value, dataset_version=context.dataset_version,
        feature_version=features.feature_version,
        values=MappingProxyType(payload["values"]), evidence_ids=features.evidence_ids,
        missing_indicators=MappingProxyType(dict(features.missing_indicators)),
        degraded=features.degraded, finalized_at=payload["finalized_at"],
        snapshot_hash=_json_hash(payload),
        schema_version=context.schema_version, calendar_version=CALENDAR_VERSION,
        feature_materialized_at=features.materialized_at,
    )


def build_feature_snapshot(
    *, context: ForecastContext, features: MarketFeatures, finalized_at: datetime | str,
    run_id: str | None = None,
) -> FeatureSnapshot:
    started_at = time.perf_counter()
    identity = validate_run_id(run_id) if run_id is not None else None
    try:
        result = _build_feature_snapshot(context=context, features=features, finalized_at=finalized_at)
    except Exception:
        quant_lifecycle_event(context=context, boundary="snapshot_finalization", started_at=started_at,
            outcome="fail_closed", run_id=identity, error_code="QUANT_SNAPSHOT_REJECTED")
        raise
    quant_lifecycle_event(context=context, boundary="snapshot_finalization", started_at=started_at,
        outcome="degraded" if result.degraded else "success", run_id=identity,
        evidence_count=len(result.evidence_ids), counts={"feature_count": len(result.values)},
        degraded_flags=["optional_feature_degradation"] if result.degraded else [])
    return result


def _materialize_outcome(
    *, snapshot: FeatureSnapshot, market: MarketDataBundle,
    outcome_materialized_at: datetime | str,
) -> LabeledSnapshot:
    observed = parse_utc_datetime(outcome_materialized_at, field_name="outcome_materialized_at")
    target = datetime.fromisoformat(snapshot.target_date).date()
    if observed < regular_session_close_utc(target):
        raise TemporalLeakageError("outcome cannot be materialized before target_date")
    if parse_utc_datetime(snapshot.finalized_at, field_name="finalized_at") >= observed:
        raise TemporalLeakageError("forecast snapshot must exist before outcome materialization")
    as_of_day = datetime.fromisoformat(snapshot.as_of.replace("Z", "+00:00")).date()
    labels = compute_forward_return_labels(
        ticker_start_adjusted_close=market.ticker.adjusted_close_on(as_of_day),
        ticker_target_adjusted_close=market.ticker.adjusted_close_on(target),
        benchmark_start_adjusted_close=market.benchmark.adjusted_close_on(as_of_day),
        benchmark_target_adjusted_close=market.benchmark.adjusted_close_on(target),
    )
    return LabeledSnapshot(
        snapshot=snapshot,
        ticker_total_return=float(labels["ticker_total_return"]),
        benchmark_total_return=float(labels["benchmark_total_return"]),
        forward_21d_excess_return=float(labels["forward_21d_excess_return"]),
        direction_positive=1 if labels["excess_return_direction"] == "POSITIVE" else 0,
        outcome_materialized_at=_iso(observed),
    )


def materialize_outcome(
    *, snapshot: FeatureSnapshot, market: MarketDataBundle,
    outcome_materialized_at: datetime | str, run_id: str | None = None,
) -> LabeledSnapshot:
    started_at = time.perf_counter()
    identity = validate_run_id(run_id) if run_id is not None else None
    lifecycle_context = {
        "ticker": snapshot.ticker, "as_of": snapshot.as_of, "target": snapshot.target_date,
        "horizon_trading_days": snapshot.horizon_trading_days,
        "analysis_mode": snapshot.analysis_mode, "dataset_version": snapshot.dataset_version,
        "feature_version": snapshot.feature_version,
    }
    try:
        result = _materialize_outcome(snapshot=snapshot, market=market,
                                      outcome_materialized_at=outcome_materialized_at)
    except Exception:
        quant_lifecycle_event(context=lifecycle_context, boundary="outcome_materialization",
            started_at=started_at, outcome="fail_closed", run_id=identity,
            evidence_count=len(snapshot.evidence_ids), error_code="QUANT_OUTCOME_REJECTED")
        raise
    quant_lifecycle_event(context=lifecycle_context, boundary="outcome_materialization",
        started_at=started_at, outcome="success", run_id=identity,
        evidence_count=len(snapshot.evidence_ids), counts={"result_count": 1})
    return result


def labeled_snapshot_from_mapping(payload: Mapping[str, Any]) -> LabeledSnapshot:
    """Load an already materialized immutable local fixture row."""
    raw_snapshot = payload.get("snapshot")
    if not isinstance(raw_snapshot, Mapping):
        raise TypeError("snapshot fixture row requires snapshot object")
    values = raw_snapshot.get("values")
    indicators = raw_snapshot.get("missing_indicators", {})
    if not isinstance(values, Mapping) or not isinstance(indicators, Mapping):
        raise TypeError("snapshot values and missing_indicators must be mappings")
    canonical = {key: raw_snapshot[key] for key in (
        "ticker", "as_of", "target_date", "horizon_trading_days", "benchmark",
        "analysis_mode", "dataset_version", "feature_version", "values",
        "evidence_ids", "missing_indicators", "degraded", "finalized_at",
    )}
    expected_hash = _json_hash(canonical)
    claimed_hash = str(raw_snapshot.get("snapshot_hash") or expected_hash)
    if claimed_hash != expected_hash:
        raise ValueError("snapshot fixture hash mismatch")
    snapshot = FeatureSnapshot(
        ticker=str(raw_snapshot["ticker"]), as_of=str(raw_snapshot["as_of"]),
        target_date=str(raw_snapshot["target_date"]),
        horizon_trading_days=int(raw_snapshot["horizon_trading_days"]),
        benchmark=str(raw_snapshot["benchmark"]), analysis_mode=str(raw_snapshot["analysis_mode"]),
        dataset_version=str(raw_snapshot["dataset_version"]), feature_version=str(raw_snapshot["feature_version"]),
        values=MappingProxyType({str(k): (_number_or_none(v)) for k, v in values.items()}),
        evidence_ids=tuple(str(item) for item in raw_snapshot.get("evidence_ids", [])),
        missing_indicators=MappingProxyType({str(k): bool(v) for k, v in indicators.items()}),
        degraded=bool(raw_snapshot.get("degraded")), finalized_at=str(raw_snapshot["finalized_at"]),
        snapshot_hash=claimed_hash,
        schema_version=str(raw_snapshot.get("schema_version") or "forecast-v1"),
        calendar_version=str(raw_snapshot.get("calendar_version") or CALENDAR_VERSION),
        feature_materialized_at=str(
            raw_snapshot.get("feature_materialized_at") or raw_snapshot["finalized_at"]
        ),
    )
    outcome_time = str(payload["outcome_materialized_at"])
    as_of_day = parse_utc_datetime(snapshot.as_of, field_name="as_of").date()
    expected_target = nth_trading_session_after(as_of_day, HORIZON_TRADING_DAYS)
    try:
        target_day = date.fromisoformat(snapshot.target_date)
    except ValueError as exc:
        raise ValueError("target_date must be ISO-8601") from exc
    if snapshot.horizon_trading_days != HORIZON_TRADING_DAYS or target_day != expected_target:
        raise TemporalLeakageError("fixture target must be the exact 21st regular session after as_of")
    observed = parse_utc_datetime(outcome_time, field_name="outcome_materialized_at")
    if observed < regular_session_close_utc(target_day):
        raise TemporalLeakageError("fixture outcome precedes target availability")
    if parse_utc_datetime(snapshot.finalized_at) >= observed:
        raise TemporalLeakageError("fixture outcome precedes forecast snapshot")
    row = LabeledSnapshot(
        snapshot=snapshot,
        ticker_total_return=float(payload["ticker_total_return"]),
        benchmark_total_return=float(payload["benchmark_total_return"]),
        forward_21d_excess_return=float(payload["forward_21d_excess_return"]),
        direction_positive=payload["direction_positive"],
        outcome_materialized_at=outcome_time,
    )
    _require_canonical_label_identity(row)
    return row


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("feature values must be finite or null")
    return result


def eligible_row_content_hash(rows: Iterable[LabeledSnapshot]) -> str:
    """Bind snapshot identity, complete feature payload, labels and outcome time."""
    ordered = sorted(rows, key=lambda row: (row.snapshot.as_of, row.snapshot.ticker))
    return _json_hash([{
        "snapshot": row.snapshot.canonical_payload(),
        "snapshot_hash": row.snapshot.snapshot_hash,
        "ticker_total_return": row.ticker_total_return,
        "benchmark_total_return": row.benchmark_total_return,
        "forward_21d_excess_return": row.forward_21d_excess_return,
        "direction_positive": row.direction_positive,
        "outcome_materialized_at": row.outcome_materialized_at,
    } for row in ordered])


def _membership_reason(row: LabeledSnapshot, constituents: list[dict[str, Any]]) -> str | None:
    cutoff = parse_utc_datetime(row.snapshot.as_of, field_name="as_of").date()
    for item in constituents:
        start = date.fromisoformat(item["effective_from"])
        end = date.fromisoformat(item["effective_to"]) if item["effective_to"] else None
        if (item["ticker"] == row.snapshot.ticker and item["benchmark"] == row.snapshot.benchmark
                and start <= cutoff and (end is None or cutoff <= end)):
            return None
    return "outside_universe_effective_membership"


def _temporal_row_reason(row: LabeledSnapshot) -> str | None:
    try:
        as_of = parse_utc_datetime(row.snapshot.as_of, field_name="as_of")
        target = date.fromisoformat(row.snapshot.target_date)
        finalized = parse_utc_datetime(row.snapshot.finalized_at, field_name="finalized_at")
        observed = parse_utc_datetime(
            row.outcome_materialized_at, field_name="outcome_materialized_at"
        )
    except (TypeError, ValueError):
        return "invalid_temporal_identity"
    if target != nth_trading_session_after(as_of.date(), HORIZON_TRADING_DAYS):
        return "invalid_target_date"
    if finalized > as_of:
        return "snapshot_finalized_after_as_of"
    if observed < regular_session_close_utc(target):
        return "outcome_before_target_availability"
    if finalized >= observed:
        return "snapshot_not_before_outcome"
    return None


def _feature_family_audit(rows: tuple[LabeledSnapshot, ...]) -> dict[str, dict[str, Any]]:
    """Measure actual observed and varying specialist values, never missing flags alone."""
    audit: dict[str, dict[str, Any]] = {}
    for family, names in FULL_FEATURE_FAMILIES.items():
        slot_missing_name = f"{family}_features__missing"
        covered_rows = [
            row for row in rows
            if not row.snapshot.missing_indicators.get(slot_missing_name, True)
            and any(row.snapshot.values.get(name) is not None for name in names)
        ]
        observed_names = sorted({
            name for name in names
            if any(row.snapshot.values.get(name) is not None for row in covered_rows)
        })
        varying_names = []
        for name in observed_names:
            values = {
                float(row.snapshot.values[name]) for row in covered_rows
                if row.snapshot.values.get(name) is not None
            }
            if len(values) > 1:
                varying_names.append(name)
        audit[family] = {
            "coverage": len(covered_rows) / len(rows) if rows else 0.0,
            "covered_row_count": len(covered_rows),
            "observed_feature_count": len(observed_names),
            "varying_feature_count": len(varying_names),
            "observed_features": observed_names,
            "varying_features": varying_names,
        }
    return audit


def _build_data_readiness(
    *, rows: Iterable[LabeledSnapshot], universe_manifest: Mapping[str, Any],
    run_id: str, source_hashes: Mapping[str, str] | None = None,
    cutoff_policy_version: str = CUTOFF_POLICY_VERSION,
    require_full_feature_coverage: bool = False,
    minimum_tickers: int = FULL_FEATURE_MINIMUM_TICKERS,
    minimum_cutoffs_per_ticker: int = FULL_FEATURE_MINIMUM_CUTOFFS_PER_TICKER,
    minimum_family_coverage: float = FULL_FEATURE_MINIMUM_FAMILY_COVERAGE,
    minimum_graph_coverage: float = FULL_FEATURE_MINIMUM_GRAPH_COVERAGE,
) -> DataReadiness:
    """Produce the mandatory coverage manifest. This function performs no fitting."""
    universe = validate_universe_manifest(universe_manifest)
    if cutoff_policy_version not in SUPPORTED_CUTOFF_POLICY_VERSIONS:
        raise ValueError("unsupported cutoff_policy_version")
    materialized = tuple(rows)
    safe_source_hashes: dict[str, str] = {}
    for key, value in (source_hashes or {}).items():
        safe_key = validate_evidence_references((key,))[0]
        safe_value = validate_evidence_references((value,))[0]
        safe_source_hashes[safe_key] = safe_value
    reasons: dict[str, int] = {}
    eligible: list[LabeledSnapshot] = []
    for row in materialized:
        reason = _membership_reason(row, universe["constituents"])
        if reason is not None:
            pass
        elif row.snapshot.horizon_trading_days != HORIZON_TRADING_DAYS:
            reason = "invalid_horizon"
        elif row.snapshot.dataset_version != MARKET_DATA_VERSION:
            reason = "dataset_version_mismatch"
        elif row.snapshot.feature_version != MARKET_FEATURE_VERSION:
            reason = "feature_version_mismatch"
        elif _temporal_row_reason(row) is not None:
            reason = _temporal_row_reason(row)
        elif _canonical_label_identity_reason(row) is not None:
            reason = _canonical_label_identity_reason(row)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            eligible.append(row)

    feature_names = sorted({name for row in eligible for name in row.snapshot.values})
    missingness = {
        name: (sum(row.snapshot.values.get(name) is None for row in eligible) / len(eligible) if eligible else 1.0)
        for name in feature_names
    }
    ticker_count = len({row.snapshot.ticker for row in eligible})
    cutoff_count = len({row.snapshot.as_of for row in eligible})
    evidence_coverage = sum(bool(row.snapshot.evidence_ids) for row in eligible) / len(eligible) if eligible else 0.0
    filing_coverage = sum(not row.snapshot.missing_indicators.get("financial_features__missing", True) for row in eligible) / len(eligible) if eligible else 0.0
    graph_coverage = sum(row.snapshot.values.get("research_graph_coverage") == 1.0 for row in eligible) / len(eligible) if eligible else 0.0
    family_audit = _feature_family_audit(tuple(eligible))
    per_ticker_cutoffs = {
        ticker: len({row.snapshot.as_of for row in eligible if row.snapshot.ticker == ticker})
        for ticker in sorted({row.snapshot.ticker for row in eligible})
    }
    dates = [row.snapshot.as_of for row in eligible]
    source_coverage = {
        "market": {"date_start": min(dates) if dates else None, "date_end": max(dates) if dates else None,
                   "row_count": len(eligible), "temporal_status": "VERIFIED_SNAPSHOT_RANGE" if eligible else "MISSING"},
        "research_news": {"date_start": None, "date_end": None, "coverage": evidence_coverage,
                          "temporal_status": "UNVERIFIED_REFERENCE_ONLY" if evidence_coverage else "MISSING"},
        "sec_filings": {"date_start": None, "date_end": None, "coverage": filing_coverage,
                        "temporal_status": "UNVERIFIED_REFERENCE_ONLY" if filing_coverage else "MISSING"},
        "temporal_graph": {"date_start": None, "date_end": None, "coverage": graph_coverage,
                           "temporal_status": "UNVERIFIED_REFERENCE_ONLY" if graph_coverage else "MISSING"},
    }
    blocking: list[str] = []
    if not eligible:
        blocking.append("no_eligible_labeled_rows")
    if ticker_count == 0 or cutoff_count == 0:
        blocking.append("empty_universe_or_cutoffs")
    if reasons.get("outside_universe_effective_membership", 0):
        blocking.append("outside_universe_effective_membership")
    temporal_blockers = {
        "invalid_temporal_identity", "invalid_target_date",
        "snapshot_finalized_after_as_of", "outcome_before_target_availability",
        "snapshot_not_before_outcome",
    }
    if temporal_blockers.intersection(reasons):
        blocking.append("invalid_temporal_row")
    label_blockers = {
        "missing_price_label", "inconsistent_excess_return",
        "inconsistent_direction_label",
    }
    if label_blockers.intersection(reasons):
        blocking.append("invalid_price_label_identity")
    if require_full_feature_coverage:
        if ticker_count < minimum_tickers:
            blocking.append("full_feature_insufficient_tickers")
        if any(count < minimum_cutoffs_per_ticker for count in per_ticker_cutoffs.values()) or not per_ticker_cutoffs:
            blocking.append("full_feature_insufficient_cutoffs_per_ticker")
        for family, audit in family_audit.items():
            if float(audit["coverage"]) < minimum_family_coverage:
                blocking.append(f"full_feature_{family}_coverage_below_threshold")
            if int(audit["observed_feature_count"]) == 0:
                blocking.append(f"full_feature_{family}_unobserved")
            if int(audit["varying_feature_count"]) == 0:
                blocking.append(f"full_feature_{family}_has_no_variation")
        if graph_coverage < minimum_graph_coverage:
            blocking.append("full_feature_graph_coverage_below_threshold")
    blocking = sorted(set(blocking))
    status = "PASS" if not blocking else "BLOCKED"
    manifest: dict[str, Any] = {
        "schema_version": "1.0", "status": status, "run_id": run_id,
        "universe_version": universe["universe_version"],
        "universe_manifest_hash": universe["manifest_hash"],
        "dataset_version": MARKET_DATA_VERSION, "feature_version": MARKET_FEATURE_VERSION,
        "cutoff_policy_version": cutoff_policy_version,
        "source_coverage": source_coverage, "ticker_count": ticker_count,
        "monthly_cutoff_count": cutoff_count, "eligible_forecast_count": len(eligible),
        "feature_missingness": missingness, "news_coverage": evidence_coverage,
        "filing_coverage": filing_coverage, "graph_coverage": graph_coverage,
        "excluded_rows_by_reason": dict(sorted(reasons.items())),
        "content_hashes": {**dict(sorted(safe_source_hashes.items())),
                           "eligible_rows_sha256": eligible_row_content_hash(eligible)},
        "dataset_builder_version": DATASET_BUILDER_VERSION,
        "label_contract": {
            "version": LABEL_CONTRACT_VERSION,
            "excess_return_formula": "ticker_total_return-minus-benchmark_total_return",
            "direction_positive_rule": "1-iff-canonical-excess-return-greater-than-0",
            "excess_return_abs_tolerance": LABEL_RETURN_ABS_TOLERANCE,
            "excess_return_rel_tolerance": 0.0,
        },
        "scientific_report_level": "SCIENTIFIC" if ticker_count >= 10 and cutoff_count >= 60 and len(eligible) >= 500 else "PRELIMINARY",
        "blocking_issues": blocking,
    }
    if require_full_feature_coverage:
        manifest.update({
            "feature_family_coverage": family_audit,
            "per_ticker_cutoff_count": per_ticker_cutoffs,
            "full_feature_contract": {
                "required": True,
                "status": "PASS" if not blocking else "BLOCKED",
                "minimum_tickers": minimum_tickers,
                "minimum_cutoffs_per_ticker": minimum_cutoffs_per_ticker,
                "minimum_family_coverage": minimum_family_coverage,
                "minimum_graph_coverage": minimum_graph_coverage,
            },
        })
    manifest_hash = _json_hash(manifest)
    manifest["manifest_hash"] = manifest_hash
    readiness = DataReadiness(MappingProxyType(manifest), manifest_hash)
    readiness.require_pass()
    return readiness


def build_data_readiness(
    *, rows: Iterable[LabeledSnapshot], universe_manifest: Mapping[str, Any],
    run_id: str, source_hashes: Mapping[str, str] | None = None,
    cutoff_policy_version: str = CUTOFF_POLICY_VERSION,
    require_full_feature_coverage: bool = False,
    minimum_tickers: int = FULL_FEATURE_MINIMUM_TICKERS,
    minimum_cutoffs_per_ticker: int = FULL_FEATURE_MINIMUM_CUTOFFS_PER_TICKER,
    minimum_family_coverage: float = FULL_FEATURE_MINIMUM_FAMILY_COVERAGE,
    minimum_graph_coverage: float = FULL_FEATURE_MINIMUM_GRAPH_COVERAGE,
) -> DataReadiness:
    started_at = time.perf_counter()
    materialized = tuple(rows)
    try:
        identity = validate_run_id(run_id)
    except Exception:
        quant_lifecycle_event(context={}, boundary="data_readiness", started_at=started_at,
            outcome="fail_closed", error_code="QUANT_READINESS_REJECTED")
        raise
    lifecycle_context = ({
        "ticker": materialized[0].snapshot.ticker,
        "as_of": materialized[-1].snapshot.as_of,
        "target": materialized[-1].snapshot.target_date,
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "analysis_mode": materialized[0].snapshot.analysis_mode,
        "dataset_version": MARKET_DATA_VERSION, "feature_version": MARKET_FEATURE_VERSION,
        "schema_version": materialized[0].snapshot.schema_version,
        "calendar_version": materialized[0].snapshot.calendar_version,
    } if materialized else {"dataset_version": MARKET_DATA_VERSION,
                            "feature_version": MARKET_FEATURE_VERSION})
    try:
        result = _build_data_readiness(rows=materialized, universe_manifest=universe_manifest,
            run_id=identity, source_hashes=source_hashes,
            cutoff_policy_version=cutoff_policy_version,
            require_full_feature_coverage=require_full_feature_coverage,
            minimum_tickers=minimum_tickers,
            minimum_cutoffs_per_ticker=minimum_cutoffs_per_ticker,
            minimum_family_coverage=minimum_family_coverage,
            minimum_graph_coverage=minimum_graph_coverage)
    except Exception:
        quant_lifecycle_event(context=lifecycle_context, boundary="data_readiness",
            started_at=started_at, outcome="fail_closed", run_id=identity,
            counts={"input_count": len(materialized)}, error_code="QUANT_READINESS_REJECTED")
        raise
    unverified = any(item.get("temporal_status") == "UNVERIFIED_REFERENCE_ONLY"
                     for item in result.manifest["source_coverage"].values())
    quant_lifecycle_event(context=lifecycle_context, boundary="data_readiness",
        started_at=started_at, outcome="degraded" if unverified else "success", run_id=identity,
        evidence_count=sum(len(row.snapshot.evidence_ids) for row in materialized),
        counts={"input_count": len(materialized),
                "eligible_count": result.manifest["eligible_forecast_count"],
                "excluded_count": sum(result.manifest["excluded_rows_by_reason"].values()),
                "ticker_count": result.manifest["ticker_count"],
                "cutoff_count": result.manifest["monthly_cutoff_count"]},
        degraded_flags=["source_coverage_unverified"] if unverified else [])
    return result


__all__ = [
    "CUTOFF_POLICY_VERSION", "MONTH_END_CUTOFF_POLICY_VERSION",
    "MONTH_TO_DATE_CUTOFF_POLICY_VERSION", "SUPPORTED_CUTOFF_POLICY_VERSIONS",
    "DATASET_BUILDER_VERSION", "DataReadiness",
    "LABEL_CONTRACT_VERSION", "LABEL_RETURN_ABS_TOLERANCE", "FeatureSnapshot",
    "LabeledSnapshot", "build_data_readiness", "eligible_row_content_hash",
    "build_feature_snapshot", "labeled_snapshot_from_mapping", "materialize_outcome",
    "validate_universe_manifest",
]
