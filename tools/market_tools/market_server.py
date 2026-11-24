"""Local deterministic Stage 07 service boundary; it never invokes an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time
from typing import Any, Mapping

from core.error_handler import VersionMismatchError
from core.forecast_contract import CALENDAR_VERSION, ForecastContext
from tools.market_tools import (QUANT_IMPLEMENTATION_IDENTITY, quant_lifecycle_event,
                                validate_evidence_references, validate_run_id)
from tools.market_tools.feature_builder import MarketFeatures, build_market_features
from tools.market_tools.forecast_model import DeterministicModel, predict
from tools.market_tools.historical_dataset_builder import DataReadiness
from tools.market_tools.market_data import MarketDataBundle
from tools.market_tools.specialist_fusion import (
    SPECIALIST_FUSION_VARIANT_TOKEN,
    apply_specialist_fusion,
)

MARKET_SERVER_VERSION = "market-service-v1"


@dataclass(frozen=True, slots=True)
class MarketForecastService:
    """Composable deterministic feature/prediction boundary for Stage 08."""
    model: DeterministicModel
    readiness: DataReadiness
    universe_manifest_hash: str
    universe_constituents: tuple[Mapping[str, Any], ...]
    model_variant: str = "full_model_v1"

    def __post_init__(self) -> None:
        self.readiness.require_pass()
        if self.readiness.manifest_hash != self.model.readiness_manifest_hash:
            raise ValueError("model was not fitted from this DATA_READINESS manifest")
        if self.universe_manifest_hash != self.readiness.manifest.get("universe_manifest_hash"):
            raise ValueError("universe manifest hash mismatch")
        if not self.universe_constituents:
            raise ValueError("universe constituents are required")
        if (self.model.dataset_version != self.readiness.manifest.get("dataset_version")
                or self.model.feature_version != self.readiness.manifest.get("feature_version")
                or self.model.training_rows != self.readiness.manifest.get("eligible_forecast_count")
                or self.model.eligible_rows_hash != self.readiness.manifest.get(
                    "content_hashes", {}
                ).get("eligible_rows_sha256")):
            raise ValueError("model identity does not match DATA_READINESS")

    def build_features(self, *, context: ForecastContext, market: MarketDataBundle,
                       research_features: Any = None, financial_features: Any = None,
                       sentiment_features: Any = None) -> MarketFeatures:
        retrieved = [market.ticker.retrieved_at, market.benchmark.retrieved_at]
        if market.sector is not None:
            retrieved.append(market.sector.retrieved_at)
        materialized_at = max(retrieved)
        return build_market_features(
            context=context, market=market, research_features=research_features,
            financial_features=financial_features, sentiment_features=sentiment_features,
            materialized_at=materialized_at,
        )

    def forecast(self, *, context: ForecastContext, market_features: MarketFeatures,
                 run_id: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        identity = validate_run_id(run_id)
        expected_as_of = context.as_of.isoformat().replace("+00:00", "Z")
        lifecycle_context = {
            **context.to_dict(), "model_version": self.model.model_version,
            "feature_version": self.model.feature_version,
        }
        try:
            cutoff = context.as_of.date()
            in_universe = any(
                str(item.get("ticker") or "").upper() == context.ticker
                and str(item.get("benchmark") or "").upper() == context.benchmark
                and date.fromisoformat(str(item.get("effective_from"))) <= cutoff
                and (
                    not item.get("effective_to")
                    or cutoff <= date.fromisoformat(str(item.get("effective_to")))
                )
                for item in self.universe_constituents
            )
            if not in_universe:
                raise VersionMismatchError(
                    "requested ticker is outside the model universe at ForecastContext.as_of"
                )
            if (market_features.ticker != context.ticker
                    or market_features.as_of != expected_as_of
                    or market_features.dataset_version != context.dataset_version
                    or market_features.feature_version != context.feature_version
                    or market_features.schema_version != context.schema_version
                    or market_features.calendar_version != CALENDAR_VERSION
                    or market_features.benchmark != context.benchmark
                    or market_features.analysis_mode != context.analysis_mode.value
                    or market_features.cutoff_policy != context.cutoff_policy.value):
                raise VersionMismatchError(
                    "market features do not match ForecastContext identity"
                )
            evidence_ids = validate_evidence_references(market_features.evidence_ids)
            output = predict(
                self.model, market_features, context=lifecycle_context, run_id=identity
            )
            result = output.to_dict()
            if SPECIALIST_FUSION_VARIANT_TOKEN in self.model_variant:
                result = apply_specialist_fusion(result, market_features)
            expected_cutoff_version = {
                "MONTH_END": "monthly_last_session_v1",
                "MONTH_TO_DATE": "monthly_to_date_v1",
            }[context.cutoff_policy.value]
            cutoff_policy_shift = (
                self.readiness.manifest.get("cutoff_policy_version")
                != expected_cutoff_version
            )
            result["degraded"] = bool(result.get("degraded") or cutoff_policy_shift)
            degraded_flags = []
            if market_features.degraded:
                degraded_flags.append("optional_feature_degradation")
            if cutoff_policy_shift:
                degraded_flags.append("cutoff_policy_distribution_shift")
            result.update({
                "ticker": context.ticker, "as_of": expected_as_of,
                "target_date": context.target_date.isoformat(),
                "horizon_trading_days": context.horizon_trading_days,
                "analysis_mode": context.analysis_mode.value,
                "cutoff_policy": context.cutoff_policy.value,
                "schema_version": context.schema_version,
                "calendar_version": CALENDAR_VERSION,
                "stage": QUANT_IMPLEMENTATION_IDENTITY["stage"],
                "stage_revision": QUANT_IMPLEMENTATION_IDENTITY["stage_revision"],
                "attempt": QUANT_IMPLEMENTATION_IDENTITY["attempt"],
                "agent": QUANT_IMPLEMENTATION_IDENTITY["agent"],
                "universe_manifest_hash": self.universe_manifest_hash,
                "readiness_manifest_hash": self.readiness.manifest_hash,
                "training_data_hash": self.model.training_data_hash,
                "eligible_rows_hash": self.model.eligible_rows_hash,
                "model_variant": self.model_variant, "benchmark": context.benchmark,
                "run_id": identity, "evidence_ids": list(evidence_ids),
                "evidence_count": len(evidence_ids),
                "degraded_flags": degraded_flags,
                "market_server_version": MARKET_SERVER_VERSION,
                "feature_materialized_at": market_features.materialized_at,
            })
        except Exception:
            quant_lifecycle_event(
                context=lifecycle_context, boundary="service_forecast",
                started_at=started_at, outcome="fail_closed", run_id=identity,
                error_code="QUANT_SERVICE_REJECTED",
            )
            raise
        event = quant_lifecycle_event(
            context=lifecycle_context, boundary="service_forecast",
            started_at=started_at,
            outcome="degraded" if result.get("degraded") else "success",
            run_id=identity, evidence_count=len(evidence_ids),
            counts={"result_count": 1},
            degraded_flags=degraded_flags,
        )
        result["latency_ms"] = event["latency_ms"]
        return result


def forecast_from_mapping(*, service: MarketForecastService,
                          context_payload: Mapping[str, Any],
                          market_features: MarketFeatures, run_id: str) -> dict[str, Any]:
    return service.forecast(context=ForecastContext.from_dict(dict(context_payload)),
                            market_features=market_features, run_id=run_id)


__all__ = ["MARKET_SERVER_VERSION", "MarketForecastService", "forecast_from_mapping"]
