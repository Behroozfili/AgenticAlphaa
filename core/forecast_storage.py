"""Content-addressed feature snapshots and bounded transient write retries."""
from __future__ import annotations
import asyncio
import hashlib
import json
import uuid
from copy import deepcopy
import httpx
from core.forecast_contract import parse_utc_datetime


async def write_with_retry(operation):
    """Retry only transient failures; the caller must supply an idempotent write."""
    for attempt in range(3):
        try:
            return await asyncio.to_thread(operation)
        except Exception as exc:
            code = str(getattr(exc, "code", ""))
            transient = isinstance(exc, (ConnectionError, TimeoutError, httpx.TransportError)) or code in {
                "40001", "40P01", "53300", "57P01", "08000", "08003", "08006", "502", "503", "504",
            }
            if not transient or attempt == 2:
                raise
            await asyncio.sleep(0.1 * (2 ** attempt))


def feature_snapshot(result, context, forecast):
    slot = result.get("market_features")
    if not isinstance(slot, dict) or not isinstance(slot.get("values"), dict) or not slot["values"]:
        raise ValueError("FEATURE_SNAPSHOT_UNAVAILABLE")
    values = deepcopy(slot["values"])
    values.pop("quant_prediction", None)  # computed output/latency, not an input feature
    for field in ("ticker", "dataset_version", "schema_version"):
        if values.get(field) != context.get(field):
            raise ValueError("FEATURE_SNAPSHOT_CONTEXT_MISMATCH")
    if parse_utc_datetime(values.get("as_of"), field_name="as_of") != parse_utc_datetime(context["as_of"], field_name="as_of"):
        raise ValueError("FEATURE_SNAPSHOT_CONTEXT_MISMATCH")
    if slot.get("version") != context.get("feature_version"):
        raise ValueError("FEATURE_SNAPSHOT_VERSION_MISMATCH")
    materialized = values.get("feature_materialized_at")
    if not materialized or materialized != forecast.get("feature_materialized_at"):
        raise ValueError("FEATURE_SNAPSHOT_TIME_MISMATCH")
    body = {
        "market_features": values,
        "missing": bool(slot.get("missing")), "degraded": bool(slot.get("degraded")),
        "forecast_context": deepcopy(context),
        "model_variant": forecast.get("model_variant"),
        "model_version": context.get("model_version"),
        "universe_manifest_hash": forecast.get("universe_manifest_hash"),
        "evidence_ids": list(slot.get("evidence_ids") or []),
    }
    # Keep exact numeric inputs and their materialization identity. Different
    # observations must not collide under the schema's original natural key.
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "snapshot_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "agenticalpha:features:" + digest)),
        "ticker": context["ticker"], "as_of": context["as_of"],
        "horizon_trading_days": context["horizon_trading_days"], "analysis_mode": context["analysis_mode"],
        "feature_set_variant": str(forecast["model_variant"]) + ":" + digest,
        "dataset_version": context["dataset_version"], "feature_version": context["feature_version"],
        "schema_version": context["schema_version"], "feature_values": body,
        "evidence_ids": list(slot.get("evidence_ids") or []), "materialized_at": materialized,
    }
