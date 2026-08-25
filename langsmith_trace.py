import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langsmith import Client
from evaluation.valid_availability import valid_metric_inventory






load_dotenv()

api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
if not api_key:
    raise RuntimeError(
        "LANGSMITH_API_KEY or LANGCHAIN_API_KEY was not found in the environment."
    )

client = Client()

DEFAULT_PROJECT = os.getenv("LANGCHAIN_PROJECT", "alpha-agent-node")


METRIC_MARKERS = {
    "_period",
    "_inputs",
    "formula",
    "interpretation",
    "confidence",
    "validation_status",
    "exact_period",
    "fallback_reason",
    "error",
}

UNAVAILABLE_LABELS = {
    "unavailable",
    "insufficient_data",
    "insufficient_history",
    "not_available",
    "missing",
    "failed",
    "error",
}


NON_METRIC_KEYS = {
    "ticker",
    "company_name",
    "sector",
    "industry",
    "current_price",
    "market_cap",
    "beta",
    "52w_high",
    "52w_low",
    "forward_pe",
    "peg_ratio",
    "annual_revenue_history",
    "sec_filing",
    "capital_allocation",
    "financial_sanity",
    "sanity_check",
}






def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def run_latency(run: Any) -> float:
    latency = getattr(run, "latency", None)

    if latency is not None:
        try:
            return round(float(latency), 3)
        except Exception:
            pass

    start = getattr(run, "start_time", None)
    end = getattr(run, "end_time", None)

    if start is not None and end is not None:
        try:
            return round((end - start).total_seconds(), 3)
        except Exception:
            pass

    return 0.0


def normalize_name(name: Any) -> str:
    return str(name or "").strip()


def first_child(
    children: List[Any],
    *,
    name: Optional[str] = None,
    names: Optional[set] = None,
) -> Optional[Any]:
    for child in children:
        child_name = normalize_name(getattr(child, "name", ""))

        if name is not None and child_name == name:
            return child

        if names is not None and child_name in names:
            return child

    return None


def all_children_named(children: List[Any], name: str) -> List[Any]:
    return [
        c for c in children
        if normalize_name(getattr(c, "name", "")) == name
    ]






def recursive_find_key(data: Any, key: str) -> Any:
    if isinstance(data, dict):
        if key in data:
            return data[key]

        for value in data.values():
            found = recursive_find_key(value, key)
            if found is not None:
                return found

    elif isinstance(data, list):
        for item in data:
            found = recursive_find_key(item, key)
            if found is not None:
                return found

    return None


def recursive_find_dict_with_keys(
    data: Any,
    required_keys: set,
) -> Optional[Dict[str, Any]]:
    if isinstance(data, dict):
        if required_keys.issubset(data.keys()):
            return data

        for value in data.values():
            found = recursive_find_dict_with_keys(value, required_keys)
            if found is not None:
                return found

    elif isinstance(data, list):
        for item in data:
            found = recursive_find_dict_with_keys(item, required_keys)
            if found is not None:
                return found

    return None






def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None


    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    candidates = []

    if match:
        candidates.append(match.group(1))


    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def extract_brain_decision(
    children: List[Any],
    fallback_query_type: str,
) -> Dict[str, Any]:

    brain_runs = all_children_named(children, "financial.brain")


    for brain in reversed(brain_runs):
        outputs = getattr(brain, "outputs", None) or {}


        intents = recursive_find_key(outputs, "intents")

        if isinstance(intents, list) and intents:
            return {
                "intents": intents,
                "extra_tools": recursive_find_key(outputs, "extra_tools") or [],
                "rationale": recursive_find_key(outputs, "rationale") or "",
            }


        serialized = json.dumps(outputs, ensure_ascii=False, default=str)
        parsed = extract_json_from_text(serialized)

        if parsed and isinstance(parsed.get("intents"), list):
            return {
                "intents": parsed.get("intents", []),
                "extra_tools": parsed.get("extra_tools", []),
                "rationale": parsed.get("rationale", ""),
            }


        inputs = getattr(brain, "inputs", None) or {}
        messages = recursive_find_key(inputs, "messages")

        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict):
                    continue

                content = str(message.get("content") or "")
                parsed = extract_json_from_text(content)

                if parsed and isinstance(parsed.get("intents"), list):
                    return {
                        "intents": parsed.get("intents", []),
                        "extra_tools": parsed.get("extra_tools", []),
                        "rationale": parsed.get("rationale", ""),
                    }

    return {
        "intents": [fallback_query_type],
        "extra_tools": [],
        "rationale": "",
    }






def extract_checker(children: List[Any]) -> Dict[str, Any]:
    checker_runs = all_children_named(children, "financial.checker")

    if not checker_runs:
        return {
            "telemetry_available": False,
            "is_complete": None,
            "score": None,
            "required_any_of_groups": None,
            "availability": None,
            "validity": None,
            "missing_groups": None,
            "issues": None,
            "feedback": None,
        }


    checker = checker_runs[-1]

    outputs = getattr(checker, "outputs", None) or {}

    is_complete = recursive_find_key(outputs, "is_complete")
    score = recursive_find_key(outputs, "score")

    intent_requirements = recursive_find_key(
        outputs,
        "intent_requirements",
    )

    if not isinstance(intent_requirements, dict):
        intent_requirements = {}

    required_groups = (
        intent_requirements.get("required_any_of_groups")
        or recursive_find_key(outputs, "required_any_of_groups")
    )

    availability = (
        intent_requirements.get("availability")
        or recursive_find_key(outputs, "availability")
    )

    validity = (
        intent_requirements.get("validity")
        or recursive_find_key(outputs, "validity")
    )

    missing_groups = (
        intent_requirements.get("missing_groups")
        if "missing_groups" in intent_requirements
        else recursive_find_key(outputs, "missing_groups")
    )

    issues = recursive_find_key(outputs, "issues")
    feedback = recursive_find_key(outputs, "feedback")

    return {
        "telemetry_available": True,
        "is_complete": (
            bool(is_complete)
            if isinstance(is_complete, bool)
            else is_complete
        ),
        "score": safe_float(score),
        "required_any_of_groups": (
            required_groups
            if isinstance(required_groups, list)
            else []
        ),
        "availability": (
            availability
            if isinstance(availability, dict)
            else {}
        ),
        "validity": validity if isinstance(validity, dict) else {},
        "missing_groups": (
            missing_groups
            if isinstance(missing_groups, list)
            else []
        ),
        "issues": (
            issues
            if isinstance(issues, list)
            else []
        ),
        "feedback": str(feedback or ""),
        "latency_seconds": run_latency(checker),
    }






def looks_like_mcp_tool_name(name: str) -> bool:
    name = normalize_name(name)

    if not name:
        return False

    if name.startswith("tool_"):
        return True

    return False


def extract_actual_mcp_calls(children: List[Any]) -> Dict[str, Any]:
    calls = []
    failures = []

    for child in children:
        name = normalize_name(getattr(child, "name", ""))
        run_type = normalize_name(getattr(child, "run_type", ""))

        if run_type != "tool":
            continue

        if not looks_like_mcp_tool_name(name):
            continue

        call = {
            "tool": name,
            "status": getattr(child, "status", None),
            "latency_seconds": run_latency(child),
        }

        error = getattr(child, "error", None)

        if error:
            call["error"] = str(error)
            failures.append(
                {
                    "tool": name,
                    "error": str(error),
                }
            )

        calls.append(call)

    return {

        "telemetry_available": bool(calls),
        "actual_tool_calls": calls,
        "tool_call_count": len(calls) if calls else None,
        "failed_tool_calls": failures if calls else None,
    }






def extract_resolved_tools(children: List[Any]) -> Dict[str, Any]:

    extraction_tools = None
    compute_tools = None
    skipped_tools = None

    for child in children:
        name = normalize_name(getattr(child, "name", ""))

        if name not in {
            "financial.extract",
            "financial.compute",
            "FinancialAnalystAgent.run",
        }:
            continue

        for source in (
            getattr(child, "inputs", None) or {},
            getattr(child, "outputs", None) or {},
        ):
            if extraction_tools is None:
                found = recursive_find_key(
                    source,
                    "resolved_extraction_tools",
                )
                if isinstance(found, list):
                    extraction_tools = found

            if compute_tools is None:
                found = recursive_find_key(
                    source,
                    "resolved_compute_tools",
                )
                if isinstance(found, list):
                    compute_tools = found

            if skipped_tools is None:
                found = recursive_find_key(
                    source,
                    "skipped_tools",
                )
                if isinstance(found, list):
                    skipped_tools = found

    return {
        "resolved_extraction_tools": extraction_tools,
        "resolved_compute_tools": compute_tools,
        "skipped_tools": skipped_tools,
        "telemetry_available": (
            extraction_tools is not None
            or compute_tools is not None
            or skipped_tools is not None
        ),
    }






def is_financial_metric_object(key: str, value: Any) -> bool:
    if key in NON_METRIC_KEYS:
        return False

    if not isinstance(value, dict):
        return False


    return bool(METRIC_MARKERS.intersection(value.keys()))


def scan_numeric_values(data: Any, result: Dict[str, int]) -> None:
    if isinstance(data, dict):
        for value in data.values():
            scan_numeric_values(value, result)

    elif isinstance(data, list):
        for item in data:
            scan_numeric_values(item, result)

    elif isinstance(data, float):
        if math.isnan(data):
            result["nan_count"] += 1
        elif math.isinf(data):
            result["inf_count"] += 1


def metric_is_unavailable(metric: Dict[str, Any]) -> bool:
    if metric.get("error"):
        return True

    for key in ("interpretation", "status", "validation_status"):
        value = metric.get(key)

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in UNAVAILABLE_LABELS:
                return True


    candidate_values = [
        v
        for k, v in metric.items()
        if not k.startswith("_")
        and k not in {
            "interpretation",
            "formula",
            "confidence",
            "note",
            "error",
            "status",
            "validation_status",
        }
    ]

    if candidate_values and all(v is None for v in candidate_values):
        return True

    return False


def calculate_validation_metrics(
    financial_summary: Dict[str, Any],
) -> Dict[str, Any]:

    result = {
        "nan_count": 0,
        "inf_count": 0,
        "metric_object_count": 0,
        "metrics_with_period": 0,
        "metrics_without_period": 0,
        "metrics_with_inputs": 0,
        "metrics_without_inputs": 0,
        "metrics_unavailable": 0,
        "metrics_valid_primary": 0,
        "metrics_fallback": 0,
        "metrics_degraded": 0,
        "metrics_with_recovery_metadata": 0,
        "period_coverage_pct": None,
        "input_provenance_coverage_pct": None,
        "recovery_metadata_coverage_pct": None,
    }

    scan_numeric_values(financial_summary, result)

    for key, value in financial_summary.items():
        if not is_financial_metric_object(key, value):
            continue

        result["metric_object_count"] += 1

        period = value.get("_period")

        if period not in (None, "", []):
            result["metrics_with_period"] += 1
        else:
            result["metrics_without_period"] += 1

        inputs = value.get("_inputs")

        if isinstance(inputs, dict) and inputs:
            result["metrics_with_inputs"] += 1
        else:
            result["metrics_without_inputs"] += 1

        if metric_is_unavailable(value):
            result["metrics_unavailable"] += 1

        status = str(value.get("validation_status") or "").upper()
        if status == "VALID_PRIMARY":
            result["metrics_valid_primary"] += 1
        elif status == "FALLBACK":
            result["metrics_fallback"] += 1
        elif status == "DEGRADED":
            result["metrics_degraded"] += 1

        required_metadata = {
            "source", "exact_period", "formula", "fallback_reason",
            "confidence", "is_approximation", "validation_status",
        }
        if required_metadata.issubset(value):
            result["metrics_with_recovery_metadata"] += 1

    total = result["metric_object_count"]

    if total:
        result["period_coverage_pct"] = round(
            result["metrics_with_period"] / total * 100,
            2,
        )

        result["input_provenance_coverage_pct"] = round(
            result["metrics_with_inputs"] / total * 100,
            2,
        )

        result["recovery_metadata_coverage_pct"] = round(
            result["metrics_with_recovery_metadata"] / total * 100,
            2,
        )

    return result






def extract_sec_execution(
    financial_summary: Dict[str, Any],
) -> Dict[str, Any]:

    sec = financial_summary.get("sec_filing")

    if not isinstance(sec, dict) or not sec:
        return {
            "requested_form_type": None,
            "cik": None,
            "accession_number": None,
            "status": None,
            "filing_text_word_count": None,
        }

    selected = sec.get("selected_filing")
    if not isinstance(selected, dict):
        selected = {}

    filing_text = sec.get("filing_text")
    if not isinstance(filing_text, dict):
        filing_text = {}

    return {
        "requested_form_type": (
            sec.get("requested_form_type")
            or sec.get("form_type")
        ),
        "cik": sec.get("cik"),
        "accession_number": (
            selected.get("accession_number")
            or sec.get("accession_number")
            or filing_text.get("accession_number")
        ),
        "status": sec.get("status"),
        "filing_text_word_count": filing_text.get("word_count"),
    }






def extract_publication_validation(
    outputs: Dict[str, Any],
) -> Dict[str, Any]:

    return {
        "publication_status": outputs.get("publication_status"),
        "confidence": outputs.get("confidence"),
        "withheld_sections": outputs.get("withheld_sections") or [],
        "section_reliability": outputs.get("section_reliability") or {},
        "publication_warnings": outputs.get("publication_warnings") or [],
        "recommendation_gate": outputs.get("recommendation_gate"),
        "final_publication_validation": (
            outputs.get("final_publication_validation")
        ),
    }


def extract_sentiment_validation(outputs: Dict[str, Any]) -> Dict[str, Any]:
    summary = outputs.get("sentiment_analysis_summary")
    if not isinstance(summary, dict):
        summary = {}

    try:
        if isinstance(summary.get("fear_greed_score"), bool):
            raise ValueError("boolean is not a sentiment score")
        score = float(summary.get("fear_greed_score"))
        valid_score = math.isfinite(score) and -1.0 <= score <= 1.0
    except (TypeError, ValueError):
        valid_score = False

    unique_documents = int(safe_float(summary.get("unique_document_count")) or 0)
    unique_domains = int(safe_float(summary.get("unique_domain_count")) or 0)
    extraction_errors = summary.get("extraction_errors") or []
    causal_mismatch = bool(summary.get("causal_mismatch_detected"))
    quality_passed = bool(
        summary.get("quality_passed") is True
        and summary.get("status") == "VALID"
        and valid_score
        and unique_documents >= 3
        and unique_domains >= 2
        and not extraction_errors
        and not causal_mismatch
    )

    confidence_components = summary.get("confidence_components")
    if not isinstance(confidence_components, dict):
        confidence_components = {
            "confidence": summary.get("fear_greed_confidence"),
            "direction_strength": summary.get("direction_strength"),
            "model_agreement_score": summary.get("model_agreement_score"),
            "evidence_independence": summary.get("evidence_independence"),
        }

    source_document_counts = summary.get("source_document_counts")
    if not isinstance(source_document_counts, dict):
        source_document_counts = {
            "raw_chunk_count": summary.get("raw_chunk_count"),
            "unique_documents": unique_documents,
            "unique_domains": unique_domains,
            "chunks_collapsed_to_documents": summary.get(
                "chunks_collapsed_to_documents"
            ),
            "duplicate_text_chunks_removed": summary.get(
                "duplicate_text_chunks_removed"
            ),
        }

    return {
        "execution_success": bool(summary.get("execution_success")),
        "pipeline_execution_success": bool(
            summary.get("pipeline_execution_success")
        ),
        "quality_passed": quality_passed,
        "quality_failures": summary.get("quality_failures") or [],
        "status": summary.get("status"),
        "polarity": summary.get("polarity"),
        "fear_greed_score": summary.get("fear_greed_score"),
        "fear_greed_label": summary.get("fear_greed_label"),
        "confidence_components": confidence_components,
        "source_document_counts": source_document_counts,
        "aspect_observations": summary.get("aspect_observations") or [],
        "causal_aspects_supported": summary.get("causal_aspects_supported") or [],
        "causal_aspects_used": summary.get("causal_aspects_used") or [],
        "causal_audit": summary.get("causal_audit") or {},
        "causal_mismatch_detected": causal_mismatch,
        "selected_attempt_number": summary.get("selected_attempt_number"),
        "loop_iterations_used": summary.get("loop_iterations_used"),
        "attempts": summary.get("attempts") or [],
        "extraction_errors": extraction_errors,
        "prior_attempt_errors": summary.get("prior_attempt_errors") or [],
        "attempt_extraction_errors": summary.get("attempt_extraction_errors") or [],
    }






def extract_agent_timings(outputs: Dict[str, Any]) -> Dict[str, Any]:
    history = outputs.get("agent_execution_history")

    if not isinstance(history, list):
        return {}

    result = {}

    for item in history:
        if not isinstance(item, dict):
            continue

        agent = item.get("agent_name")

        if not agent:
            continue

        result[agent] = {
            "outcome": item.get("outcome"),
            "execution_success": item.get("execution_success"),
            "quality_passed": item.get("quality_passed"),
            "duration_seconds": item.get("duration_s"),
            "error": item.get("error_message"),
        }

    return result






def infer_query_type(inputs: Dict[str, Any], task_query: str) -> str:
    explicit = (
        inputs.get("benchmark_query_type")
        or inputs.get("query_type")
    )

    if explicit:
        return str(explicit)

    normalized = task_query.lower()

    if "investment analysis report" in normalized:
        return "full_report"

    if "valuation" in normalized or "intrinsic value" in normalized:
        return "valuation"

    if "liquidity" in normalized:
        return "liquidity"

    if "financial health" in normalized:
        return "financial_health"

    if "growth" in normalized:
        return "growth"

    if "profitability" in normalized:
        return "profitability"

    if "cash flow" in normalized:
        return "cash_flow"

    return "overview"


def parse_trace(root_run: Any, project_name: str) -> Dict[str, Any]:
    inputs = root_run.inputs or {}
    outputs = root_run.outputs or {}

    task_query = str(
        inputs.get("task_query")
        or outputs.get("task_query")
        or ""
    )

    directives = (
        inputs.get("manager_directives")
        or outputs.get("manager_directives")
        or {}
    )

    query_type = infer_query_type(inputs, task_query)

    ticker = (
        inputs.get("ticker")
        or directives.get("ticker")
        or (outputs.get("manager_directives") or {}).get("ticker")
    )





    children = list(
        client.list_runs(
            project_name=project_name,
            trace_id=root_run.trace_id or root_run.id,
        )
    )

    children = [
        c for c in children
        if str(c.id) != str(root_run.id)
    ]





    financial_summary = outputs.get("financial_metrics_summary")

    if not isinstance(financial_summary, dict):
        financial_summary = {}

    sentiment_summary = outputs.get("sentiment_analysis_summary")
    if not isinstance(sentiment_summary, dict):
        sentiment_summary = {}

    brain = extract_brain_decision(children, query_type)
    checker = extract_checker(children)
    mcp_calls = extract_actual_mcp_calls(children)
    routing = extract_resolved_tools(children)


    checker_runs = all_children_named(children, "financial.checker")
    loop_iterations = len(checker_runs) if checker_runs else None





    return {
        "experiment": {
            "experiment_id": inputs.get(
                "experiment_id",
                os.getenv("EXPERIMENT_ID"),
            ),
            "system_version": inputs.get(
                "system_version",
                os.getenv("SYSTEM_VERSION", "proposed"),
            ),
            "query_type": query_type,
            "ticker": ticker,
            "task_query": task_query,
            "run_id": str(root_run.id),
            "trace_id": str(root_run.trace_id or root_run.id),
            "timestamp": str(root_run.start_time),
        },

        "input_configuration": {
            "manager_directives": directives,
            "user_preferences": inputs.get("user_preferences") or {},
        },

        "performance": {
            "status": root_run.status,
            "total_latency_seconds": run_latency(root_run),
            "agents": extract_agent_timings(outputs),
            "financial_loop_iterations": loop_iterations,
        },

        "financial_routing": {
            "intents": brain["intents"],
            "extra_tools": brain["extra_tools"],
            "brain_rationale": brain["rationale"],

            "resolved_extraction_tools": (
                routing["resolved_extraction_tools"]
            ),
            "resolved_compute_tools": (
                routing["resolved_compute_tools"]
            ),
            "skipped_tools": routing["skipped_tools"],
            "resolved_tool_telemetry_available": (
                routing["telemetry_available"]
            ),

            "actual_tool_calls": (
                mcp_calls["actual_tool_calls"]
            ),
            "actual_tool_call_count": (
                mcp_calls["tool_call_count"]
            ),
            "failed_tool_calls": (
                mcp_calls["failed_tool_calls"]
            ),
            "mcp_tool_telemetry_available": (
                mcp_calls["telemetry_available"]
            ),
        },

        "financial_checker": checker,

        "sec_execution": extract_sec_execution(financial_summary),

        "validation_metrics": {
            **calculate_validation_metrics(financial_summary),
            "valid_availability": valid_metric_inventory(financial_summary),
        },

        "publication_validation": extract_publication_validation(
            outputs
        ),

        "sentiment_validation": extract_sentiment_validation(outputs),

        "sentiment_analysis_summary": sentiment_summary,


        "financial_metrics_summary": financial_summary,


        "final_report": outputs.get("final_report"),

        "root_error": root_run.error,
    }






def main():
    candidate_projects = list(
        dict.fromkeys(
            [
                DEFAULT_PROJECT,
                "alpha-agent-node",
                "AgenticAlpha",
                "default",
            ]
        )
    )

    latest_data = None
    active_project = None

    for project in candidate_projects:
        print(f"Project scan: {project}")

        try:
            root_runs = list(
                client.list_runs(
                    project_name=project,
                    is_root=True,
                    limit=50,

                )
            )

            if not root_runs:
                continue

            root_run = max(
                root_runs,
                key=lambda run: getattr(run, "start_time", None) or datetime.min.replace(tzinfo=timezone.utc),
            )

            print(f"Trace found: {root_run.id}")

            latest_data = parse_trace(
                root_run=root_run,
                project_name=project,
            )

            active_project = project
            break

        except Exception as exc:
            print(
                f"⚠️ Error while reading project {project}: {exc}"
            )

    if latest_data is None:
        raise RuntimeError("No valid root trace was found.")

    ticker_part = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(latest_data["experiment"].get("ticker") or "unknown"),
    ).strip("_") or "unknown"

    query_type_part = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(latest_data["experiment"].get("query_type") or "overview"),
    ).strip("_") or "overview"

    timestamp_part = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_filename = (
        f"validation_{ticker_part}_{query_type_part}_{timestamp_part}.json"
    )

    with open(
        output_filename,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            latest_data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\nValidation export created")
    print(f"Project: {active_project}")
    print(
        "🧪 Experiment:",
        latest_data["experiment"]["experiment_id"],
    )
    print(
        "📌 Query type:",
        latest_data["experiment"]["query_type"],
    )
    print(
        "🧠 Intents:",
        latest_data["financial_routing"]["intents"],
    )

    tool_count = latest_data[
        "financial_routing"
    ]["actual_tool_call_count"]

    if tool_count is None:
        print(
            "⚠️ MCP tool calls were not traced independently in LangSmith."
        )
    else:
        print(f"MCP tool calls: {tool_count}")

    checker = latest_data["financial_checker"]

    print(
        "✅ Checker:",
        checker.get("is_complete"),
        "| Score:",
        checker.get("score"),
    )

    provenance = latest_data["validation_metrics"]

    print(
        "📊 Period coverage:",
        provenance.get("period_coverage_pct"),
    )
    print(
        "📊 Input provenance coverage:",
        provenance.get("input_provenance_coverage_pct"),
    )

    print(f"Output: {output_filename}")


if __name__ == "__main__":
    main()