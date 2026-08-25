"""
agents/financial_agent.py

"""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import END, StateGraph
import os
from agents.prompts.promt_finance_checker import CHECKER_SYSTEM_PROMPT,system_prompt
from evaluation.financial_sanity import evaluate_financial_sanity
from evaluation.earnings_quality import normalize_earnings_quality
from evaluation.claim_contract import typed_financial_claim
from agents.executor_helpers.finance_executor_helper import ( extract_ticker, annotate_confidence, build_consistency_checks, build_data_quality_score, sanitize_nans)
from agents.state import FinancialAgentState, SharedManagerState
from langsmith import traceable
from core.observability import traced_call_tool
from core.error_handler import add_breadcrumb, capture_exception
from core.progress_bus import publish as _publish_progress, session_from_directives
from agents.executor_helpers.financial_helper import (INTENTS,  BASELINE_EXTRACTION,
    BASELINE_COMPUTE, ANALYSIS_PROFILES, FULL_REPORT_PROFILES,  MONTE_CARLO_TOOLS,
    EXTRACTION_REGISTRY, EXTRACTION_ORDER, COMPUTE_ORDER,
    COMPUTE_REGISTRY, SUMMARY_CALCULATED_PASSTHROUGH, checker_requirements,
    normalize_depth,derive_compute_facts, surface_calculated_ratios,select_sec_filing_type,
)


# Backward-compatible private aliases retained for the existing test and
# extension contract while implementation lives in financial_helper.
_FULL_REPORT_PROFILES = FULL_REPORT_PROFILES
_MONTE_CARLO_TOOLS = MONTE_CARLO_TOOLS
_checker_requirements = checker_requirements
_select_sec_filing_type = select_sec_filing_type




log = logging.getLogger("financial-analyst-agent")

_FACTS_YEARS_RETAINED = 3
_FILING_TEXT_CHAR_LIMIT = 20_000
_MCP_SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[os.path.join(os.path.dirname(__file__), "..","tools","financial_tools" ,"financial_server.py")],
    env=None,   
)


class FinancialAnalystAgent:
    
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_loops: int = 3,
        mcp_server_params: StdioServerParameters | None = None,
        llm_client: anthropic.Anthropic | None = None,
    ) -> None:
        
        self._llm    = llm_client or anthropic.Anthropic()
        self._model  = model
        self._default_max_loops = max_loops
        self._server_params = mcp_server_params or _MCP_SERVER_PARAMS

        
        self._session: ClientSession | None = None

       
        self._graph = self._build_graph()

        log.info(
            "FinancialAnalystAgent initialised — model=%s, server_script=%s",
            model,
            self._server_params.args[-1] if self._server_params.args else "<none>",
        )

   

    async def _execute_sec_filing_chain(
        self, session: ClientSession, state: FinancialAgentState,
        ticker: str, form_type: str, errors: list[str],
    ) -> None:
        """Run CIK -> filing list -> filing text using only returned identifiers."""
        raw = state["raw_numerical_data"]
        session_id = session_from_directives(state["manager_directives"])
        workflow: dict[str, Any] = {"form_type": form_type, "status": "started"}
        raw["sec_filing"] = workflow

        async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message=f"Financial Agent: calling tool '{tool}'...",
                detail={"tool": tool, "form_type": form_type},
            )
            try:
                add_breadcrumb(
                    "mcp.financial", f"Calling {tool}",
                    {"tool": tool, "ticker": ticker},
                )
                result = await traced_call_tool(session, tool, arguments)
                data = json.loads(result.content[0].text)
                if data.get("error"):
                    errors.append(f"{tool} error: {data['error']}")
                    return None
                _publish_progress(
                    session_id, "agent_tool_result", agent="financial",
                    message=f"Financial Agent: '{tool}' completed",
                    detail={"tool": tool},
                )
                return data
            except Exception as exc:
                errors.append(f"{tool} exception: {exc}")
                log.exception("Executor: SEC chain step %s failed", tool)
                capture_exception("mcp.financial", exc, tags={"tool": tool})
                return None

        cik_data = await call("tool_get_cik", {"ticker": ticker})
        cik = cik_data.get("cik") if cik_data else None
        if not cik:
            workflow.update(status="unavailable", error="CIK was not returned; downstream SEC steps skipped.")
            return
        workflow["cik"] = cik

        filings_data = await call(
            "tool_list_filings", {"ticker": ticker, "form_type": form_type, "limit": 5},
        )
        filings = filings_data.get("filings", []) if filings_data else []
        filing_candidates = [
            item for item in filings
            if isinstance(item, dict) and item.get("accession_number")
        ]
        if not filing_candidates:
            workflow.update(status="unavailable", error="No real accession number was returned; filing text skipped.")
            return

        workflow["attempted_filings"] = []
        for filing in filing_candidates:
            accession = filing["accession_number"]
            workflow["attempted_filings"].append(accession)
            text_data = await call(
                "tool_get_filing_text",
                {
                    "accession_number": accession,
                    "cik": cik,
                    "primary_document": filing.get("primary_document"),
                    "form_type": form_type,
                },
            )
            if text_data is None or text_data.get("validation_status") != "VALID":
                continue
            workflow.update(
                status="complete",
                selected_filing=dict(filing),
                filing_text=text_data,
                validation_status="VALID",
            )
            return

        workflow.update(
            status="unavailable",
            validation_status="INVALID",
            error="No filing candidate passed primary-document validation.",
        )

    @traceable(name="financial.extract", run_type="tool")
    async def _execute_data_extraction(
        self,
        session: ClientSession,
        state:   FinancialAgentState,
    ) -> None:
        
        directives = state["manager_directives"]
        task_query = state["task_query"]
        ticker     = extract_ticker(task_query, directives)

        if not ticker:
            state["raw_numerical_data"]["extraction_errors"] = [
                "Could not determine ticker symbol from task_query or manager_directives."
            ]
            log.error("Executor: ticker extraction failed — aborting data pull.")
            return

        state["raw_numerical_data"]["ticker"] = ticker
        raw = state["raw_numerical_data"]
        errors: list[str] = []
        session_id = session_from_directives(state["manager_directives"])
        _publish_progress(
            session_id, "agent_brain", agent="financial",
            message=f"Financial Agent: starting raw data extraction for {ticker}",
            detail={"ticker": ticker},
        )

        extraction_tools, _ = self._resolve_tools(state)
        log.info("Executor: intent-resolved extraction tools = %s", extraction_tools)
        feedback = (state.get("validation_feedback") or "").lower()
        pending: list[str] = []
        for tool in extraction_tools:
            _, store_key = EXTRACTION_REGISTRY[tool]
            cached = raw.get(store_key)
            already_good = isinstance(cached, dict) and bool(cached) and not cached.get("error")
            if already_good and tool.lower() not in feedback:
                log.info("Executor: reusing cached %s from a previous loop.", tool)
                continue
            pending.append(tool)

        if not pending:
            log.info("Executor: all extraction tools served from cache — no calls made.")

        async def run_tool(tool: str) -> None:
            """Call one extraction tool and store its payload under raw[store_key]."""
            args_builder, store_key = EXTRACTION_REGISTRY[tool]
            args = args_builder(ticker, raw)

            if args is None:
                industry = raw.get("yahoo_ratios", {}).get("industry", "")
                log.info(
                    "Executor: skipping %s — inputs unavailable for ticker=%s "
                    "(industry=%r).", tool, ticker, industry,
                )
                if tool == "tool_get_peer_comparison":
                    raw["peer_comparison"] = {
                        "note": f"No peer mapping available for industry={industry!r}.",
                    }
                return

            log.info("Executor: calling %s for %s", tool, ticker)
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message=f"Financial Agent: calling tool '{tool}'...",
                detail={"tool": tool, **({"peers": args["peers"]} if "peers" in args else {})},
            )
            try:
                add_breadcrumb(
                    "mcp.financial", f"Calling {tool}",
                    {"tool": tool, "ticker": ticker},
                )
                result = await traced_call_tool(session, tool, args)
                data = json.loads(result.content[0].text)
                if data.get("error"):
                    errors.append(f"{tool} error: {data['error']}")
                raw[store_key] = data
            except Exception as exc:
                errors.append(f"{tool} exception: {exc}")
                raw[store_key] = {}
                log.exception("Executor: %s failed", tool)
                capture_exception("mcp.financial", exc, tags={"tool": tool})

            _publish_progress(
                session_id, "agent_tool_result", agent="financial",
                message=f"Financial Agent: '{tool}' completed",
                detail={"tool": tool},
            )

        peer_tool = "tool_get_peer_comparison"
        wave_one = [t for t in pending if t != peer_tool]
        if wave_one:
            await asyncio.gather(*(run_tool(t) for t in wave_one))
        if peer_tool in pending:
            await run_tool(peer_tool)

        if pending:
            _publish_progress(
                session_id, "agent_tool_result", agent="financial",
                message="Financial Agent: data extraction done",
                detail={"tools": len(pending), "errors": len(errors)},
            )

        form_type = select_sec_filing_type(state)
        prior_filing = raw.get("sec_filing", {})
        already_filed = (
            prior_filing.get("status") == "complete"
            and prior_filing.get("form_type") == form_type
        )
        if form_type is not None and not already_filed:
            await self._execute_sec_filing_chain(session, state, ticker, form_type, errors)
        elif already_filed:
            log.info("Executor: reusing %s filing chain result from a previous loop.", form_type)

        raw["extraction_errors"] = errors
        log.info(
            "Executor: data extraction complete — %d error(s) recorded.", len(errors)
        )

    @traceable(name="financial.compute", run_type="tool")
    async def _execute_ratio_computation(
        self,
        session: ClientSession,
        state:   FinancialAgentState,
    ) -> None:
       
        raw        = state["raw_numerical_data"]
        xbrl_src   = raw.get("xbrl_financials", {})
        session_id = session_from_directives(state["manager_directives"])

        
        async def _call(
            tool: str,
            args: dict,
            period: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> dict:
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message=f"Financial Agent: computing '{tool}'...",
                detail={"tool": tool},
            )
            try:
                add_breadcrumb("mcp.financial", f"Calling {tool}", {"tool": tool})
                result = await traced_call_tool(session, tool, args)
                parsed = json.loads(result.content[0].text)
                parsed["_inputs"] = dict(args)
                parsed["_period"] = period
                default_state = (
                    "UNAVAILABLE" if parsed.get("error")
                    else "DEGRADED" if str(period or "").upper().startswith("MIXED")
                    else "VALID_PRIMARY"
                )
                parsed.setdefault("validation_status", default_state)
                parsed.setdefault("source", "calculated_from_validated_inputs")
                parsed.setdefault("exact_period", period)
                parsed.setdefault("fallback_reason", None)
                parsed.setdefault("confidence", {
                    "VALID_PRIMARY": "high",
                    "FALLBACK": "medium",
                    "DEGRADED": "low",
                    "UNAVAILABLE": "none",
                }[default_state])
                parsed.setdefault("is_approximation", False)
                if metadata:
                    parsed.update(metadata)
                if parsed.get("error"):
                    parsed["validation_status"] = "UNAVAILABLE"
                    parsed["confidence"] = "none"
                    parsed["fallback_reason"] = (
                        f"Metric calculation failed after compatible inputs were selected: "
                        f"{parsed.get('error')}"
                    )
                _publish_progress(
                    session_id, "agent_tool_result", agent="financial",
                    message=f"Financial Agent: '{tool}' computed",
                    detail={"tool": tool, "outcome": "success"},
                )
                return parsed
            except Exception as exc:
                log.warning("Ratio computation tool '%s' failed: %s", tool, exc)
                _publish_progress(
                    session_id, "agent_tool_result", agent="financial",
                    message=f"Financial Agent: '{tool}' failed",
                    detail={"tool": tool, "outcome": "error", "error": str(exc)},
                )
                capture_exception("mcp.financial", exc, tags={"tool": tool})
                failure = {
                    "error": str(exc),
                    "_inputs": dict(args),
                    "_period": period,
                    "validation_status": "UNAVAILABLE",
                    "source": "calculation_error",
                    "exact_period": period,
                    "formula": None,
                    "fallback_reason": f"Metric calculation failed: {exc}",
                    "confidence": "none",
                    "is_approximation": False,
                }
                if metadata:
                    for key in ("source", "exact_period", "formula", "semantic_name"):
                        if key in metadata:
                            failure[key] = metadata[key]
                return failure

       
        state["calculated_ratios"]["earnings_quality"] = normalize_earnings_quality(xbrl_src)

        _, compute_tools = self._resolve_tools(state)
        log.info("Executor: intent-resolved compute tools = %s", compute_tools)

        d = derive_compute_facts(
            raw, state["calculated_ratios"], depth=self._resolve_depth(state),
        )

        for tool in compute_tools:
            spec = COMPUTE_REGISTRY[tool]
            plan = spec.mapper(d)

            if plan is None:
                
                log.info("Compute: skipping %s — required inputs missing.", tool)
                unavailable = dict(spec.unavailable)
                unavailable.update({
                    "validation_status": "UNAVAILABLE",
                    "source": "unavailable",
                    "exact_period": None,
                    "formula": unavailable.get("formula"),
                    "fallback_reason": unavailable.get("error") or "Required compatible inputs were unavailable.",
                    "confidence": "none",
                    "is_approximation": False,
                })
                state["calculated_ratios"][spec.key] = unavailable
                continue

            if isinstance(plan, dict):
                plan.setdefault(
                    "validation_status",
                    "DEGRADED"
                    if str(plan.get("_period") or "").upper().startswith("MIXED")
                    else "VALID_PRIMARY",
                )
                plan.setdefault("source", "direct_structured_metric")
                plan.setdefault("exact_period", plan.get("_period"))
                plan.setdefault("formula", None)
                plan.setdefault("fallback_reason", None)
                plan.setdefault("confidence", "low" if plan["validation_status"] == "DEGRADED" else "high")
                plan.setdefault("is_approximation", False)
                state["calculated_ratios"][spec.key] = plan
                continue

            if len(plan) == 3:
                args, period, metadata = plan
            else:
                args, period = plan
                metadata = None
            log.info("Computing %s (period=%s)", tool, period)
            result = await _call(tool, args, period=period, metadata=metadata)
            if spec.post is not None:
                spec.post(result, d)
            state["calculated_ratios"][spec.key] = result

        log.info("Executor: ratio computation complete.")

    @traceable(name="financial.checker", run_type="llm")
    async def _check_data_quality(self, state: FinancialAgentState) -> dict[str, Any]:
        
        raw  = state["raw_numerical_data"]
        calc = state["calculated_ratios"]
        loop = state["loop_counter"]
        task_query = state["task_query"]

        log.info("[Checker] Auditing iteration %d data with Claude...", loop)

        # -- Pre-flight guard: skip Claude call if state is obviously empty ---
        if not raw and not calc:
            log.warning("[Checker] Pre-flight: both raw_numerical_data and "
                        "calculated_ratios are empty — skipping Claude call.")
            return {
                "is_complete": False,
                "score":       0,
                "passed":      [],
                "failed":      ["DATA PRESENCE", "REVENUE HISTORY",
                                "CORE RATIO COVERAGE", "VALUATION SANITY",
                                "COMPOSITE SCORE", "INTERNAL CONSISTENCY",
                                "MANAGER READINESS"],
                "issues":      ["No data was extracted. The Executor produced "
                                "empty raw_numerical_data and calculated_ratios."],
                "feedback":    (
                    "No data was retrieved in this iteration. Re-call "
                    "tool_get_financial_ratios, tool_get_revenue_growth, and "
                    "tool_get_xbrl_financials with the correct ticker symbol."
                ),
            }

        # -- Assemble the audit payload for Claude ----------------------------
        yahoo   = raw.get("yahoo_ratios", {})
        growth  = raw.get("revenue_growth", {})
        xbrl    = raw.get("xbrl_financials", {})

        raw_summary = {
            "ticker":            raw.get("ticker"),
            "yahoo_error":       yahoo.get("error"),
            "current_price":     yahoo.get("current_price"),
            "eps_trailing":      yahoo.get("eps_trailing"),
            "pe_ratio_yahoo":    yahoo.get("pe_ratio"),
            "market_cap":        yahoo.get("market_cap"),
            "sector":            yahoo.get("sector"),
            "annual_revenue_count": len(growth.get("annual_revenue", [])),
            "annual_revenue_sample": growth.get("annual_revenue", [])[:3],
            "revenue_growth_ttm":   growth.get("revenue_growth_ttm"),
            "xbrl_assets_available":      bool(xbrl.get("total_assets")),
            "xbrl_liabilities_available": bool(xbrl.get("total_liabilities")),
            "extraction_errors": raw.get("extraction_errors", []),
        }

        ratio_summary = {
            "pe":             calc.get("pe", {}),
            "roe":            calc.get("roe", {}),
            "net_margin":     calc.get("net_margin", {}),
            "earnings_quality": calc.get("earnings_quality", {}),
            "de_ratio":       calc.get("de_ratio", {}),
            "cagr":           calc.get("cagr", {}),
            "composite_score": calc.get("composite_score", {}),
            "forecast": calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "valuation_input_assessments": (
                calc.get("dcf", {}).get("base", {}).get("input_assessments", {})
            ),
            "reverse_dcf": calc.get("reverse_dcf", {}),
        }
        for metric_key in SUMMARY_CALCULATED_PASSTHROUGH:
            ratio_summary.setdefault(metric_key, calc.get(metric_key, {}))
        ratio_summary["peer_comparison"] = raw.get("peer_comparison", {})
        intent_requirements = checker_requirements(state)

        deterministic_validation = evaluate_financial_sanity({
            "forecast": calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "reverse_dcf": calc.get("reverse_dcf", {}),
            "earnings_quality": calc.get("earnings_quality", {}),
        })
        if not deterministic_validation["passed"]:
            issues = [item["message"] for item in deterministic_validation["critical_failures"]]
            return {
                "is_complete": False,
                "score": deterministic_validation["score"],
                "passed": [],
                "failed": [item["rule"] for item in deterministic_validation["critical_failures"]],
                "issues": issues,
                "feedback": "Correct the deterministic financial-model failures before rerunning valuation.",
                "deterministic_validation": deterministic_validation,
                "intent_requirements": intent_requirements,
            }
        if not intent_requirements["passed"]:
            missing = intent_requirements["missing_groups"]
            log.warning(
                "[Checker] Intent coverage failed before LLM audit — missing=%s", missing,
            )
            _publish_progress(
                session_from_directives(state["manager_directives"]),
                "agent_checker", agent="financial",
                message="Financial Agent: checking data quality — needs more data",
                detail={"is_complete": False, "missing_groups": missing},
            )
            return {
                "is_complete": False,
                "score": deterministic_validation["score"],
                "passed": [],
                "failed": ["INTENT-SPECIFIC COVERAGE"],
                "issues": [f"Missing required metric groups for active intents: {missing}"],
                "feedback": (
                    "Re-run the tools for these missing intent-specific metric groups: "
                    f"{missing}. Do not fetch unrelated analysis bundles."
                ),
                "deterministic_validation": deterministic_validation,
                "intent_requirements": intent_requirements,
            }

        user_content = (
            f"TASK QUERY:\n{task_query}\n\n"
            f"LOOP ITERATION: {loop}\n\n"
            f"RAW DATA SUMMARY:\n{json.dumps(raw_summary, indent=2)}\n\n"
            f"CALCULATED RATIOS:\n{json.dumps(ratio_summary, indent=2)}\n\n"
            f"ACTIVE INTENT COMPLETENESS CONTRACT:\n"
            f"{json.dumps(intent_requirements, indent=2)}\n\n"
            f"DETERMINISTIC FINANCIAL VALIDATION:\n{json.dumps(deterministic_validation, indent=2)}\n\n"
            "Audit only the active intent contract for completeness. Missing metrics "
            "outside that contract are irrelevant; every required any-of group must pass. "
            "Return your JSON verdict."
        )

        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=768,
                system=CHECKER_SYSTEM_PROMPT,
                temperature=0.0,
                messages=[{"role": "user", "content": user_content}],
            )
            verdict_text = response.content[0].text.strip()

            verdict_text = verdict_text.replace("```json", "").replace("```", "").strip()
            verdict = json.loads(verdict_text)

        except json.JSONDecodeError as exc:
            log.warning("[Checker] Claude returned non-JSON verdict: %s | raw=%s",
                        exc, verdict_text[:300])
            return {
                "is_complete": False,
                "score":       0,
                "passed":      [],
                "failed":      ["CHECKER PARSE ERROR"],
                "issues":      [f"Claude Checker returned malformed JSON: {exc}"],
                "feedback":    "Checker could not parse its own verdict. "
                               "Re-run full extraction with tool_get_financial_ratios, "
                               "tool_get_revenue_growth, and tool_get_xbrl_financials.",
            }

        except Exception as exc:
            log.exception("[Checker] Claude API call failed: %s", exc)
            
            return {
                "is_complete": False,
                "score":       0,
                "passed":      [],
                "failed":      ["CHECKER API ERROR"],
                "issues":      [f"Claude Checker API call failed: {exc}"],
                "feedback":    "Checker API call failed. Retry full data extraction "
                               "on the next iteration.",
            }

       
        is_complete = bool(verdict.get("is_complete", False))
        score       = int(verdict.get("score", 0))
        passed      = verdict.get("passed", [])
        failed      = verdict.get("failed", [])
        issues      = verdict.get("issues", [])
        feedback    = str(verdict.get("feedback", "")) if not is_complete else ""

        if is_complete:
            log.info(
                "[Checker] PASS — score=%d | criteria passed=%d | iteration=%d",
                score, len(passed), loop,
            )
        else:
            log.warning(
                "[Checker] FAIL — score=%d | failed=%s | issues=%s",
                score, failed, issues,
            )

        _publish_progress(
            session_from_directives(state["manager_directives"]), "agent_checker", agent="financial",
            message=(
                f"Financial Agent: checking data quality — "
                f"{'sufficient ✓' if is_complete else 'needs more data'} (score={score})"
            ),
            detail={"is_complete": is_complete, "score": score, "failed": failed},
        )

        return {
            "is_complete": is_complete,
            "score":       score,
            "passed":      passed,
            "failed":      failed,
            "issues":      issues,
            "feedback":    feedback,
            "deterministic_validation": deterministic_validation,
            "intent_requirements": intent_requirements,
        }

    # =========================================================================
    # LAYER 3 — BRAIN / ORCHESTRATOR (Lifecycle & Loop Gateway)
    # =========================================================================

    @traceable(name="financial.brain", run_type="llm")
    async def _brain(self, state: FinancialAgentState) -> dict[str, Any]:
        
        iteration   = state["loop_counter"]
        feedback    = state["validation_feedback"]
        task_query  = state["task_query"]
        directives  = state["manager_directives"]

        

        user_message = (
            f"ITERATION: {iteration}\n"
            f"TASK QUERY: {task_query}\n"
            f"MANAGER DIRECTIVES: {json.dumps(directives)}\n"
            f"CHECKER FEEDBACK: {feedback if feedback else 'None — first iteration or prior run passed.'}\n\n"
            "Classify the analysis intents for this iteration."
        )

        
        state["messages"].append({"role": "user", "content": user_message})

        log.info("Brain: invoking Claude for iteration %d intent classification...", iteration)
        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=512,
                temperature=0.0,
                system=system_prompt,
                messages=state["messages"],
            )
            raw_plan = response.content[0].text.strip()
            state["messages"].append({"role": "assistant", "content": raw_plan})

            clean = raw_plan.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            result = self._validate_brain_output(parsed)
            result["raw_plan"] = raw_plan
            log.info(
                "Brain intents for iteration %d: %s (%s)",
                iteration, result["intents"], result["rationale"],
            )
            return result

        except json.JSONDecodeError:
            
            log.warning("Brain: Claude response was not valid JSON — defaulting to overview.")
            return {
                "intents": ["overview"], "extra_tools": [],
                "rationale": "Brain response was not valid JSON; defaulted to overview.",
                "raw_plan": raw_plan,
            }

        except Exception as exc:
            log.exception("Brain: Claude API call failed — defaulting to overview.")
            capture_exception(
                "financial.brain", exc, tags={"iteration": str(iteration)},
            )
            default_plan = "Default plan: overview bundle (extraction + core ratios)."
            state["messages"].append({"role": "assistant", "content": default_plan})
            return {
                "intents": ["overview"], "extra_tools": [],
                "rationale": "Brain API call failed; defaulted to overview.",
                "raw_plan": default_plan,
            }

    @staticmethod
    def _validate_brain_output(parsed: dict[str, Any]) -> dict[str, Any]:
        
        raw_intents = parsed.get("intents") or []
        if not isinstance(raw_intents, list):
            raw_intents = []
        intents = [i for i in raw_intents if isinstance(i, str) and i in INTENTS]

        raw_extras = parsed.get("extra_tools") or []
        if not isinstance(raw_extras, list):
            raw_extras = []
        known_tools = set(EXTRACTION_REGISTRY) | set(COMPUTE_REGISTRY)
        extra_tools = [t for t in raw_extras if isinstance(t, str) and t in known_tools]

        if not intents:
            intents = ["overview"]

        return {
            "intents": intents,
            "extra_tools": extra_tools,
            "rationale": str(parsed.get("rationale", "")),
        }

    # =========================================================================
    # INTENT → TOOL RESOLUTION
    # =========================================================================

    def _resolve_tools(
        self, state: FinancialAgentState,
    ) -> tuple[list[str], list[str]]:
        
     
        chosen = list(state.get("intents") or []) or ["overview"]

        if "full_report" in chosen:
            chosen = [i for i in chosen if i != "full_report"] + list(FULL_REPORT_PROFILES)

        extraction: set[str] = set(BASELINE_EXTRACTION)
        compute:    set[str] = set(BASELINE_COMPUTE)
        for intent in chosen:
            profile = ANALYSIS_PROFILES.get(intent)
            if profile is None:
                continue
            extraction.update(profile["extraction"])
            compute.update(profile["compute"])

        for tool in state.get("extra_tools") or []:
            if tool in EXTRACTION_REGISTRY:
                extraction.add(tool)
            elif tool in COMPUTE_REGISTRY:
                compute.add(tool)

        if self._resolve_depth(state) == "shallow":
            compute -= MONTE_CARLO_TOOLS

        return (
            [t for t in EXTRACTION_ORDER if t in extraction],
            [t for t in COMPUTE_ORDER if t in compute],
        )

    @staticmethod
    def _resolve_depth(state: FinancialAgentState) -> str:
        """Normalized analysis depth for this iteration.

        Single source of truth for two consumers that must not diverge:
        _resolve_tools() drops the Monte-Carlo tools at "shallow", and
        derive_compute_facts() sizes the Monte-Carlo sample count from it.
        """
        depth = normalize_depth(state["manager_directives"])
        if "full_report" in (state.get("intents") or []) and depth != "shallow":
            return "deep"
        return depth

    # =========================================================================
    # LANGGRAPH NODES — thin wrappers over the layer methods above
    # =========================================================================
   

    async def _node_brain(self, state: FinancialAgentState) -> dict[str, Any]:
     
        new_counter = state["loop_counter"] + 1
        log.info("--- Loop iteration %d ---", new_counter)
        brain_output = await self._brain(state)
        log.info(
            "Brain intents: %s — %s",
            brain_output["intents"], brain_output["rationale"],
        )
       
        return {
            "loop_counter": new_counter,
            "intents":      brain_output["intents"],
            "extra_tools":  brain_output["extra_tools"],
            "rationale":    brain_output["rationale"],
        }

    async def _node_extract(self, state: FinancialAgentState) -> dict[str, Any]:
        await self._execute_data_extraction(self._session, state)
        return {"raw_numerical_data": state["raw_numerical_data"]}

    async def _node_compute(self, state: FinancialAgentState) -> dict[str, Any]:
        await self._execute_ratio_computation(self._session, state)
        return {"calculated_ratios": state["calculated_ratios"]}

    async def _node_checker(self, state: FinancialAgentState) -> dict[str, Any]:
        check_result = await self._check_data_quality(state)
        is_complete = check_result["is_complete"]
        if is_complete:
            log.info("[Checker] PASSED (score=%s) after %d iteration(s).",
                     check_result.get("score"), state["loop_counter"])
        else:
            log.warning("[Checker] FAILED iteration %d (score=%s). Feedback: %s",
                        state["loop_counter"], check_result.get("score"),
                        str(check_result.get("feedback", ""))[:120])
        # Cached so run() doesn't recompute evaluate_financial_sanity() over
        # the same inputs after the graph finishes.
        return {
            "is_complete":         is_complete,
            "validation_feedback": check_result["feedback"],
            "financial_sanity":    check_result.get("deterministic_validation", {}),
        }

    def _should_continue(self, state: FinancialAgentState):
       
        directives = state["manager_directives"]
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        if state["loop_counter"] >= max_loops:
            log.warning("[Router] loop_counter=%d >= max_loops=%d → END.",
                        state["loop_counter"], max_loops)
            return "__end__"
        if state["is_complete"]:
            return "__end__"
        return "brain"

    def _build_graph(self):
       
        builder = StateGraph(FinancialAgentState)

        builder.add_node("brain",    self._node_brain)
        builder.add_node("extract",  self._node_extract)
        builder.add_node("compute",  self._node_compute)
        builder.add_node("checker",  self._node_checker)

        builder.set_entry_point("brain")
        builder.add_edge("brain",   "extract")
        builder.add_edge("extract", "compute")
        builder.add_edge("compute", "checker")
        builder.add_conditional_edges(
            "checker",
            self._should_continue,
            {"brain": "brain", "__end__": END},
        )

        return builder.compile()

    # =========================================================================
    # ENTRY GATEWAY — run()
    # =========================================================================

    @staticmethod
    def _trim_filing(sec_filing: dict[str, Any]) -> dict[str, Any]:
        """Cap filing prose before it crosses into the Manager's state.

        The identifiers (CIK, accession number, form type) are what make the
        filing citable and are always kept in full; only the body text is cut.
        """
        if not isinstance(sec_filing, dict) or not sec_filing:
            return {}

        trimmed = dict(sec_filing)
        filing_text = trimmed.get("filing_text")
        if not isinstance(filing_text, dict):
            return trimmed

        filing_text = dict(filing_text)
        body = filing_text.get("text")
        if isinstance(body, str) and len(body) > _FILING_TEXT_CHAR_LIMIT:
            filing_text["text"] = body[:_FILING_TEXT_CHAR_LIMIT]
            filing_text["text_truncated"] = True
            filing_text["text_original_chars"] = len(body)
        trimmed["filing_text"] = filing_text
        return trimmed

    @traceable(name="FinancialAnalystAgent.run", run_type="chain")
    async def run(self, shared_state: SharedManagerState) -> SharedManagerState:
        
        directives = shared_state.get("manager_directives", {})
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        log.info(
            "FinancialAnalystAgent.run() started — task_query='%s', max_loops=%d",
            shared_state.get("task_query", ""),
            max_loops,
        )

        # -- Initialise agent-local state -------------------------------------
        state: FinancialAgentState = {
            "messages":            [],
            "raw_numerical_data":  {},
            "calculated_ratios":   {},
            "loop_counter":        0,
            "validation_feedback": "",
            "is_complete":         False,
            "task_query":          shared_state["task_query"],
            "manager_directives":  dict(shared_state.get("manager_directives", {})),
            
            "intents":             [],
            "extra_tools":         [],
            "rationale":           "",
            "financial_sanity":    {},
        }

   
        async with stdio_client(self._server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                await session.initialize()
                log.info("MCP session initialised successfully.")

                
                self._session = session

              
                try:
                    state = await self._graph.ainvoke(
                        state,
                        config={"recursion_limit": (max_loops + 1) * 4},
                    )
                finally:
                    self._session = None

                if not state["is_complete"]:
                    log.warning(
                        "Safety guardrail hit: exiting after %d loops without full validation.",
                        state["loop_counter"],
                    )

        
        raw  = state["raw_numerical_data"]
        calc = state["calculated_ratios"]

       
        annotate_confidence(calc)
        consistency_checks = build_consistency_checks(raw, calc)
        data_quality = build_data_quality_score(raw, calc)
        # Reuse the checker's verdict when it ran on this same final state.
        # It is only recomputed if the graph exited before any checker pass
        # (e.g. max_loops reached on a path that never populated it).
        financial_sanity = state.get("financial_sanity") or evaluate_financial_sanity({
            "forecast": calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "reverse_dcf": calc.get("reverse_dcf", {}),
            "earnings_quality": calc.get("earnings_quality", {}),
        })

        financial_facts = []
        for series_name, entries in raw.get("xbrl_financials", {}).items():
            if not isinstance(entries, list):
                continue
            metric = series_name.removesuffix("_annual")
            for entry in entries[:_FACTS_YEARS_RETAINED]:
                if not isinstance(entry, dict) or "value" not in entry:
                    continue
                financial_facts.append(typed_financial_claim({
                    "metric": metric, "value": entry.get("value"), "unit": entry.get("unit"),
                    "ticker": raw.get("ticker"), "fiscal_year": entry.get("fiscal_year"),
                    "fiscal_quarter": entry.get("fiscal_quarter"),
                    "period_start": entry.get("period_start"), "period_end": entry.get("period_end"),
                    "duration_months": entry.get("duration_months"),
                    "filing_type": entry.get("filing_type"),
                    "accession_number": entry.get("accession_number"),
                    "filed_at": entry.get("filed_at"),
                    "xbrl_concept": entry.get("xbrl_concept"),
                    "metric_scope": entry.get("metric_scope"),
                    "gaap_status": entry.get("gaap_status"),
                    "validation_status": entry.get("validation_status"),
                    "source": entry.get("source"),
                    "selected_source": entry.get("selected_source"),
                    "exact_period": entry.get("exact_period"),
                    "fallback_reason": entry.get("fallback_reason"),
                    "confidence": entry.get("confidence"),
                    "is_approximation": entry.get("is_approximation", False),
                    "provenance_status": "COMPLETE" if entry.get("accession_number") else "INCOMPLETE",
                    # Distinct historical facts are retained, not merged into one claim.
                    "comparison": True,
                }))

        current_ratio_metric = calc.get("current_ratio_calc", {})
        if not current_ratio_metric and raw.get("yahoo_ratios", {}).get("current_ratio") is not None:
            yahoo = raw.get("yahoo_ratios", {})
            current_ratio_metric = {
                "current_ratio": yahoo.get("current_ratio"),
                "interpretation": "sourced_from_yahoo",
                "semantic_name": "current_ratio",
                "formula": "Current Assets / Current Liabilities",
                "source": "yahoo_finance.currentRatio",
                "exact_period": {
                    "basis": "snapshot", "period_start": None,
                    "period_end": yahoo.get("financial_data_period_end"),
                },
                "fallback_reason": "Current Ratio compute path was not active; preserved Yahoo's direct metric.",
                "confidence": "medium",
                "is_approximation": False,
                "validation_status": "FALLBACK",
                "_inputs": {"current_ratio": yahoo.get("current_ratio")},
                "_period": f"snapshot:{yahoo.get('financial_data_period_end') or 'mrq'}",
            }

        financial_metrics_summary: dict[str, Any] = {
            # Identification
            "ticker":        raw.get("ticker"),
            "company_name":  raw.get("yahoo_ratios", {}).get("company_name"),
            "sector":        raw.get("yahoo_ratios", {}).get("sector"),
            "industry":      raw.get("yahoo_ratios", {}).get("industry"),

            # Market data
            "current_price": raw.get("yahoo_ratios", {}).get("current_price"),
            "market_cap":    raw.get("yahoo_ratios", {}).get("market_cap"),
            "beta":          raw.get("yahoo_ratios", {}).get("beta"),
            "52w_high":      raw.get("yahoo_ratios", {}).get("52w_high"),
            "52w_low":       raw.get("yahoo_ratios", {}).get("52w_low"),

            # Calculated ratios (verified)
            "pe_ratio":      calc.get("pe", {}),
            "forward_pe":    raw.get("yahoo_ratios", {}).get("forward_pe"),
            "peg_ratio":     raw.get("yahoo_ratios", {}).get("peg_ratio"),
            "roe":           calc.get("roe", {}),
            "net_margin":    calc.get("net_margin", {}),
            "de_ratio":      calc.get("de_ratio", {}),
            "revenue_cagr":  calc.get("cagr", {}),
            "forecast":      calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "valuation_input_assessments": (
                calc.get("dcf", {}).get("base", {}).get("input_assessments", {})
            ),
            "dcf_monte_carlo": calc.get("dcf_monte_carlo", {}),
            "current_ratio": current_ratio_metric,
            "composite_score": calc.get("composite_score", {}),

         
            "net_debt_to_ebitda": calc.get("net_debt_to_ebitda", {}),
            "fcf_margin":         calc.get("fcf_margin", {}),
            "fcf_yield":          calc.get("fcf_yield", {}),
            "ocf_ratio":          calc.get("ocf_ratio", {}),
            "reverse_dcf":        calc.get("reverse_dcf", {}),
            "roic":               calc.get("roic", {}),
            "roce":               calc.get("roce", {}),
            "debt_ratio":         calc.get("debt_ratio", {}),
            "equity_ratio":       calc.get("equity_ratio", {}),
            "altman_z_score":     calc.get("altman_z_score", {}),

            "peer_comparison": raw.get("peer_comparison", {}),

            
            "annual_revenue_history": raw.get("revenue_growth", {}).get("annual_revenue", []),
            "revenue_growth_ttm":     raw.get("revenue_growth", {}).get("revenue_growth_ttm"),

            # SEC EDGAR
            "xbrl_total_assets":      (
                raw.get("xbrl_financials", {}).get("total_assets", [{}])[:1]
            ),
            "xbrl_total_liabilities": (
                raw.get("xbrl_financials", {}).get("total_liabilities", [{}])[:1]
            ),

            "capital_allocation": {
                "operating_cash_flow": (
                    raw.get("xbrl_financials", {}).get("operating_cash_flow_annual", [{}])[:1]
                ),
                "capex": (
                    raw.get("xbrl_financials", {}).get("capex_annual", [{}])[:1]
                ),
                "dividends_paid": (
                    raw.get("xbrl_financials", {}).get("dividends_paid_annual", [{}])[:1]
                ),
                "buybacks": (
                    raw.get("xbrl_financials", {}).get("buybacks_annual", [{}])[:1]
                ),
            },

            "consistency_checks": consistency_checks,
            "data_quality":       data_quality,
            "financial_sanity":   financial_sanity,
            "financial_facts":    financial_facts,

            "loop_iterations_used":  state["loop_counter"],
            "validation_passed":     state["is_complete"],
            "extraction_errors":     raw.get("extraction_errors", []),
            "analysis_intents":      state.get("intents", []),
            "analysis_rationale":    state.get("rationale", ""),
        }

        surface_calculated_ratios(financial_metrics_summary, calc)
        financial_metrics_summary["sec_filing"] = self._trim_filing(raw.get("sec_filing", {}))

      
        financial_metrics_summary = sanitize_nans(financial_metrics_summary)

        # Commit to the shared state contract
        shared_state["financial_metrics_summary"] = financial_metrics_summary  # type: ignore[literal-required]

        log.info(
            "FinancialAnalystAgent.run() complete — ticker=%s, score=%s, grade=%s",
            financial_metrics_summary["ticker"],
            calc.get("composite_score", {}).get("score"),
            calc.get("composite_score", {}).get("grade"),
        )

        return shared_state
