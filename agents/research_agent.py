"""agents/research_agent.py — ResearchAgent
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Literal

import anthropic
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.state import ResearchAgentState, SharedManagerState
from dotenv import load_dotenv
from langsmith import traceable
from core.observability import sentry_enabled
from core.progress_bus import publish as _publish_progress, session_from_directives
from tools.research_tools.context_synthesizer import synthesize_research_context
from agents.prompts.propt_search_agent import (BRAIN_SYSTEM_PROMPT, CHECKER_SYSTEM_PROMPT)
from agents.executor_helpers.search_executor_helpers import (TICKER_ARG_BY_TOOL, 
                                                             ensure_ticker_argument, 
                                                             tool_call_signature, 
                                                             sec_filing_doc_key, 
                                                             sec_filing_sections, 
                                                             build_sec_filing_coverage, 
                                                             format_skip_notice)

load_dotenv()

logger = logging.getLogger(__name__)

# ── Default guardrail ────────────────────────────────────────────
_DEFAULT_MAX_LOOPS: int = 3

# ── MCP Server launch parameters ────────────────────────────────
_MCP_SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[os.path.join(os.path.dirname(__file__), "..","tools","research_tools" ,"research_server.py")],
    env=None,   # inherits current environment (API keys etc.)
)


# ══════════════════════════════════════════════════════════════════
# ResearchAgent
# ══════════════════════════════════════════════════════════════════

class ResearchAgent:
    """
    LangGraph-powered Research Agent for the Alpha-Agent Node platform.

    Encapsulates:
      - Anthropic claude-haiku-4-5 as the planning and validation LLM.
      - MCP client connection to research_server.py (7 research tools).
      - A 3-node LangGraph state machine: Brain → Executor → Checker.
      - A conditional routing edge with a max-loop guardrail.

    Contract
    ─────────
      Input  : SharedManagerState  (from Manager Agent)
      Output : SharedManagerState  (populated aggregated_research_context)

    Parameters
    ----------
    anthropic_api_key : str, optional
        Anthropic API key. Defaults to env var ANTHROPIC_API_KEY.
    model : str, optional
        Claude model identifier. Default: "claude-haiku-4-5".
    max_loops : int, optional
        Maximum Brain→Executor→Checker iterations before forced exit.
        Default: 3.
    mcp_server_params : StdioServerParameters, optional
        MCP server launch parameters. Defaults to research_server.py.

    Example
    -------
    >>> agent = ResearchAgent()
    >>> shared = SharedManagerState(
    ...     task_query="Analyze NVIDIA competitive position in AI chips",
    ...     aggregated_research_context=[],
    ...     manager_directives={"ticker": "NVDA", "days_back": 14},
    ... )
    >>> result = asyncio.run(agent.run(shared))
    >>> print(len(result["aggregated_research_context"]), "chunks collected")
    """

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        model: str = "claude-haiku-4-5",
        max_loops: int = _DEFAULT_MAX_LOOPS,
        mcp_server_params: StdioServerParameters | None = None,
        llm_client: anthropic.Anthropic | None = None,
    ) -> None:
   
        self._llm = llm_client or anthropic.Anthropic(
            api_key=anthropic_api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self._model    = model
        self._max_loops = max_loops
        self._mcp_params = mcp_server_params or _MCP_SERVER_PARAMS

        # ── Compile the internal LangGraph state machine ─────────
        self._graph = self._build_graph()
        logger.info(
            "ResearchAgent initialised (model=%s, max_loops=%d).",
            model, max_loops,
        )

    # ══════════════════════════════════════════════════════════════
    # Public gateway
    # ══════════════════════════════════════════════════════════════

    @traceable(name="ResearchAgent.run", run_type="chain")
    async def run(self, shared_state: SharedManagerState) -> SharedManagerState:
        """
        Public entry point called by the Manager Agent.

        Translates the incoming SharedManagerState into an isolated
        ResearchAgentState, runs the internal LangGraph loop, then
        maps the gathered context_chunks back into the shared state's
        aggregated_research_context before returning.

        Parameters
        ----------
        shared_state : SharedManagerState
            Contract object from the Manager Agent. Must contain at least
            ``task_query``. ``aggregated_research_context`` will be populated.

        Returns
        -------
        SharedManagerState
            Updated contract with ``aggregated_research_context`` filled.

        Raises
        ------
        ValueError
            If ``task_query`` is missing or empty in shared_state.
        RuntimeError
            If the internal graph fails with an unrecoverable error.
        """
        task_query = shared_state.get("task_query", "").strip()
        if not task_query:
            raise ValueError(
                "SharedManagerState.task_query must be a non-empty string."
            )

        logger.info("ResearchAgent.run() | query='%s'", task_query)

        # ── Instantiate isolated internal state ──────────────────
        internal_state: ResearchAgentState = {
            "messages":           [],
            "context_chunks":     [],
            "called_tool_signatures": [],
            "loop_counter":       0,
            "validation_feedback": "",
            "is_complete":        False,
            "task_query":         shared_state["task_query"],
            "manager_directives": dict(shared_state.get("manager_directives", {})),
        }

        # ── Execute the compiled graph ───────────────────────────
        try:
            final_state: ResearchAgentState = await self._graph.ainvoke(
                internal_state,
                config={"recursion_limit": (self._max_loops + 1) * 3},
            )
        except Exception as exc:
            logger.exception("ResearchAgent graph execution failed: %s", exc)
            raise RuntimeError(f"ResearchAgent internal graph failed: {exc}") from exc

        # ── Map results back into shared contract ─────────────────
        # ADDITIVE synthesis step: compress the raw chunks into one dense
        # executive summary for the Manager to skim, WITHOUT removing the
        # raw chunks themselves — exact numbers (e.g. an analyst's price-
        # target percentage, a 10-Q dollar figure) must stay traceable for
        # downstream numeric faithfulness validation. This call happens
        # entirely AFTER the internal Brain/Checker/Executor loop above
        # has finished — the loop's own logic is completely untouched by
        # this step. A synthesis failure never loses data or raises: see
        # synthesize_research_context's own docstring for why it returns
        # None on failure instead of raising.
        #
        # context_synthesizer now orders/labels chunks by source priority
        # (SEC filings first for financial fundamentals, see its module
        # docstring) IF each chunk is tagged {"source": ..., "text": ...}.
        # Every chunk here already carries a "[TOOL: <name>]" tag (added by
        # _format_tool_result for human readability) — reuse it instead of
        # threading a parallel tagged-chunks list through ResearchAgentState.
        tagged_chunks = [
            self._tag_chunk_with_source(c) for c in final_state["context_chunks"]
        ]
        synthesized_summary = await synthesize_research_context(
            chunks=tagged_chunks,
            task_query=task_query,
            llm_client=self._llm,
            model=self._model,
        )

        existing = shared_state.get("aggregated_research_context") or []
        new_chunks = final_state["context_chunks"] + (
            [synthesized_summary] if synthesized_summary else []
        )
        updated_shared: SharedManagerState = {
            **shared_state,
            "aggregated_research_context": existing + new_chunks,
        }

        logger.info(
            "ResearchAgent.run() complete | chunks=%d | synthesized_summary=%s | loops=%d",
            len(final_state["context_chunks"]),
            bool(synthesized_summary),
            final_state["loop_counter"],
        )
        return updated_shared

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Brain (Planner)
    # ══════════════════════════════════════════════════════════════

    @traceable(name="research.brain", run_type="llm")
    async def _brain_node(
        self, state: ResearchAgentState
    ) -> dict[str, Any]:
        """
        Planner node — generates a structured JSON action plan for the Executor.

        Reads the task_query, all previous messages, and any validation_feedback
        from the Checker. Calls Claude to produce a JSON plan specifying which
        MCP tools to invoke and with what arguments.

        State reads
        -----------
        state["task_query"]
        state["manager_directives"]
        state["messages"]          — conversation history
        state["validation_feedback"] — Checker's critique (empty on first pass)
        state["loop_counter"]      — current iteration number

        State mutations returned
        ------------------------
        messages : list[dict]
            Appends the new user prompt and the Brain's assistant response.

        Parameters
        ----------
        state : ResearchAgentState
            Current graph state snapshot.

        Returns
        -------
        dict[str, Any]
            Partial state update: {"messages": [user_msg, assistant_msg]}
        """
        task_query   = state["task_query"]
        directives   = state["manager_directives"]
        feedback     = state["validation_feedback"]
        loop_num     = state["loop_counter"]

        logger.info("[Brain] iteration=%d | feedback='%s'", loop_num, feedback[:80] if feedback else "none")

        # Build the user prompt for this planning cycle
        directive_str = json.dumps(directives, indent=2) if directives else "None"
        feedback_block = (
            f"\n\nVALIDATOR FEEDBACK (address this in your plan):\n{feedback}"
            if feedback else ""
        )

        user_content = (
            f"RESEARCH TASK:\n{task_query}\n\n"
            f"MANAGER DIRECTIVES:\n{directive_str}"
            f"{feedback_block}\n\n"
            f"ITERATION: {loop_num + 1} of {self._max_loops}\n\n"
            "Produce the JSON action plan now."
        )

        # Build message history for Claude
        history = state["messages"]
        messages_for_api = history + [{"role": "user", "content": user_content}]

        response = await asyncio.to_thread(
            self._llm.messages.create,
            model=self._model,
            max_tokens=1024,
            temperature=0,
            system=BRAIN_SYSTEM_PROMPT,
            messages=messages_for_api,
        )
        plan_text = response.content[0].text.strip()
        logger.debug("[Brain] plan output: %s", plan_text[:300])

        _publish_progress(
            session_from_directives(directives), "agent_brain", agent="research",
            message=f"Research Agent: planning research (iteration {loop_num + 1})",
            detail={"loop": loop_num + 1},
        )

        return {
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": plan_text},
            ]
        }

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Executor (Tool Caller)
    # ══════════════════════════════════════════════════════════════

    @traceable(name="research.executor", run_type="tool")
    async def _executor_node(
        self, state: ResearchAgentState
    ) -> dict[str, Any]:
        """
        Execution node — parses the Brain's JSON plan and calls MCP tools.

        Connects to research_server.py via stdio MCP transport, discovers
        available tools, then executes the actions specified in the Brain's
        most recent plan. Each successful tool result is appended to
        context_chunks as a formatted text string.

        Fails gracefully: a tool failure is logged as a warning and appended
        as an error marker in context_chunks; it does NOT crash the loop.

        State reads
        -----------
        state["messages"][-1]["content"]  — Brain's JSON plan (last assistant msg)

        State mutations returned
        ------------------------
        loop_counter  : int     — incremented by 1
        context_chunks: list[str] — new text chunks appended

        Parameters
        ----------
        state : ResearchAgentState
            Current graph state snapshot.

        Returns
        -------
        dict[str, Any]
            Partial state update: {"loop_counter": n+1, "context_chunks": [...]}
        """
        new_counter = state["loop_counter"] + 1
        logger.info("[Executor] loop_counter → %d", new_counter)
        session_id = session_from_directives(state["manager_directives"])

        # ── Parse Brain's JSON plan ──────────────────────────────
        plan_text = state["messages"][-1]["content"]
        actions   = self._parse_plan(plan_text)

        if not actions:
            logger.warning("[Executor] Brain produced no parseable actions.")
            return {
                "loop_counter":  new_counter,
                "context_chunks": ["[EXECUTOR WARNING] Brain produced no valid action plan."],
                "called_tool_signatures": [],
            }

        # ── Execute tools via MCP ────────────────────────────────
        new_chunks: list[str] = []
        new_signatures: list[str] = []
        seen_signatures: set[str] = set(state.get("called_tool_signatures", []))
        sec_filing_coverage = build_sec_filing_coverage(seen_signatures)

        try:
            from contextlib import AsyncExitStack
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(
                    stdio_client(self._mcp_params)
                )
                session: ClientSession = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                for action in actions:
                    tool_name = action.get("tool", "")
                    arguments = action.get("arguments", {})

                    if not tool_name:
                        logger.warning("[Executor] Skipping action with no tool name.")
                        continue

                    arguments = ensure_ticker_argument(
                        tool_name, arguments,
                        state["manager_directives"],
                    )

                    query_hint = arguments.get("query") or arguments.get("entity") or arguments.get("ticker") or "N/A"
                    signature = tool_call_signature(tool_name, arguments)

                    if signature in seen_signatures:
                        logger.info(
                            "[Executor] SKIPPING exact-duplicate call: tool=%s args=%s",
                            tool_name, arguments,
                        )
                        new_chunks.append(format_skip_notice(
                            tool_name, query_hint, "duplicate",
                            "This exact call (same tool, same arguments) was already made earlier "
                            "in this session -- its result is already present in the context above. "
                            "No new data was fetched. If you need more, change the arguments "
                            "(different sections/mode/query), don't repeat the same call."
                        ))
                        continue

                    doc_key = sec_filing_doc_key(arguments) if tool_name == "sec_edgar_filing" else None
                    requested_sections = sec_filing_sections(arguments) if doc_key else set()
                    already_covered = sec_filing_coverage.get(doc_key, set()) if doc_key else set()
                    if doc_key and requested_sections and requested_sections <= already_covered:
                        logger.info(
                            "[Executor] SKIPPING redundant sec_edgar_filing call: sections=%s "
                            "already covered=%s for doc_key=%s",
                            requested_sections, already_covered, doc_key,
                        )
                        new_chunks.append(format_skip_notice(
                            "sec_edgar_filing", query_hint, "redundant subset",
                            f"Section(s) {sorted(requested_sections)} for this filing "
                            f"(ticker={arguments.get('ticker')}, form_type={arguments.get('form_type', '10-K')}, "
                            f"mode={arguments.get('mode', 'preview')}) were already fetched in an earlier "
                            f"call -- see that sec_edgar_filing result above. No new data was fetched. "
                            f"Request a DIFFERENT section, or use mode='full'/query=... on this section "
                            f"for more depth instead of re-requesting it."
                        ))
                        continue

                    logger.info("[Executor] calling tool='%s' args=%s", tool_name, arguments)
                    _publish_progress(
                        session_id, "agent_tool_call", agent="research",
                        message=f"Research Agent: calling tool '{tool_name}'...",
                        detail={"tool": tool_name},
                    )

                    try:
                        if sentry_enabled():
                            import sentry_sdk
                            sentry_sdk.add_breadcrumb(
                                category="mcp.research",
                                message=f"Calling tool: {tool_name}",
                                data={"tool_name": tool_name, "arguments": str(arguments)[:200]},
                                level="info",
                            )
                        result = await session.call_tool(tool_name, arguments)
                        raw_text = (
                            result.content[0].text
                            if result.content and result.content[0].type == "text"
                            else ""
                        )
                        # Format as a labelled chunk for readability in LLM prompts
                        chunk = self._format_tool_result(tool_name, arguments, raw_text)
                        new_chunks.append(chunk)
                        new_signatures.append(signature)
                        seen_signatures.add(signature)
                        if doc_key:
                            sec_filing_coverage.setdefault(doc_key, set()).update(requested_sections)
                        logger.info(
                            "[Executor] tool='%s' → %d chars", tool_name, len(raw_text)
                        )
                        _publish_progress(
                            session_id, "agent_tool_result", agent="research",
                            message=f"Research Agent: tool '{tool_name}' succeeded",
                            detail={"tool": tool_name, "outcome": "success", "chars": len(raw_text)},
                        )

                    except Exception as tool_exc:
                        err_msg = (
                            f"[TOOL ERROR] {tool_name} failed: {tool_exc}"
                        )
                        logger.warning(err_msg)
                        if sentry_enabled():
                            import sentry_sdk
                            with sentry_sdk.push_scope() as scope:
                                scope.set_tag("tool", tool_name)
                                scope.set_tag("component", "mcp.research")
                                sentry_sdk.capture_exception(tool_exc)
                        new_chunks.append(err_msg)
                        _publish_progress(
                            session_id, "agent_tool_result", agent="research",
                            message=f"Research Agent: tool '{tool_name}' failed",
                            detail={"tool": tool_name, "outcome": "error", "error": str(tool_exc)},
                        )

        except Exception as mcp_exc:
            err_msg = f"[MCP CONNECTION ERROR] Could not connect to research_server: {mcp_exc}"
            logger.error(err_msg)
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "mcp_connection")
                    scope.set_tag("server", "research-agent-mcp")
                    sentry_sdk.capture_exception(mcp_exc)
            new_chunks.append(err_msg)

        return {
            "loop_counter":   new_counter,
            "context_chunks": new_chunks,
            "called_tool_signatures": new_signatures,
        }

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Checker (Validator / Critic)
    # ══════════════════════════════════════════════════════════════

    @traceable(name="research.checker", run_type="llm")
    async def _checker_node(
        self, state: ResearchAgentState
    ) -> dict[str, Any]:
        """
        Critic node — audits gathered context_chunks for completeness.

        Calls Claude with the accumulated context chunks and the original
        task_query. Outputs a JSON verdict:
            - is_complete=True  → enough data; graph exits to END.
            - is_complete=False → insufficient; sets validation_feedback
                                  for the Brain to act on in the next loop.

        State reads
        -----------
        state["context_chunks"]                       — all gathered text so far
        state["task_query"]                           — original research question

        State mutations returned
        ------------------------
        is_complete         : bool
        validation_feedback : str

        Parameters
        ----------
        state : ResearchAgentState
            Current graph state snapshot.

        Returns
        -------
        dict[str, Any]
            Partial state update: {"is_complete": bool, "validation_feedback": str}
        """
        task_query    = state["task_query"]
        all_chunks    = state["context_chunks"]
        loop_counter  = state["loop_counter"]

        logger.info(
            "[Checker] auditing %d chunks (loop=%d)", len(all_chunks), loop_counter
        )

        if not all_chunks:
            logger.warning("[Checker] No context chunks to audit.")
            return {
                "is_complete":        False,
                "validation_feedback": "No data was retrieved. Retry with different tools or broader queries.",
            }

        # Give each chunk a small, fixed budget instead of truncating the
        # whole concatenated string. The old approach (truncate the joined
        # string to 12,000 chars) let a single large early chunk — e.g. a
        # sec_edgar_filing chunk, routinely 20,000-50,000+ chars — consume
        # the ENTIRE budget, silently hiding every tool call made in later
        # iterations from the Checker's view. This caused confirmed
        # redundant re-fetching: the Checker, unable to see that
        # news_search / rag_hybrid_query had already run (their results
        # existed but were past the cutoff), told the Brain "insufficient,
        # get more data" — and the Brain re-issued a near-identical query,
        # returning near-identical results, in a later loop iteration.
        # A per-chunk cap ensures the Checker always sees WHICH tools have
        # already been called, even if it can't see each one's full text.
        per_chunk_budget = max(800, 12_000 // max(len(all_chunks), 1))

        def _budget_chunk(c: str) -> str:
            if len(c) <= per_chunk_budget:
                return c
            # Mark truncated chunks explicitly. Without this, the Checker
            # LLM has no way to distinguish "this tool genuinely returned
            # little data" from "this tool returned a lot of data and we
            # cut it off" — the former might legitimately mean "insufficient,
            # try a different tool", the latter should NOT trigger that same
            # verdict/re-fetch. A prior version of this function truncated
            # the whole joined string with a trailing marker; this per-chunk
            # rewrite (see comment above) preserves per-tool visibility but
            # had dropped the marker — restored here, now per-chunk instead
            # of once for the whole blob.
            return c[:per_chunk_budget] + "\n[truncated for audit]"

        combined = "\n\n---\n\n".join(_budget_chunk(c) for c in all_chunks)

        user_content = (
            f"ORIGINAL RESEARCH QUERY:\n{task_query}\n\n"
            f"GATHERED CONTEXT ({len(all_chunks)} chunks, loop {loop_counter}):\n\n"
            f"{combined}\n\n"
            "Audit the context and return your JSON verdict."
        )

        response = await asyncio.to_thread(
            self._llm.messages.create,
            model=self._model,
            max_tokens=512,
            temperature=0,
            system=CHECKER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        verdict_text = response.content[0].text.strip()

        try:
            verdict = json.loads(
                verdict_text.replace("```json", "").replace("```", "").strip()
            )
        except json.JSONDecodeError:
            logger.warning("[Checker] Could not parse verdict JSON: %s", verdict_text[:200])
            verdict = {"is_complete": False, "feedback": "Checker JSON parse error; retry."}

        is_complete = bool(verdict.get("is_complete", False))
        feedback    = str(verdict.get("feedback", ""))

        logger.info(
            "[Checker] is_complete=%s score=%s missing='%s'",
            is_complete,
            verdict.get("score", "N/A"),
            verdict.get("missing", "")[:80],
        )
        _publish_progress(
            session_from_directives(state["manager_directives"]), "agent_checker", agent="research",
            message=(
                f"Research Agent: checking data quality — "
                f"{'sufficient ✓' if is_complete else 'needs more data'}"
            ),
            detail={"is_complete": is_complete, "score": verdict.get("score")},
        )

        return {
            "is_complete":        is_complete,
            "validation_feedback": "" if is_complete else feedback,
        }

    # ══════════════════════════════════════════════════════════════
    # Conditional routing edge
    # ══════════════════════════════════════════════════════════════

    def _should_continue(
        self, state: ResearchAgentState
    ) -> Literal["brain", "__end__"]:
        """
        Conditional routing edge called after every Checker evaluation.

        Decision logic (evaluated in priority order):
          1. loop_counter >= max_loops  → force exit to END (guardrail).
          2. is_complete == True        → normal exit to END.
          3. Otherwise                 → loop back to Brain.

        This method shields the infrastructure from infinite tool loops
        and severe token budget inflation by enforcing a hard iteration cap.

        Parameters
        ----------
        state : ResearchAgentState
            Current graph state snapshot after the Checker node.

        Returns
        -------
        Literal["brain", "__end__"]
            "brain"    → continue the loop (route to _brain_node).
            "__end__"  → terminate the graph (route to END).
        """
        loop_counter = state["loop_counter"]
        is_complete  = state["is_complete"]
        max_loops    = state["manager_directives"].get("max_loops", self._max_loops)

        # Guardrail: hard cap on iterations
        if loop_counter >= max_loops:
            logger.warning(
                "[Router] loop_counter=%d >= max_loops=%d → forcing END.",
                loop_counter, max_loops,
            )
            return "__end__"

        # Normal completion
        if is_complete:
            logger.info("[Router] is_complete=True → routing to END.")
            return "__end__"

        # Continue loop
        logger.info(
            "[Router] is_complete=False, loop=%d/%d → routing to Brain.",
            loop_counter, max_loops,
        )
        return "brain"

    # ══════════════════════════════════════════════════════════════
    # Graph builder
    # ══════════════════════════════════════════════════════════════

    def _build_graph(self) -> StateGraph:
        """
        Compile and return the internal LangGraph state machine.

        Graph topology:
            START → brain → executor → checker → (conditional) → brain | END

        All nodes are async instance methods. The conditional edge
        calls _should_continue() after each Checker evaluation.

        Returns
        -------
        StateGraph
            Compiled LangGraph application ready for ainvoke().
        """
        builder = StateGraph(ResearchAgentState)

        # ── Register nodes ───────────────────────────────────────
        builder.add_node("brain",    self._brain_node)
        builder.add_node("executor", self._executor_node)
        builder.add_node("checker",  self._checker_node)

        # ── Wire edges ───────────────────────────────────────────
        builder.set_entry_point("brain")
        builder.add_edge("brain",    "executor")
        builder.add_edge("executor", "checker")

        # Conditional edge: checker → brain | END
        builder.add_conditional_edges(
            "checker",
            self._should_continue,
            {
                "brain":    "brain",
                "__end__":  END,
            },
        )

        return builder.compile()

    # ══════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════

    def _parse_plan(self, plan_text: str) -> list[dict[str, Any]]:
        """
        Parse the Brain's JSON output into a list of action dicts.

        Strips markdown fences if present and extracts the "actions" array.
        Returns an empty list (instead of raising) on any parse failure so
        the Executor can handle it gracefully without crashing the loop.

        Parameters
        ----------
        plan_text : str
            Raw JSON string output from the Brain node.

        Returns
        -------
        list[dict[str, Any]]
            List of action dicts: [{"tool": str, "arguments": dict}, ...]
            Returns [] on parse failure.
        """
        try:
            clean = plan_text.replace("```json", "").replace("```", "").strip()
            plan  = json.loads(clean)
            actions = plan.get("actions", [])
            if not isinstance(actions, list):
                logger.warning("[Parser] 'actions' is not a list: %s", type(actions))
                return []
            return actions
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("[Parser] Failed to parse Brain plan: %s | text=%s", exc, plan_text[:200])
            return []

    @staticmethod
    def _format_tool_result(
        tool_name: str,
        arguments: dict[str, Any],
        raw_text: str,
    ) -> str:
        """
        Format a raw MCP tool result into a labelled context chunk string.

        The formatted chunk is human-readable and LLM-friendly, including
        the tool name, key argument for identification, and the result body.

        Parameters
        ----------
        tool_name : str
            Name of the MCP tool that was called.
        arguments : dict[str, Any]
            Arguments that were passed to the tool.
        raw_text : str
            Raw JSON string returned by the MCP tool.

        Returns
        -------
        str
            Formatted multi-line context chunk ready for LLM injection.
        """
        query_hint = arguments.get("query") or arguments.get("entity") or arguments.get("ticker") or "N/A"
        separator  = "─" * 60
        return (
            f"{separator}\n"
            f"[TOOL: {tool_name}] | [QUERY: {query_hint}]\n"
            f"{separator}\n"
            f"{raw_text}\n"
        )

    _TOOL_TAG_RE = re.compile(r"\[TOOL:\s*([^\]]+)\]")

    @classmethod
    def _tag_chunk_with_source(cls, chunk: str) -> dict:
        """
        Recover the tool name _format_tool_result already embedded as
        "[TOOL: <name>]" and reshape the chunk into the
        {"source": ..., "text": ...} form context_synthesizer.py's
        source-priority ordering expects. Falls back to source=None
        (synthesizer treats that as "unknown priority", not "lowest") if
        the tag is missing or the chunk came from somewhere else, e.g. the
        "[TOOL ERROR] ..." chunks built inline in _executor_node rather
        than through _format_tool_result.
        """
        m = cls._TOOL_TAG_RE.search(chunk)
        return {"source": m.group(1).strip() if m else None, "text": chunk}