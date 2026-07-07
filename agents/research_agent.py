"""
agents/research_agent.py — ResearchAgent
==========================================
Full LangGraph implementation of the Research Agent for Alpha-Agent Node.

Architecture
─────────────
  SharedManagerState (in)
        │
        ▼
  ResearchAgent.run()
        │  instantiates ResearchAgentState
        │
        ▼
  ┌──────────────────────────────────┐
  │     LangGraph Internal Loop      │
  │                                  │
  │  [brain_node]                    │
  │      │ structured JSON plan      │
  │      ▼                           │
  │  [executor_node]                 │
  │      │ MCP tool calls            │
  │      ▼                           │
  │  [checker_node]                  │
  │      │                           │
  │      ├── is_complete=True ──────►END
  │      │                           │
  │      └── loop_counter >= max ───►END (force)
  │                                  │
  │      └── needs_more ────────────►[brain_node] (loop)
  └──────────────────────────────────┘
        │
        ▼
  SharedManagerState (out) — aggregated_research_context populated
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Literal

import anthropic
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.state import ResearchAgentState, SharedManagerState
from dotenv import load_dotenv
from langsmith import traceable
from core.observability import sentry_enabled
from core.progress_bus import publish as _publish_progress, session_from_shared
from tools.research_tools.context_synthesizer import synthesize_research_context

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

# ── Brain node system prompt ────────────────────────────────────
_BRAIN_SYSTEM_PROMPT = """\
You are the Research Brain for Alpha-Agent Node, a production-grade financial intelligence system.

Your role: Given a research task and optional validator feedback, produce a precise JSON action plan
that directs the Executor which MCP tools to call next.

Available MCP tools and their EXACT argument schema (call by exact name,
use EXACTLY these argument key names — they differ between tools):

  - tavily_search
      arguments: {"query": "<search text>"}

  - news_search
      arguments: {"query": "<search text>"}

  - sec_edgar_search
      arguments: {"query": "<search text>", "ticker": "<TICKER>", "form_type": "<optional 10-K/10-Q/8-K>"}
      "ticker" is REQUIRED whenever a specific company is being researched —
      omitting it runs an unscoped full-text search across ALL filers and
      returns irrelevant results.

  - sec_edgar_filing
      arguments: {"ticker": "<TICKER>", "form_type": "<10-K or 10-Q>", "sections": [<list>]}
      "ticker" is REQUIRED — this call fails validation without it.
      "sections" accepts: "business", "risk_factors", "mda", "financial_statements",
      or "all". Can be a single string or a list. Defaults to "all" if omitted.
      For a comprehensive investment analysis task, explicitly request
      ["mda", "risk_factors"] together — risk_factors covers competitive
      threats, supply chain dependencies, and geopolitical/regulatory
      exposure (e.g. export restrictions) that MD&A alone does not, and is
      needed for a complete risk picture. Do not restrict to a single
      section like ["financial_statements"] alone for this kind of task.

  - rag_vector_search
      arguments: {"query": "<search text>", "ticker_filter": "<TICKER>"}
      "ticker_filter" is REQUIRED whenever the task concerns a specific
      company — omitting it searches the ENTIRE vector store unfiltered
      and can return chunks about unrelated companies.

  - rag_graph_traverse
      arguments: {"entity": "<TICKER>"}
      "entity" is REQUIRED — without it the tool cannot run at all.

  - rag_hybrid_query
      arguments: {"query": "<search text>", "entity": "<TICKER>"}
      "entity" is REQUIRED whenever a specific company is being researched.
      Do NOT rely on the ticker symbol appearing inside "query" — this tool
      only has a weak, best-effort fallback that looks for an ALL-CAPS
      token in the query text (and fails silently whenever the query is
      phrased using the company's name instead of its ticker symbol, e.g.
      "Microsoft AI infrastructure spending" contains no extractable
      ticker). Passing "entity" explicitly is the only reliable path —
      never omit it.

THE TICKER: the target company's ticker symbol is provided to you in
MANAGER DIRECTIVES as "ticker" (e.g. "MSFT"). Whenever it is present,
you MUST include it as the relevant argument (see schemas above — the
key name differs per tool: "ticker", "ticker_filter", or "entity") on
EVERY call to a ticker-scoped tool. This is not optional and does not
depend on whether the ticker symbol happens to appear in your own
"query" text.

QUERY TEXT FORMAT — this differs by tool and matters a lot:
  - news_search and sec_edgar_search are KEYWORD/boolean full-text search
    engines, not semantic search. They match literal words, not meaning.
    Use 2-5 concise keywords, e.g. "AAPL earnings Q2" or "Apple analyst
    rating upgrade" — NEVER a full sentence like "Apple stock price
    movement analyst ratings sentiment". A long natural-language query
    against a keyword engine routinely returns zero or near-zero results,
    even when the topic is well-covered — this has been observed in
    production and is not a data availability problem, it's a query
    phrasing problem.
  - tavily_search, rag_vector_search, rag_graph_traverse, and
    rag_hybrid_query use semantic/embedding-based search and handle
    natural-language phrasing fine — you don't need to compress these
    into keywords.

Output format (strict JSON, no markdown fences):
{
  "reasoning": "<one sentence: why these tools in this order>",
  "actions": [
    {
      "tool": "<exact tool name>",
      "arguments": { <key-value pairs matching the exact schema above> }
    }
  ]
}

Rules:
- Output ONLY valid JSON. No preamble, no explanation outside JSON.
- Choose 1–3 tools per loop. Do not repeat a tool with identical arguments.
- Incorporate validator feedback when provided to target missing information.
- Prefer rag_hybrid_query for complex multi-faceted queries.
- Always include "query" in arguments for search tools.
- For news_search and sec_edgar_search, keep "query" to 2-5 keywords
  (see QUERY TEXT FORMAT above) — never a full sentence.
- Always include the ticker argument (see per-tool schema above) whenever
  MANAGER DIRECTIVES provides one and the tool is ticker-scoped.
"""

# ── Checker node system prompt ───────────────────────────────────
_CHECKER_SYSTEM_PROMPT = """\
You are the Research Validator for Alpha-Agent Node.

Your role: Audit the gathered context chunks and decide if they are sufficient
to answer the original research query as completely as the available authoritative
sources allow.

Completeness criteria:
  1. Recent data: at least one source from the last 30 days (or the most recent
     periodic filing available, e.g. the latest 10-Q/10-K).
  2. Factual depth: numbers, dates, or named entities relevant to the query.
  3. Multi-source coverage: results from at least 2 different tool types.
  4. No hallucination risk: findings are grounded in retrieved text, not assumed.

CRITICAL — negative findings are valid findings:
  - If an authoritative primary source has already been retrieved (e.g. the
    relevant 10-Q/10-K sections are present in the context) and it simply does
    NOT separately disclose the specific figure the query asks for (e.g. a
    company reports R&D as a single line and does not break out "AI capex"),
    then the correct, COMPLETE answer is that the source does not disclose it.
    Mark is_complete=true and state the absence as the finding. Do NOT keep
    demanding a number that the source does not contain.
  - Do NOT request a section or document that is already present in the gathered
    context. Check what was retrieved before asking for more.
  - Distinguish "not yet retrieved" (→ keep searching) from "retrieved but the
    data does not exist in the source" (→ complete). Only the former justifies
    is_complete=false.

Output format (strict JSON, no markdown fences):
{
  "is_complete": true | false,
  "score": <int 0-100>,
  "missing": "<specific information still genuinely missing AND obtainable, or 'nothing' if complete>",
  "feedback": "<actionable instruction for the Brain to retrieve genuinely-missing AND obtainable data, or '' if complete>"
}

Rules:
- Output ONLY valid JSON.
- Be rigorous, but do not penalise the agent for data that authoritative sources
  genuinely do not contain — that absence is itself the answer.
- If is_complete is true, feedback must be an empty string "".
"""


# Maps each ticker-scoped MCP tool to the exact argument key it expects for
# the ticker/entity. Names deliberately differ between tools (see
# tools/research_tools/research_server.py / rag/hybrid_rag.py):
#   rag_vector_search  -> ticker_filter
#   rag_graph_traverse -> entity
#   rag_hybrid_query   -> entity
#   sec_edgar_search   -> ticker
#   sec_edgar_filing   -> ticker
_TICKER_ARG_BY_TOOL: dict[str, str] = {
    "rag_vector_search":  "ticker_filter",
    "rag_graph_traverse": "entity",
    "rag_hybrid_query":   "entity",
    "sec_edgar_search":   "ticker",
    "sec_edgar_filing":   "ticker",
}


def _ensure_ticker_argument(
    tool_name: str,
    arguments: dict[str, Any],
    directives: dict[str, Any],
) -> dict[str, Any]:
    """
    Defensive safety net, independent of LLM Brain behaviour: if this tool
    is ticker-scoped and the Brain's plan omitted the ticker argument (or
    supplied it under the wrong key), inject it from manager_directives.

    Root cause this guards against: rag_hybrid_query's built-in fallback
    (_extract_ticker_from_query in rag/hybrid_rag.py) only works when the
    literal ticker symbol happens to appear inside the free-text query
    (e.g. "Apple AAPL earnings" -> "AAPL" extracted by luck). Queries
    phrased using the company name instead ("Microsoft AI infrastructure
    spending") have no extractable ticker, silently degrading
    rag_hybrid_query to vector-only mode AND removing the ticker filter
    from the underlying vector search entirely — letting chunks from
    unrelated tickers leak into the results. The system prompt now
    instructs the Brain to always pass the ticker explicitly, but a prompt
    instruction can be missed; this function makes correctness NOT depend
    on the LLM remembering.

    A ticker already present in `arguments` (whatever the LLM supplied) is
    always preserved — this only fills in a MISSING value, never overrides
    an explicit one.
    """
    arg_key = _TICKER_ARG_BY_TOOL.get(tool_name)
    if not arg_key:
        return arguments  # not a ticker-scoped tool — nothing to do

    if arguments.get(arg_key):
        return arguments  # Brain already supplied it

    ticker = directives.get("ticker")
    if not ticker:
        return arguments  # nothing to inject (no ticker known for this task)

    patched = dict(arguments)
    patched[arg_key] = ticker
    logger.info(
        "[Executor] Injected missing ticker argument for '%s': %s='%s'",
        tool_name, arg_key, ticker,
    )
    return patched


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
        # ── LLM client ───────────────────────────────────────────
        # Accept an injected client so tests can pass a mock without
        # making real API calls or needing ANTHROPIC_API_KEY set.
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
            "loop_counter":       0,
            "validation_feedback": "",
            "is_complete":        False,
            "shared_manager_ref": shared_state,
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
        synthesized_summary = await synthesize_research_context(
            chunks=final_state["context_chunks"],
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
        state["shared_manager_ref"]["task_query"]
        state["shared_manager_ref"]["manager_directives"]
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
        ref          = state["shared_manager_ref"]
        task_query   = ref["task_query"]
        directives   = ref.get("manager_directives", {})
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
            system=_BRAIN_SYSTEM_PROMPT,
            messages=messages_for_api,
        )
        plan_text = response.content[0].text.strip()
        logger.debug("[Brain] plan output: %s", plan_text[:300])

        _publish_progress(
            session_from_shared(ref), "agent_brain", agent="research",
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
        session_id = session_from_shared(state["shared_manager_ref"])

        # ── Parse Brain's JSON plan ──────────────────────────────
        plan_text = state["messages"][-1]["content"]
        actions   = self._parse_plan(plan_text)

        if not actions:
            logger.warning("[Executor] Brain produced no parseable actions.")
            return {
                "loop_counter":  new_counter,
                "context_chunks": ["[EXECUTOR WARNING] Brain produced no valid action plan."],
            }

        # ── Execute tools via MCP ────────────────────────────────
        new_chunks: list[str] = []

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

                    arguments = _ensure_ticker_argument(
                        tool_name, arguments,
                        state["shared_manager_ref"].get("manager_directives", {}),
                    )

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
        state["shared_manager_ref"]["task_query"]     — original research question

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
        task_query    = state["shared_manager_ref"]["task_query"]
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
            system=_CHECKER_SYSTEM_PROMPT,
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
            session_from_shared(state["shared_manager_ref"]), "agent_checker", agent="research",
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
        max_loops    = state["shared_manager_ref"].get(
            "manager_directives", {}
        ).get("max_loops", self._max_loops)

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