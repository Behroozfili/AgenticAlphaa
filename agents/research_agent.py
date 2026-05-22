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

Available MCP tools (call by exact name):
  - tavily_search         : Real-time web search (best for breaking news, macro events)
  - news_search           : NewsAPI financial articles (best for recent press coverage)
  - sec_edgar_search      : SEC EDGAR full-text filing search
  - sec_edgar_filing      : Fetch and parse a specific SEC 10-K / 10-Q filing
  - rag_vector_search     : Semantic search over pre-ingested knowledge base
  - rag_graph_traverse    : Neo4j entity-relationship traversal (competitors, supply chain)
  - rag_hybrid_query      : Combined vector + graph search (best for complex queries)

Output format (strict JSON, no markdown fences):
{
  "reasoning": "<one sentence: why these tools in this order>",
  "actions": [
    {
      "tool": "<exact tool name>",
      "arguments": { <key-value pairs matching the tool's inputSchema> }
    }
  ]
}

Rules:
- Output ONLY valid JSON. No preamble, no explanation outside JSON.
- Choose 1–3 tools per loop. Do not repeat a tool with identical arguments.
- Incorporate validator feedback when provided to target missing information.
- Prefer rag_hybrid_query for complex multi-faceted queries.
- Always include "query" in arguments for search tools.
"""

# ── Checker node system prompt ───────────────────────────────────
_CHECKER_SYSTEM_PROMPT = """\
You are the Research Validator for Alpha-Agent Node.

Your role: Audit the gathered context chunks and decide if they are sufficient
to answer the original research query completely and accurately.

Completeness criteria:
  1. Recent data: at least one source from the last 30 days.
  2. Factual depth: numbers, dates, or named entities relevant to the query.
  3. Multi-source coverage: results from at least 2 different tool types.
  4. No hallucination risk: findings are grounded in retrieved text, not assumed.

Output format (strict JSON, no markdown fences):
{
  "is_complete": true | false,
  "score": <int 0-100>,
  "missing": "<what specific information is still needed, or 'nothing' if complete>",
  "feedback": "<actionable instruction for the Brain to improve the next search, or '' if complete>"
}

Rules:
- Output ONLY valid JSON.
- Be strict: partial data or vague summaries are NOT sufficient.
- If is_complete is true, feedback must be an empty string "".
"""


# ══════════════════════════════════════════════════════════════════
# ResearchAgent
# ══════════════════════════════════════════════════════════════════

class ResearchAgent:
    """
    LangGraph-powered Research Agent for the Alpha-Agent Node platform.

    Encapsulates:
      - Anthropic Claude-3-5-Sonnet as the planning and validation LLM.
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
        Claude model identifier. Default: "claude-sonnet-4-20250514".
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
        model: str = "claude-sonnet-4-20250514",
        max_loops: int = _DEFAULT_MAX_LOOPS,
        mcp_server_params: StdioServerParameters | None = None,
    ) -> None:
        # ── LLM client ───────────────────────────────────────────
        self._llm = anthropic.Anthropic(
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
        existing = shared_state.get("aggregated_research_context") or []
        updated_shared: SharedManagerState = {
            **shared_state,
            "aggregated_research_context": existing + final_state["context_chunks"],
        }

        logger.info(
            "ResearchAgent.run() complete | chunks=%d | loops=%d",
            len(final_state["context_chunks"]),
            final_state["loop_counter"],
        )
        return updated_shared

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Brain (Planner)
    # ══════════════════════════════════════════════════════════════

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

        response = self._llm.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_BRAIN_SYSTEM_PROMPT,
            messages=messages_for_api,
        )
        plan_text = response.content[0].text.strip()
        logger.debug("[Brain] plan output: %s", plan_text[:300])

        return {
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": plan_text},
            ]
        }

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Executor (Tool Caller)
    # ══════════════════════════════════════════════════════════════

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

                    logger.info("[Executor] calling tool='%s' args=%s", tool_name, arguments)

                    try:
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

                    except Exception as tool_exc:
                        err_msg = (
                            f"[TOOL ERROR] {tool_name} failed: {tool_exc}"
                        )
                        logger.warning(err_msg)
                        new_chunks.append(err_msg)

        except Exception as mcp_exc:
            err_msg = f"[MCP CONNECTION ERROR] Could not connect to research_server: {mcp_exc}"
            logger.error(err_msg)
            new_chunks.append(err_msg)

        return {
            "loop_counter":   new_counter,
            "context_chunks": new_chunks,
        }

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Checker (Validator / Critic)
    # ══════════════════════════════════════════════════════════════

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

        # Combine chunks (truncate to avoid token overflow)
        combined = "\n\n---\n\n".join(all_chunks)
        if len(combined) > 12_000:
            combined = combined[:12_000] + "\n...[truncated for audit]"

        user_content = (
            f"ORIGINAL RESEARCH QUERY:\n{task_query}\n\n"
            f"GATHERED CONTEXT ({len(all_chunks)} chunks, loop {loop_counter}):\n\n"
            f"{combined}\n\n"
            "Audit the context and return your JSON verdict."
        )

        response = self._llm.messages.create(
            model=self._model,
            max_tokens=512,
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
