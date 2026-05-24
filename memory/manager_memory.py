"""
memory/manager_memory.py — ManagerMemory
=========================================
Two-level cognitive retention system for the ManagerAgent.

Architecture
------------
The ManagerMemory is split into two isolated layers that mirror how a
human analyst would think:

  ┌─────────────────────────────────────────────────────────┐
  │                     ManagerMemory                       │
  │                                                         │
  │  SHORT-TERM (Ephemeral / Session-Scoped)                │
  │    ShortTermMemory                                      │
  │      ├─ session_messages   : full LLM conversation log  │
  │      ├─ agent_execution_log: ordered dispatch history   │
  │      └─ evaluation_feed    : per-step Brain feedback     │
  │                                                         │
  │  LONG-TERM (Semantic / Cross-Session)                   │
  │    LongTermMemory                                       │
  │      ├─ operational_heuristics : routing lessons learned │
  │      ├─ ticker_insights        : per-ticker fact cache   │
  │      └─ user_preferences       : cross-session prefs    │
  └─────────────────────────────────────────────────────────┘

Design principles
-----------------
  - Short-term memory is fully in-memory; reset on each ``new_session()``.
  - Long-term memory is dict-based with clean ``load()`` / ``persist()``
    hooks so it can be backed by a JSON file, SQLite, or a vector store
    downstream — without changing the ManagerAgent's code.
  - All public methods are synchronous. Async persistence can be layered
    on top via the ``AsyncManagerMemory`` adapter pattern (future work).
  - Thread-safe for single-threaded asyncio usage (no explicit locking
    needed inside a single event loop).

Usage
-----
    memory = ManagerMemory()
    memory.new_session(task_query="Analyse NVDA for Q1 2025")

    # Short-term writes
    memory.add_message(role="user", content="...")
    memory.log_agent_dispatch(agent_name="ResearchAgent", ...)
    memory.add_evaluation_feedback(step="research", ...)

    # Long-term writes
    memory.store_heuristic(key="nvda_research_depth", value="advanced")
    memory.store_ticker_insight(ticker="NVDA", insight={...})

    # Recall
    ctx = memory.recall_short_term()
    hints = memory.recall_long_term(ticker="NVDA")
"""

from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


from supabase import create_client, Client

log = logging.getLogger("manager-memory")


# ══════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════

@dataclass
class AgentExecutionRecord:
    """
    Immutable log entry for a single specialist-agent dispatch.

    Attributes
    ----------
    agent_name    : str
        Class name of the dispatched agent, e.g. ``"ResearchAgent"``.
    dispatched_at : float
        Unix timestamp (seconds) at the moment of dispatch.
    directives    : dict[str, Any]
        Copy of ``manager_directives`` active at dispatch time.
    outcome       : str
        One of ``"success"`` | ``"partial"`` | ``"error"`` | ``"pending"``.
        Updated by the ManagerAgent after the agent's ``run()`` returns.
    duration_s    : float | None
        Wall-clock seconds the agent's ``run()`` took. None until resolved.
    error_message : str | None
        Exception message if ``outcome == "error"``. None otherwise.
    result_keys   : list[str]
        Top-level keys the agent wrote into SharedManagerState,
        e.g. ``["aggregated_research_context"]``.
    """
    agent_name:    str
    dispatched_at: float
    directives:    dict[str, Any]
    outcome:       str                = "pending"
    duration_s:    float | None       = None
    error_message: str | None         = None
    result_keys:   list[str]          = field(default_factory=list)


@dataclass
class EvaluationFeedback:
    """
    Brain's structured evaluation of a completed agent step.

    Attributes
    ----------
    step          : str
        Label for the evaluated step, e.g. ``"research"``, ``"financial"``.
    timestamp     : float
        Unix timestamp when the evaluation was recorded.
    passed        : bool
        Whether the Brain judged this step's output as sufficient.
    score         : int
        Brain's quality score 0-100 for this step's output.
    issues        : list[str]
        Specific data quality problems the Brain identified.
    next_action   : str
        Brain's recommended next action, e.g.
        ``"proceed_to_financial"`` | ``"re_run_research"`` | ``"finalise"``.
    raw_verdict   : str
        Full JSON string from the Brain's evaluation response.
    """
    step:        str
    timestamp:   float
    passed:      bool
    score:       int
    issues:      list[str]
    next_action: str
    raw_verdict: str


# ══════════════════════════════════════════════════════════════════
# Short-Term Memory
# ══════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    Ephemeral, session-scoped memory for the current ManagerAgent run.

    Resets completely on each ``new_session()`` call. Never persisted
    to disk. Holds the live LLM conversation, agent dispatch log,
    and evaluation feedback for the active orchestration session.

    Parameters
    ----------
    max_messages : int
        Maximum number of LLM messages to retain before oldest are
        trimmed (FIFO). Default: 50. Prevents context window overflow
        on very long orchestration sessions.
    """

    def __init__(self, max_messages: int = 50) -> None:
        self._max_messages:    int                      = max_messages
        self.session_id:       str                      = ""
        self.task_query:       str                      = ""
        self.session_start_ts: float                    = 0.0
        self.messages:         list[dict[str, str]]     = []
        self.agent_log:        list[AgentExecutionRecord] = []
        self.eval_feedback:    list[EvaluationFeedback]   = []

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def reset(self, session_id: str, task_query: str) -> None:
        """
        Wipe all session data and start a fresh short-term context.

        Parameters
        ----------
        session_id : str
            Unique identifier for this session (e.g. a UUID or timestamp).
        task_query : str
            The user's task query for this session.
        """
        self.session_id       = session_id
        self.task_query       = task_query
        self.session_start_ts = time.time()
        self.messages         = []
        self.agent_log        = []
        self.eval_feedback    = []
        log.info("[ShortTerm] Session reset: id=%s query='%s'", session_id, task_query[:60])

    # ------------------------------------------------------------------
    # Message log
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message to the running LLM conversation log.

        Trims the oldest message when ``max_messages`` is exceeded,
        always preserving at least the last ``max_messages`` entries.

        Parameters
        ----------
        role    : str   ``"user"`` | ``"assistant"`` | ``"system"``
        content : str   Raw message content.
        """
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self._max_messages:
            self.messages = self.messages[-self._max_messages:]
            log.debug("[ShortTerm] Message log trimmed to %d entries.", self._max_messages)

    def get_messages(self) -> list[dict[str, str]]:
        """
        Return a shallow copy of the current message log.

        Returns
        -------
        list[dict[str, str]]
            Ordered list of ``{"role": ..., "content": ...}`` dicts.
        """
        return list(self.messages)

    # ------------------------------------------------------------------
    # Agent execution log
    # ------------------------------------------------------------------

    def log_dispatch(
        self,
        agent_name: str,
        directives: dict[str, Any],
    ) -> AgentExecutionRecord:
        """
        Record a new agent dispatch and return the mutable record.

        The caller holds the returned record and updates ``outcome``,
        ``duration_s``, and ``result_keys`` after the agent completes.

        Parameters
        ----------
        agent_name : str
            Class name of the agent being dispatched.
        directives : dict[str, Any]
            Active ``manager_directives`` at dispatch time.

        Returns
        -------
        AgentExecutionRecord
            Mutable record; caller updates it in place.
        """
        record = AgentExecutionRecord(
            agent_name=agent_name,
            dispatched_at=time.time(),
            directives=deepcopy(directives),
        )
        self.agent_log.append(record)
        log.info("[ShortTerm] Dispatch logged: agent=%s", agent_name)
        return record

    def get_agent_log(self) -> list[AgentExecutionRecord]:
        """Return a shallow copy of the agent execution log."""
        return list(self.agent_log)

    def get_last_dispatch(self) -> AgentExecutionRecord | None:
        """Return the most recent dispatch record, or None if log is empty."""
        return self.agent_log[-1] if self.agent_log else None

    def agents_run(self) -> list[str]:
        """Return the ordered list of agent names that have been dispatched."""
        return [r.agent_name for r in self.agent_log]

    # ------------------------------------------------------------------
    # Evaluation feedback
    # ------------------------------------------------------------------

    def add_evaluation(self, feedback: EvaluationFeedback) -> None:
        """
        Store a Brain evaluation verdict for a completed agent step.

        Parameters
        ----------
        feedback : EvaluationFeedback
            Structured evaluation produced by the Brain after reviewing
            a specialist agent's output.
        """
        self.eval_feedback.append(feedback)
        log.info(
            "[ShortTerm] Evaluation stored: step=%s passed=%s score=%d next=%s",
            feedback.step, feedback.passed, feedback.score, feedback.next_action,
        )

    def get_last_evaluation(self) -> EvaluationFeedback | None:
        """Return the most recent evaluation feedback, or None."""
        return self.eval_feedback[-1] if self.eval_feedback else None

    def get_evaluations(self) -> list[EvaluationFeedback]:
        """Return a shallow copy of all evaluation feedbacks this session."""
        return list(self.eval_feedback)

    # ------------------------------------------------------------------
    # Summary for Brain context injection
    # ------------------------------------------------------------------

    def to_context_dict(self) -> dict[str, Any]:
        """
        Serialise the short-term memory into a Brain-readable summary.

        Returns a trimmed dict suitable for injection into a Claude prompt.
        Excludes full message history (passed separately) to avoid duplication.

        Returns
        -------
        dict[str, Any]
            Keys: session_id, task_query, session_elapsed_s,
                  agents_dispatched, last_evaluation, pending_agents.
        """
        last_eval = self.get_last_evaluation()
        return {
            "session_id":        self.session_id,
            "task_query":        self.task_query,
            "session_elapsed_s": round(time.time() - self.session_start_ts, 1),
            "agents_dispatched": [
                {
                    "agent":    r.agent_name,
                    "outcome":  r.outcome,
                    "duration": r.duration_s,
                    "keys":     r.result_keys,
                }
                for r in self.agent_log
            ],
            "last_evaluation": {
                "step":        last_eval.step,
                "passed":      last_eval.passed,
                "score":       last_eval.score,
                "next_action": last_eval.next_action,
                "issues":      last_eval.issues,
            } if last_eval else None,
        }


# ══════════════════════════════════════════════════════════════════
# Long-Term Memory
# ══════════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    Persistent, cross-session memory for the ManagerAgent.

    Stores three categories of durable knowledge:

    1. ``operational_heuristics`` — routing lessons learned across runs.
       Example: ``{"NVDA_preferred_depth": "advanced",
                   "research_before_financial": True}``

    2. ``ticker_insights`` — per-ticker cached facts and observations.
       Example: ``{"NVDA": {"sector": "Technology",
                             "last_grade": "A",
                             "last_sentiment": "Extreme Greed"}}``

    3. ``user_preferences`` — cross-session user-stated preferences.
       Example: ``{"report_format": "concise", "always_include_peers": True}``

    Persistence
    -----------
    Backed by a ``long_term_memory`` table in Supabase (PostgreSQL + JSONB).
    ``load()`` reads the row for the given ``user_id``; ``persist()`` upserts it.
    The interface is intentionally minimal so the ManagerAgent never needs
    to change when the storage backend changes.

    Required Supabase table
    -----------------------
    Run once in the Supabase SQL Editor::

        CREATE TABLE long_term_memory (
            user_id                 TEXT PRIMARY KEY,
            operational_heuristics  JSONB DEFAULT '{}',
            ticker_insights         JSONB DEFAULT '{}',
            user_preferences        JSONB DEFAULT '{}',
            updated_at              TIMESTAMPTZ DEFAULT NOW()
        );

    Parameters
    ----------
    user_id : str
        Identifier for the user whose memory is being managed.
        Supplied by the API layer from the incoming request; falls back
        to the ``DEFAULT_USER_ID`` env var for CLI / test usage.
    supabase_client : supabase.Client | None
        An already-initialised Supabase client. When None the class
        creates one from ``SUPABASE_URL`` / ``SUPABASE_KEY`` env vars.
    max_heuristics : int
        Maximum heuristic entries retained in memory. Default: 100.
    max_ticker_insights : int
        Maximum ticker entries retained in memory. Default: 200.
    """

    def __init__(
        self,
        user_id:          str            = "",
        supabase_client:  Client | None  = None,
        max_heuristics:     int = 100,
        max_ticker_insights: int = 200,
    ) -> None:
        # Resolve user_id — API passes it explicitly; CLI falls back to env var
        self._user_id: str = user_id or os.getenv("DEFAULT_USER_ID", "default_user")

        self._max_heuristics:      int = max_heuristics
        self._max_ticker_insights: int = max_ticker_insights

        # Supabase client — reuse injected client or create from env vars
        if supabase_client is not None:
            self._db: Client = supabase_client
        else:
            url = os.getenv("SUPABASE_URL", "")
            key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
            if not url or not key:
                raise ValueError(
                    "[LongTerm] SUPABASE_URL and SUPABASE_KEY must be set "
                    "when no supabase_client is injected."
                )
            self._db = create_client(url, key)

        # In-memory stores (populated by load())
        self.operational_heuristics: dict[str, Any]  = {}
        self.ticker_insights:        dict[str, dict] = {}
        self.user_preferences:       dict[str, Any]  = {}

        self.load()

    # ------------------------------------------------------------------
    # Operational heuristics
    # ------------------------------------------------------------------

    def store_heuristic(self, key: str, value: Any) -> None:
        """
        Store or update an operational heuristic.

        Evicts the oldest entry when the cap is reached (FIFO).

        Parameters
        ----------
        key   : str   Heuristic identifier (e.g. ``"nvda_search_depth"``).
        value : Any   JSON-serialisable value.
        """
        if key not in self.operational_heuristics and \
                len(self.operational_heuristics) >= self._max_heuristics:
            oldest = next(iter(self.operational_heuristics))
            del self.operational_heuristics[oldest]
            log.debug("[LongTerm] Heuristic cap reached — evicted key='%s'.", oldest)
        self.operational_heuristics[key] = value
        log.debug("[LongTerm] Heuristic stored: %s = %r", key, value)

    def get_heuristic(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a heuristic by key.

        Parameters
        ----------
        key     : str   Heuristic identifier.
        default : Any   Value returned when key is absent.
        """
        return self.operational_heuristics.get(key, default)

    def get_all_heuristics(self) -> dict[str, Any]:
        """Return a shallow copy of all stored heuristics."""
        return dict(self.operational_heuristics)

    # ------------------------------------------------------------------
    # Ticker insights
    # ------------------------------------------------------------------

    def store_ticker_insight(self, ticker: str, insight: dict[str, Any]) -> None:
        """
        Upsert a structured insight dict for a given ticker.

        Merges ``insight`` into any existing entry for the ticker.
        Evicts the least-recently-added ticker when the cap is reached.

        Parameters
        ----------
        ticker  : str              Uppercase ticker symbol.
        insight : dict[str, Any]   Key-value insight data.
        """
        ticker = ticker.upper()
        if ticker not in self.ticker_insights and \
                len(self.ticker_insights) >= self._max_ticker_insights:
            oldest = next(iter(self.ticker_insights))
            del self.ticker_insights[oldest]
            log.debug("[LongTerm] Ticker cap reached — evicted ticker='%s'.", oldest)
        existing = self.ticker_insights.get(ticker, {})
        existing.update(insight)
        existing["last_updated"] = time.time()
        self.ticker_insights[ticker] = existing
        log.info("[LongTerm] Ticker insight updated: %s keys=%s", ticker, list(insight.keys()))

    def get_ticker_insight(self, ticker: str) -> dict[str, Any]:
        """
        Retrieve all stored insights for a ticker.

        Parameters
        ----------
        ticker : str   Uppercase ticker symbol.

        Returns
        -------
        dict[str, Any]
            Stored insight dict, or empty dict if ticker not found.
        """
        return dict(self.ticker_insights.get(ticker.upper(), {}))

    # ------------------------------------------------------------------
    # User preferences
    # ------------------------------------------------------------------

    def store_preference(self, key: str, value: Any) -> None:
        """
        Store or update a cross-session user preference.

        Parameters
        ----------
        key   : str   Preference key (e.g. ``"report_format"``).
        value : Any   JSON-serialisable preference value.
        """
        self.user_preferences[key] = value
        log.debug("[LongTerm] Preference stored: %s = %r", key, value)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a user preference by key."""
        return self.user_preferences.get(key, default)

    def get_all_preferences(self) -> dict[str, Any]:
        """Return a shallow copy of all stored preferences."""
        return dict(self.user_preferences)

    # ------------------------------------------------------------------
    # Recall — unified summary for Brain context injection
    # ------------------------------------------------------------------

    def recall(self, ticker: str | None = None) -> dict[str, Any]:
        """
        Produce a unified recall payload for Brain context injection.

        Parameters
        ----------
        ticker : str | None
            If provided, includes ticker-specific insights in the recall.

        Returns
        -------
        dict[str, Any]
            Keys: heuristics, ticker_insight (if ticker given),
                  user_preferences, total_tickers_cached.
        """
        result: dict[str, Any] = {
            "heuristics":          self.get_all_heuristics(),
            "user_preferences":    self.get_all_preferences(),
            "total_tickers_cached": len(self.ticker_insights),
        }
        if ticker:
            result["ticker_insight"] = self.get_ticker_insight(ticker)
        return result

    # ------------------------------------------------------------------
    # Persistence — load / save
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load long-term memory from Supabase for the current user.

        Performs a ``SELECT`` on ``long_term_memory`` filtered by
        ``user_id``. Silently initialises to empty state when no row
        exists yet (first-time user).

        Raises
        ------
        Exception
            Propagated on unexpected Supabase / network errors.
        """
        try:
            result = (
                self._db
                .table("long_term_memory")
                .select("operational_heuristics, ticker_insights, user_preferences")
                .eq("user_id", self._user_id)
                .execute()
            )
            if result.data:
                row = result.data[0]
                self.operational_heuristics = row.get("operational_heuristics", {})
                self.ticker_insights        = row.get("ticker_insights", {})
                self.user_preferences       = row.get("user_preferences", {})
                log.info(
                    "[LongTerm] Loaded from Supabase — user=%s heuristics=%d tickers=%d prefs=%d",
                    self._user_id,
                    len(self.operational_heuristics),
                    len(self.ticker_insights),
                    len(self.user_preferences),
                )
            else:
                log.info("[LongTerm] No existing row for user=%s — starting empty.", self._user_id)
        except Exception as exc:
            log.error("[LongTerm] Load failed (%s) — starting empty.", exc)
            self.operational_heuristics = {}
            self.ticker_insights        = {}
            self.user_preferences       = {}

    def persist(self) -> None:
        """
        Save the current long-term memory to Supabase.

        Performs an ``UPSERT`` on ``long_term_memory`` keyed by ``user_id``.
        Creates the row on first call; updates it on subsequent calls.

        Raises
        ------
        Exception
            Propagated on unexpected Supabase / network errors.
        """
        try:
            self._db.table("long_term_memory").upsert(
                {
                    "user_id":                self._user_id,
                    "operational_heuristics": self.operational_heuristics,
                    "ticker_insights":        self.ticker_insights,
                    "user_preferences":       self.user_preferences,
                    "updated_at":             "now()",
                },
                on_conflict="user_id",
            ).execute()
            log.info("[LongTerm] Persisted to Supabase — user=%s.", self._user_id)
        except Exception as exc:
            log.error("[LongTerm] Persist failed: %s", exc)
            raise


# ══════════════════════════════════════════════════════════════════
# ManagerMemory — unified facade
# ══════════════════════════════════════════════════════════════════

class ManagerMemory:
    """
    Unified memory facade for the ManagerAgent.

    Composes ShortTermMemory (ephemeral, session-scoped) and LongTermMemory
    (persistent, cross-session) behind a single coherent API.

    The ManagerAgent interacts exclusively with this class — it never
    accesses ``ShortTermMemory`` or ``LongTermMemory`` directly.

    Parameters
    ----------
    user_id : str
        User identifier passed from the API layer. Falls back to the
        ``DEFAULT_USER_ID`` env var when empty (CLI / test usage).
    supabase_client : supabase.Client | None
        Optional pre-built Supabase client. When None, ``LongTermMemory``
        creates one from ``SUPABASE_URL`` / ``SUPABASE_KEY`` env vars.
    max_messages : int
        Maximum LLM messages retained in short-term memory. Default: 50.
    max_heuristics : int
        Maximum heuristic entries in long-term memory. Default: 100.
    max_ticker_insights : int
        Maximum ticker entries in long-term memory. Default: 200.

    Attributes
    ----------
    short : ShortTermMemory
        Ephemeral session memory.
    long  : LongTermMemory
        Persistent cross-session memory.

    Example
    -------
    >>> # CLI / test usage — user_id from env var DEFAULT_USER_ID
    >>> memory = ManagerMemory()

    >>> # API usage — user_id injected from the incoming request
    >>> memory = ManagerMemory(user_id="user_abc123")

    >>> # Inject an existing Supabase client (e.g. shared across the app)
    >>> memory = ManagerMemory(user_id="user_abc123", supabase_client=sb)

    >>> memory.new_session(session_id="sess_001", task_query="Analyse NVDA")
    >>> memory.add_message(role="user", content="Analyse NVDA for Q1 2025")
    >>> record = memory.log_dispatch("ResearchAgent", {"ticker": "NVDA"})
    >>> record.outcome    = "success"
    >>> record.duration_s = 12.4
    >>> record.result_keys = ["aggregated_research_context"]
    >>> memory.persist_long_term()
    """

    def __init__(
        self,
        user_id:             str           = "",
        supabase_client:     Client | None = None,
        max_messages:        int           = 50,
        max_heuristics:      int           = 100,
        max_ticker_insights: int           = 200,
    ) -> None:
        self.short = ShortTermMemory(max_messages=max_messages)
        self.long  = LongTermMemory(
            user_id=user_id,
            supabase_client=supabase_client,
            max_heuristics=max_heuristics,
            max_ticker_insights=max_ticker_insights,
        )
        log.info("ManagerMemory initialised (user=%s).", user_id or "from_env")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(self, session_id: str, task_query: str) -> None:
        """
        Start a new orchestration session.

        Resets all short-term memory while preserving long-term memory.

        Parameters
        ----------
        session_id  : str   Unique session identifier.
        task_query  : str   The user's task for this session.
        """
        self.short.reset(session_id=session_id, task_query=task_query)
        log.info("ManagerMemory: new session started — id=%s", session_id)

    # ------------------------------------------------------------------
    # Short-term delegation
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the short-term LLM conversation log."""
        self.short.add_message(role=role, content=content)

    def get_messages(self) -> list[dict[str, str]]:
        """Return the current short-term message log."""
        return self.short.get_messages()

    def log_dispatch(
        self,
        agent_name: str,
        directives: dict[str, Any],
    ) -> AgentExecutionRecord:
        """
        Record a new agent dispatch in short-term memory.

        Returns the mutable ``AgentExecutionRecord`` so the caller can
        update ``outcome``, ``duration_s``, and ``result_keys`` after
        the agent completes.
        """
        return self.short.log_dispatch(agent_name=agent_name, directives=directives)

    def add_evaluation(self, feedback: EvaluationFeedback) -> None:
        """Store a Brain evaluation verdict in short-term memory."""
        self.short.add_evaluation(feedback)

    def get_last_evaluation(self) -> EvaluationFeedback | None:
        """Return the most recent Brain evaluation, or None."""
        return self.short.get_last_evaluation()

    def agents_run(self) -> list[str]:
        """Return the ordered list of agents dispatched this session."""
        return self.short.agents_run()

    # ------------------------------------------------------------------
    # Long-term delegation
    # ------------------------------------------------------------------

    def store_heuristic(self, key: str, value: Any) -> None:
        """Store an operational heuristic in long-term memory."""
        self.long.store_heuristic(key=key, value=value)

    def get_heuristic(self, key: str, default: Any = None) -> Any:
        """Retrieve a long-term heuristic."""
        return self.long.get_heuristic(key=key, default=default)

    def store_ticker_insight(self, ticker: str, insight: dict[str, Any]) -> None:
        """Upsert structured insight data for a ticker in long-term memory."""
        self.long.store_ticker_insight(ticker=ticker, insight=insight)

    def get_ticker_insight(self, ticker: str) -> dict[str, Any]:
        """Retrieve all long-term insights for a ticker."""
        return self.long.get_ticker_insight(ticker=ticker)

    def store_preference(self, key: str, value: Any) -> None:
        """Store a cross-session user preference in long-term memory."""
        self.long.store_preference(key=key, value=value)

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Retrieve a user preference from long-term memory."""
        return self.long.get_preference(key=key, default=default)

    def persist_long_term(self) -> None:
        """Flush long-term memory to disk (no-op if no path configured)."""
        self.long.persist()

    # ------------------------------------------------------------------
    # Unified recall — for Brain context injection
    # ------------------------------------------------------------------

    def recall(self, ticker: str | None = None) -> dict[str, Any]:
        """
        Produce the unified memory context payload for Brain injection.

        Merges short-term session context with long-term durable knowledge
        into a single structured dict ready for JSON-serialisation into a
        Claude prompt.

        Parameters
        ----------
        ticker : str | None
            If provided, includes ticker-specific long-term insights.

        Returns
        -------
        dict[str, Any]
            Keys: short_term, long_term.
        """
        return {
            "short_term": self.short.to_context_dict(),
            "long_term":  self.long.recall(ticker=ticker),
        }