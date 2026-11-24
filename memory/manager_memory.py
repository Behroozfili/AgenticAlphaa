"""
memory/manager_memory.py — ManagerMemory
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


from supabase import create_client, Client  # noqa: E402
from core.forecast_contract import AnalysisMode, HORIZON_TRADING_DAYS, parse_utc_datetime

log = logging.getLogger("manager-memory")


# ══════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════

@dataclass
class AgentExecutionRecord:
  
    agent_name:    str
    dispatched_at: float
    directives:    dict[str, Any]
    outcome:       str                = "pending"
    duration_s:    float | None       = None
    error_message: str | None         = None
    result_keys:   list[str]          = field(default_factory=list)


@dataclass
class EvaluationFeedback:
    
   
    step:        str
    timestamp:   float
    passed:      bool
    score:       int
    issues:      list[str]
    next_action: str
    raw_verdict: str


@dataclass(frozen=True, slots=True)
class MonthlyMemoryKey:
    """Natural identity for a monthly analysis, separate from legacy ticker data."""

    ticker: str
    as_of: str
    horizon: int
    analysis_mode: str
    schema_version: str

    @classmethod
    def create(
        cls,
        *,
        ticker: str,
        as_of: Any,
        horizon: int = HORIZON_TRADING_DAYS,
        analysis_mode: AnalysisMode | str = AnalysisMode.LIVE,
        schema_version: str,
    ) -> "MonthlyMemoryKey":
        canonical_ticker = ticker.strip().upper()
        if not canonical_ticker:
            raise ValueError("ticker must not be empty")
        if horizon != HORIZON_TRADING_DAYS:
            raise ValueError("monthly memory horizon must be exactly 21 trading sessions")
        version = schema_version.strip()
        if not version:
            raise ValueError("schema_version must not be empty")
        mode = AnalysisMode(analysis_mode).value
        cutoff = parse_utc_datetime(as_of, field_name="as_of").isoformat()
        return cls(canonical_ticker, cutoff, horizon, mode, version)

    def as_tuple(self) -> tuple[str, str, int, str, str]:
        return (
            self.ticker,
            self.as_of,
            self.horizon,
            self.analysis_mode,
            self.schema_version,
        )


# ══════════════════════════════════════════════════════════════════
# Short-Term Memory
# ══════════════════════════════════════════════════════════════════

class ShortTermMemory:
  

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
        
        self.session_id       = session_id
        self.task_query       = task_query
        self.session_start_ts = time.time()
        self.messages         = []
        self.agent_log        = []
        self.eval_feedback    = []
        log.info("short_term_session_reset")

    # ------------------------------------------------------------------
    # Message log
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self._max_messages:
            self.messages = self.messages[-self._max_messages:]
            log.debug("[ShortTerm] Message log trimmed to %d entries.", self._max_messages)

    def get_messages(self) -> list[dict[str, str]]:
      
        return list(self.messages)

    # ------------------------------------------------------------------
    # Agent execution log
    # ------------------------------------------------------------------

    def log_dispatch(
        self,
        agent_name: str,
        directives: dict[str, Any],
    ) -> AgentExecutionRecord:
     
        record = AgentExecutionRecord(
            agent_name=agent_name,
            dispatched_at=time.time(),
            directives=deepcopy(directives),
        )
        self.agent_log.append(record)
        log.info("short_term_dispatch_recorded")
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
    
        self.eval_feedback.append(feedback)
        log.info("short_term_evaluation_recorded")

    def get_last_evaluation(self) -> EvaluationFeedback | None:
       
        return self.eval_feedback[-1] if self.eval_feedback else None

    def get_evaluations(self) -> list[EvaluationFeedback]:
      
        return list(self.eval_feedback)

    # ------------------------------------------------------------------
    # Summary for Brain context injection
    # ------------------------------------------------------------------

    def to_context_dict(self) -> dict[str, Any]:
       
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
    

    def __init__(
        self,
        user_id:          str            = "",
        supabase_client:  Client | None  = None,
        max_heuristics:     int = 100,
        max_ticker_insights: int = 200,
    ) -> None:
        
        self._user_id: str = user_id or os.getenv("DEFAULT_USER_ID", "default_user")

        self._max_heuristics:      int = max_heuristics
        self._max_ticker_insights: int = max_ticker_insights

        
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

      
        self.operational_heuristics: dict[str, Any]  = {}
        self.ticker_insights:        dict[str, dict] = {}
        self.user_preferences:       dict[str, Any]  = {}
        # Monthly cache is separate from generic heuristics. Durable monthly
        # insights are written by the API into immutable monthly_forecasts.
        self.monthly_insights: dict[MonthlyMemoryKey, dict[str, Any]] = {}

      
    @classmethod
    def create(
        cls,
        user_id:             str           = "",
        supabase_client:     "Client | None" = None,
        max_heuristics:      int           = 100,
        max_ticker_insights: int           = 200,
    ) -> "LongTermMemory":
     
        instance = cls(
            user_id=user_id,
            supabase_client=supabase_client,
            max_heuristics=max_heuristics,
            max_ticker_insights=max_ticker_insights,
        )
        instance.load()
        return instance

    # ------------------------------------------------------------------
    # Operational heuristics
    # ------------------------------------------------------------------

    def store_heuristic(self, key: str, value: Any) -> None:
        
        if key not in self.operational_heuristics and \
                len(self.operational_heuristics) >= self._max_heuristics:
            oldest = next(iter(self.operational_heuristics))
            del self.operational_heuristics[oldest]
            log.debug("long_term_heuristic_evicted")
        self.operational_heuristics[key] = value
        log.debug("long_term_heuristic_stored")

    def get_heuristic(self, key: str, default: Any = None) -> Any:
     
        return self.operational_heuristics.get(key, default)

    def get_all_heuristics(self) -> dict[str, Any]:
        """Return a shallow copy of all stored heuristics."""
        return dict(self.operational_heuristics)

    # ------------------------------------------------------------------
    # Ticker insights
    # ------------------------------------------------------------------

    def store_ticker_insight(self, ticker: str, insight: dict[str, Any]) -> None:
     
        ticker = ticker.upper()
        if ticker not in self.ticker_insights and \
                len(self.ticker_insights) >= self._max_ticker_insights:
            oldest = next(iter(self.ticker_insights))
            del self.ticker_insights[oldest]
            log.debug("long_term_ticker_evicted")
        existing = self.ticker_insights.get(ticker, {})
        existing.update(insight)
        existing["last_updated"] = time.time()
        self.ticker_insights[ticker] = existing
        log.info("long_term_ticker_insight_updated")

    def get_ticker_insight(self, ticker: str) -> dict[str, Any]:
     
        return dict(self.ticker_insights.get(ticker.upper(), {}))

    def store_monthly_insight(
        self,
        *,
        ticker: str,
        as_of: Any,
        horizon: int = HORIZON_TRADING_DAYS,
        analysis_mode: AnalysisMode | str = AnalysisMode.LIVE,
        schema_version: str,
        insight: dict[str, Any],
    ) -> MonthlyMemoryKey:
        """Store one monthly value without merging with legacy ticker memory."""
        key = MonthlyMemoryKey.create(
            ticker=ticker,
            as_of=as_of,
            horizon=horizon,
            analysis_mode=analysis_mode,
            schema_version=schema_version,
        )
        self.monthly_insights[key] = deepcopy(dict(insight))
        return key

    def get_monthly_insight(
        self,
        *,
        ticker: str,
        as_of: Any,
        horizon: int = HORIZON_TRADING_DAYS,
        analysis_mode: AnalysisMode | str = AnalysisMode.LIVE,
        schema_version: str,
        model_version: str | None = None,
        dataset_version: str | None = None,
        feature_version: str | None = None,
        cutoff_policy: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        key = MonthlyMemoryKey.create(
            ticker=ticker,
            as_of=as_of,
            horizon=horizon,
            analysis_mode=analysis_mode,
            schema_version=schema_version,
        )
        cached = self.monthly_insights.get(key)
        filters = {k: v for k, v in {
            "model_version": model_version, "dataset_version": dataset_version,
            "feature_version": feature_version, "cutoff_policy": cutoff_policy, "run_id": run_id,
        }.items() if v is not None}
        if cached is not None and all(
            (cached.get("versions", {}).get(k) if k.endswith("_version") else cached.get(k)) == v
            for k, v in filters.items()
        ):
            return deepcopy(cached)
        # Exact cutoff/mode/schema only: never mix live memory into backtests.
        try:
            query = (self._db.table("monthly_forecasts")
                .select("forecast_payload,feature_snapshot_id")
                .eq("ticker", key.ticker).eq("as_of", key.as_of)
                .eq("horizon_trading_days", key.horizon)
                .eq("schema_version", key.schema_version)
                .eq("forecast_payload->>analysis_mode", key.analysis_mode))
            for field, value in filters.items():
                query = query.eq(field, value)
            response = query.order("created_at", desc=True).limit(1).execute()
            rows = response.data
            if not isinstance(rows, list) or not rows:
                return {}
            payload = rows[0].get("forecast_payload") or {}
            insight = payload.get("monthly_insight")
            if not isinstance(insight, dict):
                return {}
            recovered = deepcopy(insight)
            recovered["feature_snapshot_id"] = rows[0].get("feature_snapshot_id")
            self.monthly_insights[key] = recovered
            return deepcopy(recovered)
        except Exception:
            log.warning("monthly_memory_recall_failed code=MONTHLY_MEMORY_RECALL_FAILED")
            return {}

    # ------------------------------------------------------------------
    # User preferences
    # ------------------------------------------------------------------

    def store_preference(self, key: str, value: Any) -> None:
      
        self.user_preferences[key] = value
        log.debug("long_term_preference_stored")

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
      
        result: dict[str, Any] = {
            "heuristics":          self.get_all_heuristics(),
            "user_preferences":    self.get_all_preferences(),
            "total_tickers_cached": len(self.ticker_insights),
            "total_monthly_analyses_cached": len(self.monthly_insights),
        }
        if ticker:
            result["ticker_insight"] = self.get_ticker_insight(ticker)
        return result

    # ------------------------------------------------------------------
    # Persistence — load / save
    # ------------------------------------------------------------------

    def load(self) -> None:
       
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
                    "long_term_load_completed heuristics=%d tickers=%d preferences=%d",
                    len(self.operational_heuristics),
                    len(self.ticker_insights),
                    len(self.user_preferences),
                )
            else:
                log.info("long_term_load_empty")
        except Exception:
            log.error("long_term_load_failed code=MEMORY_LOAD_FAILED")
            self.operational_heuristics = {}
            self.ticker_insights        = {}
            self.user_preferences       = {}

    def persist(self) -> None:
    
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
            log.info("long_term_persist_completed")
        except Exception:
            log.error("long_term_persist_failed code=MEMORY_PERSIST_FAILED")
            raise


# ══════════════════════════════════════════════════════════════════
# ManagerMemory — unified facade
# ══════════════════════════════════════════════════════════════════

class ManagerMemory:
    

    def __init__(
        self,
        user_id:             str           = "",
        supabase_client:     Client | None = None,
        max_messages:        int           = 50,
        max_heuristics:      int           = 100,
        max_ticker_insights: int           = 200,
    ) -> None:
        self.short = ShortTermMemory(max_messages=max_messages)
        self.long  = LongTermMemory.create(
            user_id=user_id,
            supabase_client=supabase_client,
            max_heuristics=max_heuristics,
            max_ticker_insights=max_ticker_insights,
        )
        log.info("manager_memory_initialized")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(self, session_id: str, task_query: str) -> None:
       
        self.short.reset(session_id=session_id, task_query=task_query)
        log.info("manager_memory_session_started")

    # ------------------------------------------------------------------
    # Short-term delegation
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
       
        self.short.add_message(role=role, content=content)

    def get_messages(self) -> list[dict[str, str]]:
        
        return self.short.get_messages()

    def log_dispatch(
        self,
        agent_name: str,
        directives: dict[str, Any],
    ) -> AgentExecutionRecord:
     
        return self.short.log_dispatch(agent_name=agent_name, directives=directives)

    def add_evaluation(self, feedback: EvaluationFeedback) -> None:
        
        self.short.add_evaluation(feedback)

    def get_last_evaluation(self) -> EvaluationFeedback | None:
       
        return self.short.get_last_evaluation()

    def agents_run(self) -> list[str]:
       
        return self.short.agents_run()

    # ------------------------------------------------------------------
    # Long-term delegation
    # ------------------------------------------------------------------

    def store_heuristic(self, key: str, value: Any) -> None:
       
        self.long.store_heuristic(key=key, value=value)

    def get_heuristic(self, key: str, default: Any = None) -> Any:
       
        return self.long.get_heuristic(key=key, default=default)

    def store_ticker_insight(self, ticker: str, insight: dict[str, Any]) -> None:
   
        self.long.store_ticker_insight(ticker=ticker, insight=insight)

    def get_ticker_insight(self, ticker: str) -> dict[str, Any]:
        """Retrieve all long-term insights for a ticker."""
        return self.long.get_ticker_insight(ticker=ticker)

    def store_monthly_insight(self, **kwargs: Any) -> MonthlyMemoryKey:
        return self.long.store_monthly_insight(**kwargs)

    def get_monthly_insight(self, **kwargs: Any) -> dict[str, Any]:
        return self.long.get_monthly_insight(**kwargs)

    def store_preference(self, key: str, value: Any) -> None:
        
        self.long.store_preference(key=key, value=value)

    def get_preference(self, key: str, default: Any = None) -> Any:
     
        return self.long.get_preference(key=key, default=default)

    def persist_long_term(self) -> None:
        
        self.long.persist()

    # ------------------------------------------------------------------
    # Unified recall — for Brain context injection
    # ------------------------------------------------------------------

    def recall(self, ticker: str | None = None) -> dict[str, Any]:
       
        return {
            "short_term": self.short.to_context_dict(),
            "long_term":  self.long.recall(ticker=ticker),
        }
