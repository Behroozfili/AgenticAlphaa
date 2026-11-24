"""Request-scoped runtime factories for the FastAPI layer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Depends, Request

from api.config import Settings, get_settings
from memory.manager_memory import ManagerMemory
from langsmith_trace import emit_stage10a_event, new_stage10a_run_id

log = logging.getLogger("api.dependencies")
settings = get_settings()  # backward-compatible patch/import surface


def _request_telemetry(request: Request) -> tuple[str, Any]:
    run_id = getattr(request.state, "stage10a_run_id", None)
    if not isinstance(run_id, str) or not run_id:
        run_id = new_stage10a_run_id()
        request.state.stage10a_run_id = run_id
    sink = getattr(request.app.state, "telemetry_sink", None)
    return run_id, sink if callable(sink) else None


def get_user_id(request: Request) -> str:
    """Return only middleware-verified identity; client identity headers are ignored."""
    identity = getattr(request.state, "user_id", None)
    return identity if isinstance(identity, str) and identity.strip() else settings.DEFAULT_USER_ID


def get_manager_memory(
    request: Request,
    user_id: str = Depends(get_user_id),
) -> ManagerMemory:
    """Create and load a new memory facade for every request."""
    run_id, sink = _request_telemetry(request)
    started_at = time.perf_counter()
    try:
        client = getattr(request.app.state, "supabase", None)
        if client is None:
            raise RuntimeError("Supabase client is unavailable")
        memory = ManagerMemory(user_id=user_id, supabase_client=client)
    except Exception:
        emit_stage10a_event(
            "manager.memory.failed",
            metadata={"run_id": run_id, "component": "manager_memory", "status": "FAILED", "error_code": "MEMORY_CREATE_FAILED"},
            started_at=started_at,
            sink=sink,
        )
        log.error("manager_memory_create_failed code=MEMORY_CREATE_FAILED")
        raise
    emit_stage10a_event(
        "manager.memory.created",
        metadata={"run_id": run_id, "component": "manager_memory", "status": "READY"},
        started_at=started_at,
        sink=sink,
    )
    return memory


ManagerBuilder = Callable[[ManagerMemory], Any]


@dataclass(frozen=True, slots=True)
class ManagerAgentFactory:
    """Stateless manager factory; only thread-safe clients are retained.

    ``ManagerAgent`` owns mutable graph and memory state, so a new instance is
    built for each request. Supabase and Neo4j connection clients may live for
    the application lifecycle and are never used as request-state containers.
    """

    supabase_client: Any
    neo4j_driver: Any
    runtime_settings: Settings
    manager_builder: ManagerBuilder | None = None
    market_forecast_service: Any | None = None
    market_data_provider: Any | None = None

    @staticmethod
    def _agent_types() -> tuple[type, type, type, type]:
        """Import the heavy agent modules once before the first live request."""
        from agents.financial_agent import FinancialAnalystAgent
        from agents.manager_agent import ManagerAgent
        from agents.research_agent import ResearchAgent
        from agents.sentiment_agent import SentimentAgent

        return FinancialAnalystAgent, ManagerAgent, ResearchAgent, SentimentAgent

    def preload(self) -> None:
        """Warm module imports during API startup; retain no mutable agent state."""
        self._agent_types()

    def create_memory(
        self,
        *,
        user_id: str | None = None,
        run_id: str | None = None,
        sink: Any = None,
    ) -> ManagerMemory:
        correlation_id = run_id or new_stage10a_run_id()
        started_at = time.perf_counter()
        try:
            memory = ManagerMemory(
                user_id=user_id or self.runtime_settings.DEFAULT_USER_ID,
                supabase_client=self.supabase_client,
            )
        except Exception:
            emit_stage10a_event(
                "manager.memory.failed",
                metadata={"run_id": correlation_id, "component": "manager_memory", "status": "FAILED", "error_code": "MEMORY_CREATE_FAILED"},
                started_at=started_at,
                sink=sink,
            )
            raise
        emit_stage10a_event(
            "manager.memory.created",
            metadata={"run_id": correlation_id, "component": "manager_memory", "status": "READY"},
            started_at=started_at,
            sink=sink,
        )
        return memory

    def create_manager(self, memory: ManagerMemory) -> Any:
        if self.manager_builder is not None:
            return self.manager_builder(memory)

        FinancialAnalystAgent, ManagerAgent, ResearchAgent, SentimentAgent = self._agent_types()

        script_path = Path(__file__).resolve().parents[1] / "tools" / "sentiment_tools" / "sentiment_server.py"
        return ManagerAgent(
            research_agent=ResearchAgent(),
            financial_agent=FinancialAnalystAgent(),
            sentiment_agent=SentimentAgent(server_script_path=str(script_path)),
            memory=memory,
            model=self.runtime_settings.ANTHROPIC_MODEL,
            max_routing_loops=self.runtime_settings.MAX_ROUTING_LOOPS,
            market_forecast_service=self.market_forecast_service,
            market_data_provider=self.market_data_provider,
        )

    def for_request(
        self,
        memory: ManagerMemory,
        *,
        run_id: str | None = None,
        sink: Any = None,
    ) -> Any:
        correlation_id = run_id or new_stage10a_run_id()
        started_at = time.perf_counter()
        try:
            manager = self.create_manager(memory)
        except Exception:
            emit_stage10a_event(
                "manager.factory.failed",
                metadata={"run_id": correlation_id, "component": "manager_factory", "status": "FAILED", "error_code": "MANAGER_CREATE_FAILED"},
                started_at=started_at,
                sink=sink,
            )
            raise
        emit_stage10a_event(
            "manager.factory.created",
            metadata={"run_id": correlation_id, "component": "manager_factory", "status": "READY"},
            started_at=started_at,
            sink=sink,
        )
        return manager

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Compatibility shim for Stage-10B's existing route.

        The old route calls ``app.state.manager_agent.run``. Keeping that
        surface on this stateless factory preserves imports while ensuring each
        invocation receives a fresh Manager and fresh memory.
        """
        correlation_id = kwargs.pop("_stage10a_run_id", None) or new_stage10a_run_id()
        sink = kwargs.pop("_stage10a_event_sink", None)
        memory = self.create_memory(run_id=correlation_id, sink=sink)
        manager = self.for_request(memory, run_id=correlation_id, sink=sink)
        result: Awaitable[Any] = manager.run(*args, **kwargs)
        return await result


def get_manager_factory(request: Request) -> ManagerAgentFactory:
    factory = getattr(request.app.state, "manager_factory", None)
    if not isinstance(factory, ManagerAgentFactory):
        raise RuntimeError("Manager factory is unavailable")
    return factory


def get_manager_agent(
    memory: ManagerMemory = Depends(get_manager_memory),
    factory: ManagerAgentFactory = Depends(get_manager_factory),
) -> Any:
    """Return a fresh ManagerAgent bound to this request's memory."""
    return factory.for_request(memory)


def get_request_manager(request: Request, memory: ManagerMemory) -> Any:
    """Resolve a request-bound Manager while preserving legacy test/app surfaces."""
    factory = getattr(request.app.state, "manager_factory", None)
    if isinstance(factory, ManagerAgentFactory):
        run_id, sink = _request_telemetry(request)
        return factory.for_request(memory, run_id=run_id, sink=sink)
    return request.app.state.manager_agent
