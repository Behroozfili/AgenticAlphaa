"""
api/dependencies.py
-------------------
FastAPI Dependency Injection helpers for Alpha-Agent Node.

Usage in any route
------------------
    from fastapi import Depends
    from api.dependencies import get_manager_memory

    @router.post("/analyze")
    async def analyze(
        req: AnalyzeRequest,
        request: Request,
        memory: ManagerMemory = Depends(get_manager_memory),
    ):
        result = await request.app.state.manager_agent.run(
            task_query=req.query,
            user_preferences=memory.long.get_all_preferences(),
        )

Why DI instead of manual instantiation
---------------------------------------
- Testable: override get_manager_memory in tests with one line:
      app.dependency_overrides[get_manager_memory] = lambda: mock_memory
- Swappable: change the backend (Supabase → Redis) in ONE place
- No global state: each request gets its own scoped memory instance
"""

from __future__ import annotations

import logging

from fastapi import Depends, Request

from memory.manager_memory import ManagerMemory
from api.config import settings

log = logging.getLogger("api.dependencies")


# ─────────────────────────────────────────────────────────────
# user_id extractor
# ─────────────────────────────────────────────────────────────

def get_user_id(request: Request) -> str:
    """
    Extract user_id from the request.

    Priority:
      1. X-User-Id header  (set by API gateway / future JWT middleware)
      2. Body field         (current approach — parsed separately per route)
      3. DEFAULT_USER_ID    (fallback from settings)

    When JWT auth is added, replace body parsing with:
        return request.state.user_id  # set by JWT middleware

    Note: routes that need user_id from the body (like /analyze) should
    pass it explicitly to get_manager_memory() rather than relying on this.
    """
    return (
        request.headers.get("X-User-Id")
        or settings.DEFAULT_USER_ID
    )


# ─────────────────────────────────────────────────────────────
# ManagerMemory factory
# ─────────────────────────────────────────────────────────────

def get_manager_memory(
    request: Request,
    user_id: str = Depends(get_user_id),
) -> ManagerMemory:
    """
    Dependency that creates a per-request, user-scoped ManagerMemory.

    Injects the shared Supabase client from app.state so no new
    connection is opened per request.

    Parameters
    ----------
    request : Request
        FastAPI request — used to access app.state.supabase.
    user_id : str
        Resolved by get_user_id(). Routes can override this by passing
        user_id explicitly when calling get_manager_memory directly.

    Returns
    -------
    ManagerMemory
        Ready-to-use memory instance with long-term data already loaded
        from Supabase (load() is called in ManagerMemory.__init__).

    Test override example
    ----------------------
        from api.dependencies import get_manager_memory
        from unittest.mock import MagicMock

        mock_memory = MagicMock(spec=ManagerMemory)
        app.dependency_overrides[get_manager_memory] = lambda: mock_memory
    """
    supabase_client = request.app.state.supabase
    log.debug("Creating ManagerMemory for user_id=%s", user_id)
    return ManagerMemory(
        user_id=user_id,
        supabase_client=supabase_client,
    )
