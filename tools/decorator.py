import asyncio
import logging
from functools import wraps

from langsmith import traceable

logger = logging.getLogger(__name__)


def handle_error(error: Exception, name: str):
    logger.exception(f"Error in {name}")
    raise


def traced_and_handled(name: str):
    def decorator(func):

        if asyncio.iscoroutinefunction(func):

            @traceable(name=name)
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)

                except asyncio.CancelledError:
                    # اجازه بده cancellation طبیعی کار کند
                    raise

                except Exception as error:
                    handle_error(error, name)

            return async_wrapper

        @traceable(name=name)
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except Exception as error:
                handle_error(error, name)

        return sync_wrapper

    return decorator