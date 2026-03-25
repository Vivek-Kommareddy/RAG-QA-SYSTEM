"""Custom middleware and exception handlers for the FastAPI application.

Components
----------
- :class:`RequestLoggingMiddleware` – logs every request with method, path and
  response time.
- :func:`global_exception_handler` – catches unhandled exceptions and returns a
  structured JSON error response instead of crashing.
"""

from __future__ import annotations

import logging
import time
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log the HTTP method, path, status code, and latency for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        """Process the request, log timing information, and return the response.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response from the downstream handler.
        """
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d  (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a JSON 500 response for any unhandled exception.

    This prevents FastAPI from returning an HTML error page and ensures the
    API always speaks JSON even in error conditions.

    Args:
        request: The incoming request that triggered the exception.
        exc: The unhandled exception.

    Returns:
        JSON response with ``detail`` and ``type`` fields.
    """
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc) or "An internal server error occurred.",
            "type": type(exc).__name__,
        },
    )
