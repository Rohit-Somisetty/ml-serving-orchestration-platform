from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ml_platform.monitoring.logger import StructuredLogger


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, logger: StructuredLogger) -> None:
        super().__init__(app)
        self.logger = logger

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.canary_used = False
        start = time.perf_counter()
        response: Response | None = None
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except HTTPException as exc:  # pragma: no cover - defensive
            status_code = exc.status_code
            error_msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            self.logger.log_event(
                {
                    "event": "request_error",
                    "request_id": request_id,
                    "path": request.url.path,
                    "status": status_code,
                    "error": error_msg,
                    "canary_used": bool(getattr(request.state, "canary_used", False)),
                },
            )
            raise
        except Exception as exc:  # pragma: no cover - logged for visibility
            status_code = 500
            error_msg = str(exc)
            self.logger.log_event(
                {
                    "event": "request_error",
                    "request_id": request_id,
                    "path": request.url.path,
                    "status": status_code,
                    "error": error_msg,
                    "canary_used": bool(getattr(request.state, "canary_used", False)),
                },
            )
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.logger.log_event(
                {
                    "event": "request_complete",
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 2),
                    "canary_used": bool(getattr(request.state, "canary_used", False)),
                },
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id
        assert response is not None
        return response
