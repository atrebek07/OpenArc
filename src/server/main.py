# The first implementation of the OpenAI-like API was contributed by @gapeleon.
# They are one hero among many future heroes working to make OpenArc better.

import logging
import os
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from src.server.deps import _registry
from src.server.routes.openai import router as openai_router
from src.server.routes.openarc import router as openarc_router

logger = logging.getLogger(__name__)
_access_logger = logging.getLogger("openarc.access")


class RequestLoggingMiddleware:
    """Plain ASGI wrapper, deliberately NOT a BaseHTTPMiddleware: base middleware
    interposes its own receive queue and swallows the http.disconnect message,
    which breaks handler-level is_disconnected() watchers — the mid-denoise cancel
    on /v1/images/generations needs the raw receive stream to reach the route."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        start_time = time.time()
        _access_logger.info(
            f"Request received: {method} {path} from {client_ip}"
        )

        status_code = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start" and status_code is None:
                status_code = message.get("status")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            process_time = time.time() - start_time
            _access_logger.error(
                f"Request failed: {method} {path} "
                f"error={str(e)} duration={process_time:.3f}s"
            )
            raise

        process_time = time.time() - start_time
        _access_logger.info(
            f"Request completed: {method} {path} "
            f"status={status_code} duration={process_time:.3f}s"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Let OPENARC_STARTUP_MODELS (exported by `openarc serve start`) win over the
    # config file's startup_models key.
    models = os.getenv("OPENARC_STARTUP_MODELS", "").strip()
    if not models:
        from src.cli.modules.server_config import ServerConfig

        configured = ServerConfig().load_config().get("startup_models") or ""
        models = configured.strip() if isinstance(configured, str) else ""

    if models:
        from pathlib import Path

        from src.cli.modules.server_config import ServerConfig

        # All config parsing, env interpolation and relative-path resolution
        # happens inside ServerConfig so startup and the CLI share one loader.
        server_config = ServerConfig()

        for name in (m.strip() for m in models.split(",")):
            if not name:
                continue
            try:
                model_config = server_config.get_model_load_config(name)
            except Exception as e:
                logger.error(f"Startup: invalid config for '{name}': {e}")
                continue

            if model_config is None:
                logger.warning(f"Startup: model '{name}' not in config, skipping")
                continue

            try:
                if model_config.cache_dir:
                    # Create the cache directory at startup if it doesn't exist.
                    Path(model_config.cache_dir).mkdir(parents=True, exist_ok=True)
                await _registry.register_load(model_config)
                logger.info(f"Startup: loaded '{name}'")
            except Exception as e:
                logger.error(f"Startup: failed to load '{name}': {e}")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}", exc_info=True)
    return JSONResponse(
        status_code=422, content={"status": "error", "detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    logger.error(f"Full traceback:\n{''.join(traceback.format_tb(exc.__traceback__))}")
    return JSONResponse(
        status_code=500, content={"status": "error", "detail": str(exc)}
    )


@app.get("/readyz")
async def readyz():
    """Readiness probe: 200 when every model that should be loaded is loaded.

    Intentionally unauthenticated so orchestrators (e.g. Kubernetes) can probe
    it without credentials.
    """
    result = await _registry.readiness()
    status_code = 200 if result["ready"] else 503
    return JSONResponse(status_code=status_code, content=result)


app.include_router(openarc_router)
app.include_router(openai_router)
