import argparse
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from samplespace.core.config import get_settings
from samplespace.routers.main import api_router

config = get_settings()

logger = logging.getLogger(__name__)


def generate_operation_id(route: APIRoute) -> str:
    """Generate clean camelCase operationIds for OpenAPI spec."""
    parts = route.name.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: load heavy models at startup, clean up on shutdown."""
    logger.info("Starting SampleSpace backend...")
    # CLAP model will be loaded here in Phase 2
    yield
    logger.info("Shutting down SampleSpace backend...")


app = FastAPI(
    title="SampleSpace",
    openapi_url=f"{config.API_PREFIX}/openapi.json",
    docs_url=f"{config.API_PREFIX}/docs",
    generate_unique_id_function=generate_operation_id,
    lifespan=lifespan,
)


def _get_allowed_origins() -> list[str]:
    return config.ALLOWED_ORIGINS.get(config.ENVIRONMENT, config.ALLOWED_ORIGINS["development"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix=config.API_PREFIX)


@app.middleware("http")
async def log_request(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    url_path = str(request.url.path)
    noisy_patterns = ["/api/health/", "/api/health"]
    if not any(url_path.endswith(pattern) for pattern in noisy_patterns) and request.method != "OPTIONS":
        logger.info(f"Request: {request.method} {request.url}")
    return await call_next(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    uvicorn.run(
        "samplespace.app:app",
        host="127.0.0.1",
        port=args.port,
        reload=args.reload,
    )
