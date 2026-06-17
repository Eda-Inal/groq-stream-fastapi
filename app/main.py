"""
FastAPI application entry point.

This module initializes the FastAPI app, registers API routers,
and exposes a basic health check endpoint.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api.v1.router import router as api_v1_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.models.document_chunk import DocumentChunk
from app.db.session import AsyncSessionLocal

setup_logging()

logger = structlog.get_logger()


async def _check_embedding_dim() -> None:
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DocumentChunk.embedding).limit(1)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return
            actual_dim = len(row)
            if actual_dim != settings.embedding_dim:
                logger.warning(
                    "embedding_dim_mismatch",
                    db_dim=actual_dim,
                    config_dim=settings.embedding_dim,
                    action_required=(
                        f"DB chunks have {actual_dim}-dim embeddings but "
                        f"EMBEDDING_DIM={settings.embedding_dim} in config. "
                        "Run an Alembic migration to alter the vector column, "
                        "then reprocess all documents."
                    ),
                )
    except Exception:
        logger.warning("embedding_dim_check_failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _check_embedding_dim()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        dict[str, str]: Simple status response used to verify
        that the application is running.
    """
    return {"status": "ok"}


# python -m app.main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)