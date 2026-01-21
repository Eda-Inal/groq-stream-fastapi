"""
FastAPI application entry point.

This module initializes the FastAPI app, registers API routers,
and exposes a basic health check endpoint.
"""

from fastapi  import FastAPI
from app.api.v1.router import router as api_v1_router

app = FastAPI()

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