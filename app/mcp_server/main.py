from fastapi import FastAPI

from app.mcp_server.routes import router

app = FastAPI(title="MCP Server")

app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
