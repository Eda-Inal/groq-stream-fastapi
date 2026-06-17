from fastapi import FastAPI

from app.tool_server.routes import router

app = FastAPI(title="Tool Server")

app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# python -m app.tool_server.main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.tool_server.main:app", host="0.0.0.0", port=8001, reload=True)
