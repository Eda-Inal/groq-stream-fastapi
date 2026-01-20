from fastapi  import FastAPI
from app.api.v1.router import router as api_v1_router

app = FastAPI()

app.include_router(api_v1_router)

@app.get("/health")
async def health_check() -> dict[str,str]:
    return {"status": "ok"}