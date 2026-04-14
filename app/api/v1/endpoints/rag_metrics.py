from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.document import count_document_chunks, count_documents
from app.db.session import get_db
from app.services.mcp.remote_client import RemoteMCPClient

router = APIRouter(prefix="/rag", tags=["rag"])

# Module-level client — metrics are recorded inside the MCP process,
# so we fetch them from MCP instead of reading a local (always-zero) singleton.
_mcp_client = RemoteMCPClient()


@router.get("/metrics")
async def get_rag_metrics(db: AsyncSession = Depends(get_db)) -> dict[str, float | int]:
    total_documents = await count_documents(db)
    total_chunks = await count_document_chunks(db)
    mcp_metrics = await _mcp_client.get_metrics()
    return {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        **mcp_metrics,
    }
