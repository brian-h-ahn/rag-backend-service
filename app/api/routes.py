from fastapi import APIRouter
from app.schemas.dto import IngestRequest, QueryRequest
from app.rag.pipeline import RagPipeline
from app.core.config import settings

router = APIRouter()
rag = RagPipeline()

@router.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}

@router.post("/ingest")
def ingest(req: IngestRequest):
    return rag.ingest_text(req.text, source=req.source)

@router.post("/query")
def query(req: QueryRequest):
    top_k = req.top_k or settings.top_k
    return rag.query(req.question, top_k=top_k)
