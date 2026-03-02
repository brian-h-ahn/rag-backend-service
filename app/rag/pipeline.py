import os
import time
import logging
from app.core.config import settings
from app.rag.chunking import chunk_text
from app.rag.embeddings import Embedder
from app.rag.vectorstore import FaissStore

logger = logging.getLogger("rag")

class RagPipeline:
    def __init__(self):
        self.embedder = Embedder(settings.embedding_model)

        # we can choose model, thus check the model to get the dim
        probe = self.embedder.encode(["probe"])
        dim = probe.shape[1]

        index_path = os.path.join(settings.index_dir, "faiss.index")
        meta_path = os.path.join(settings.index_dir, "meta.json")
        self.store = FaissStore(index_path=index_path, meta_path=meta_path, dim=dim)

    def ingest_text(self, text: str, source: str):
        t0 = time.perf_counter()

        chunks = chunk_text(text)
        if not chunks:
            return {"chunks": 0, "timings_ms": {"total": 0}}
        t1 = time.perf_counter()

        vectors = self.embedder.encode(chunks)
        t2 = time.perf_counter()

        metadatas = [{"source": source, "chunk": c} for c in chunks]
        self.store.add(vectors, metadatas)
        self.store.persist()
        t3 = time.perf_counter()

        timings = {
            "chunk_ms": int((t1 - t0) * 1000), "embed_ms": int((t2 - t1) * 1000),
            "store_ms": int((t3 - t2) * 1000), "total_ms": int((t3 - t0) * 1000),
        }

        return {"chunks": len(chunks), "timings_ms": timings}

    def query(self, question: str, top_k: int):
        t0 = time.perf_counter()

        qvec = self.embedder.encode([question])
        t1 = time.perf_counter()

        hits = self.store.search(qvec, top_k=top_k)
        t2 = time.perf_counter()

        evidence = [h["metadata"]["chunk"] for h in hits]
        answer = self._simple_answer(question, evidence)
        t3 = time.perf_counter()

        timings = {
            "embed_ms": int((t1 - t0) * 1000), "search_ms": int((t2 - t1) * 1000),
            "assemble_ms": int((t3 - t2) * 1000), "total_ms": int((t3 - t0) * 1000),
        }

        return {
            "question": question,
            "answer": answer,
            "hits": hits,
            "timings_ms": timings,
        }

    def _simple_answer(self, question: str, evidence: list[str]) -> str:
        if not evidence:
            return "No evidence. Try to ask differently."
        # Simple baseline: to be seen as summary using top evidences
        joined = "\n\n---\n\n".join(e[:500] for e in evidence)
        return f"Q: {question}\n\nCompose the answer based on below evidences:\n\n{joined}"