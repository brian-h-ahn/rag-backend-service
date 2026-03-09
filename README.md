# RAG Backend Service (FastAPI + FAISS)

A backend system implementing Retrieval-Augmented Generation (RAG) with
document ingestion, embedding pipelines, vector search, and API query
orchestration.

## Tech Stack

- Python 3.12
- FastAPI
- FAISS
- sentence-transformers
- Pydantic
- Uvicorn

---

## Project Structure
```
app/
  main.py
  routes/
  pipeline/
  embeddings/
  storage/

config.py
requirements.txt
```
## Overview
This project is a backend Retrieval-Augmented Generation (RAG) service built with FastAPI. It ingests text, converts it into vector embeddings, stores them in FAISS, and retrieves relevant context using cosine similarity search.

The purpose of this project is to demonstrate backend engineering capability: clean API boundaries, pipeline orchestration, persistence, observability, and production tradeoff awareness.

---

## Architecture

High-level data flow:

Client → FastAPI Routes → Pipeline → Chunking → Embeddings

The pipeline produces two outputs:

- Vector data → stored in FAISS index → persisted to disk
- Metadata (chunk text, source, mapping info) → stored in JSON → persisted to disk

This separation keeps vector search fast while preserving contextual information needed to interpret results.

---

## Features

### 1) Text Ingestion
- Accepts raw text input
- Splits text into overlapping chunks
- Converts chunks into normalized embedding vectors
- Stores vectors in FAISS index
- Persists index and metadata to disk

### 2) Semantic Query
- Converts user question into embedding
- Performs cosine similarity search
- Returns top-k relevant chunks
- Generates lightweight answer from retrieved context

### 3) Observability
- Request ID attached to each HTTP call
- Latency measurement at API boundary
- Timing breakdown inside pipeline (embed/search/total)

### 4) Persistence
- Persists both the FAISS index and metadata to disk
- Restores state on startup so the service can restart without losing the retrieval database
- Treats persistence as part of the backend lifecycle (startup initialization vs runtime requests)

---

## Example Request Flow

1.  Client sends `POST /query`
2.  Question converted to embedding vector
3.  FAISS performs similarity search
4.  Metadata retrieved for contextual chunks
5.  API returns answer, evidence, scores, and timing data

## API Endpoints

### POST /ingest
Ingest text into the vector database.

Request:
```
{
  "text": "string",
  "source": "optional"
}
```

Response:
```
{
  "status": "ok",
  "chunks": 42,
  "timing": { ... }
}
```

---

### POST /query
Query the system using natural language.

Request:
```
{
  "question": "string",
  "top_k": 3
}
```

Response:
```
{
  "answer": "string",
  "evidence": [...],
  "scores": [...],
  "timing": { ... }
}
```

---

## Error Handling

- Request validation is handled with DTOs at the API boundary
- The service returns structured error responses for invalid inputs (e.g., empty text/question)
- Errors are logged with a request ID so failures can be traced end-to-end

## Running the Service

### Requirements
- Python 3.12+
- FastAPI
- SentenceTransformers
- FAISS

### Setup
```
pip install -r requirements.txt
```

### Run
```
uvicorn app.main:app --reload
```

---

## Engineering Decisions

### Why FAISS IndexFlatIP
- Provides exact cosine-similarity search with deterministic results
- Simple to reason about, debug, and validate during early development
- Avoids premature optimization before real scale requirements are known
- Serves as a strong baseline before moving to approximate nearest neighbor approaches (e.g., IVF, HNSW) when dataset size grows

### Why Normalize Embeddings
- Ensures similarity is based on direction rather than magnitude
- Prevents longer vectors from appearing artificially more important

### Why Separate Metadata
- Keeps vector index lightweight
- Enables flexible filtering and document lifecycle management

### Why Pipeline Orchestration
- Keeps API layer thin
- Centralizes retrieval logic
- Makes the system easier to test and extend

---

## Scaling Boundary

- Current retrieval uses exact search (IndexFlatIP), which is a strong baseline and works well for small to medium datasets.
- As the number of vectors grows (typically into the high hundreds of thousands to millions), approximate nearest neighbor indexes (e.g., IVF/HNSW) become important to reduce query latency.

---

## Limitations

- No deduplication during ingest (vectors may repeat)
- Single-node FAISS deployment
- Persistence is not concurrency-safe
- Basic answer synthesis (non-LLM)

These are intentional simplifications to keep the system transparent and focused on retrieval fundamentals.

---

## Production Evolution Path

The system is structured so it can evolve toward production needs such as:

- Idempotent ingest (document IDs + upsert)
- Concurrency-safe persistence
- Background ingestion worker
- Approximate nearest neighbor indexing for scalable retrieval

---

## Interview Talking Points

This project demonstrates:

- API design and service boundaries
- Backend pipeline orchestration
- Vector retrieval systems
- Observability and performance measurement
- Persistence and lifecycle management
- Clear production tradeoff reasoning

