# Qdrant Agent (LangGraph) – README

This folder contains **`QdrantAgent.py`**, a minimal **agentic RAG** example built with **LangGraph** + **LangChain** using **Qdrant** as the vector database.  
The agent retrieves relevant context from Qdrant, (optionally) calls tools, and then generates an answer with short citations.

---

## What is Qdrant?

**Qdrant** is an open-source **vector database** (written in Rust) designed for fast **similarity search** over embeddings.  
You store “points” = **vector(s) + metadata (payload)** and query the nearest vectors using cosine/dot/Euclidean distance.  
It supports persistent storage, filtering on metadata, REST/gRPC APIs, and runs locally (Docker) or as a managed cloud service.

**Typical uses**
- RAG (retrieve top-K relevant chunks for an LLM)
- Image/audio/text semantic search
- Hybrid search (dense + metadata filters)

---

## How this project works (high level)

1. **Ingest**: Split documents → create embeddings → **upsert** to Qdrant (vector + payload like `source`, `chunk_id`).
2. **Query**: For a user question, create a query embedding → **search** Qdrant for top-K chunks.
3. **Answer**: LangGraph node(s) build a prompt with those chunks → LLM generates a grounded answer.  
   If you add tools (e.g., calculator/web fetch), an agent step can decide to call them before answering.

---

## Quickstart

### 1) Requirements
- Python 3.10+
- Qdrant running **locally** (Docker) or Qdrant Cloud URL/API key
- An embedding model (OpenAI, or local via Ollama), and an LLM (OpenAI/Ollama)

### 2) Run Qdrant locally
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
