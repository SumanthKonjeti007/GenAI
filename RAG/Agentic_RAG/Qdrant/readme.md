# Qdrant Agent - README

This folder contains an agentic RAG (Retrieval-Augmented Generation) implementation using **Qdrant** as the vector database, built with **LangGraph** and **LangChain**.

## What is Qdrant?

**Qdrant** is an open-source vector database written in Rust, designed for efficient similarity search over high-dimensional vectors (embeddings). It stores vectors along with metadata (payloads) and enables fast nearest-neighbor searches using distance metrics like cosine similarity, dot product, or Euclidean distance.

### Key Features:
- Fast vector similarity search
- Persistent storage
- Metadata filtering capabilities
- REST and gRPC APIs
- Can run locally (Docker) or as a managed cloud service
- Supports hybrid search (combining dense vectors with sparse keyword search)

## Uses of Qdrant

Qdrant is commonly used for:

1. **RAG Systems**: Storing document embeddings and retrieving relevant chunks for LLM context
2. **Semantic Search**: Finding similar content based on meaning rather than keywords
3. **Recommendation Systems**: Finding similar items or users
4. **Image/Audio Search**: Searching multimedia content by semantic similarity
5. **Anomaly Detection**: Identifying outliers in high-dimensional data

## Qdrant vs Supabase vs Pinecone

### Qdrant
- **Type**: Open-source vector database
- **Deployment**: Self-hosted (Docker) or cloud-managed
- **Cost**: Free for self-hosted, pay-per-use for cloud
- **Best for**: Projects needing full control, cost-effective solutions, or hybrid search
- **Strengths**: Fast performance, flexible filtering, open-source

### Supabase
- **Type**: Open-source Firebase alternative (PostgreSQL-based)
- **Vector Support**: Has pgvector extension for vector search
- **Deployment**: Managed cloud service (also self-hostable)
- **Best for**: Full-stack applications needing both relational data and vector search
- **Strengths**: Combines traditional database features with vector capabilities, real-time features

### Pinecone
- **Type**: Managed vector database service
- **Deployment**: Cloud-only (fully managed)
- **Cost**: Pay-per-use pricing model
- **Best for**: Production applications requiring minimal setup and maintenance
- **Strengths**: Easy to use, scalable, no infrastructure management

### Key Differences:

| Feature | Qdrant | Supabase | Pinecone |
|---------|--------|----------|----------|
| Open Source | Yes | Yes | No |
| Self-Hosted | Yes | Yes | No |
| Vector-Only | Yes | No (PostgreSQL + vectors) | Yes |
| Setup Complexity | Medium | Medium | Low |
| Cost (Self-Hosted) | Free | Free | N/A |
| Best Use Case | Vector-focused apps | Full-stack apps | Production RAG |

## Project Overview

This implementation (`QdrantAgent.py`) demonstrates:

- Loading documents from HuggingFace datasets
- Splitting documents into chunks
- Creating embeddings using OpenAI
- Storing vectors in Qdrant
- Building a LangGraph agent with retrieval tools
- Combining multiple retrievers (HuggingFace docs, Transformers docs)
- Web search integration via Brave Search API

## Setup

1. Install dependencies:
```bash
pip install qdrant-client langchain langgraph openai python-dotenv
```

2. Set up environment variables (`.env` file):
```
QDRANT_URL=your_qdrant_url
QDRANT_KEY=your_qdrant_api_key
BRAVE_API_KEY=your_brave_api_key
OPENAI_API_KEY=your_openai_api_key
```

3. Run Qdrant locally (optional):
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

4. Run the agent:
```bash
python QdrantAgent.py
```

## How It Works

1. **Document Processing**: Loads documentation from HuggingFace datasets and splits them into chunks
2. **Embedding Creation**: Generates embeddings using OpenAI's embedding model
3. **Vector Storage**: Stores embeddings and metadata in Qdrant collections
4. **Retrieval**: Uses Qdrant to find relevant document chunks based on query similarity
5. **Agent Execution**: LangGraph orchestrates the agent to use retrieval tools and generate responses
