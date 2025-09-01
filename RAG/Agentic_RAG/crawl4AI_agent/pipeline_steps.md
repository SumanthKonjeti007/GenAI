# RAG Data Ingestion Pipeline

## Simple Steps:

1. **Fetch URLs** from sitemap
2. **Crawl websites** in parallel
3. **Extract markdown** content
4. **Split into chunks**
5. **Process each chunk** (parallel):
   - Generate title & summary (GPT-4)
   - Create embedding vector
   - Add metadata
6. **Store in database** (Supabase)

## Key Features:
- **Parallel processing** for speed
- **AI enrichment** with GPT-4
- **Vector embeddings** for similarity search
- **Structured storage** for RAG retrieval
