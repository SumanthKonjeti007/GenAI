"""
Configuration file for SimpleRAG system
"""

# Model configurations
EMBEDDING_MODELS = {
    'bge-base': 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf',
    'bge-large': 'hf.co/CompendiumLabs/bge-large-en-v1.5-gguf',
    'all-minilm': 'hf.co/CompendiumLabs/all-MiniLM-L6-v2-gguf'
}

LANGUAGE_MODELS = {
    'llama-3.2-1b': 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF',
    'llama-3.2-3b': 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF',
    'gemma-2b': 'hf.co/bartowski/gemma-2b-it-gguf',
    'mistral-7b': 'hf.co/bartowski/Mistral-7B-Instruct-v0.2-GGUF'
}

# Default models
DEFAULT_EMBEDDING_MODEL = 'bge-base'
DEFAULT_LANGUAGE_MODEL = 'llama-3.2-1b'

# RAG settings
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200

# Database settings
MAX_DATABASE_SIZE_MB = 1000  # Maximum database size in MB

# Prompt templates
SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on provided context."
USER_PROMPT_TEMPLATE = """Use only the following context to answer the question. Don't make up information:

Context:
{context}

Question: {question}

Answer:"""
