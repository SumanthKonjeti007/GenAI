# from __future__ import annotations as _annotations

# from dataclasses import dataclass
# from dotenv import load_dotenv
# import logfire
# import asyncio
# import httpx
# import os

# from pydantic_ai import Agent, ModelRetry, RunContext
# from pydantic_ai.models.openai import OpenAIModel
# from openai import AsyncOpenAI
# from supabase import Client
# from typing import List

# load_dotenv()

# llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
# model = OpenAIModel(llm)

# logfire.configure(send_to_logfire='if-token-present')

# @dataclass
# class PydanticAIDeps:
#     supabase: Client
#     openai_client: AsyncOpenAI

# system_prompt = """
# You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
# including examples, an API reference, and other resources to help you build Pydantic AI agents.

# Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

# Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

# When you first look at the documentation, always start with RAG.
# Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

# Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
# """

# pydantic_ai_expert = Agent(
#     model,
#     system_prompt=system_prompt,
#     deps_type=PydanticAIDeps,
#     retries=2
# )

# async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
#     """Get embedding vector from OpenAI."""
#     try:
#         response = await openai_client.embeddings.create(
#             model="text-embedding-3-small",
#             input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         return [0] * 1536  # Return zero vector on error

# @pydantic_ai_expert.tool
# async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
#     """
#     Retrieve relevant documentation chunks based on the query with RAG.
    
#     Args:
#         ctx: The context including the Supabase client and OpenAI client
#         user_query: The user's question or query
        
#     Returns:
#         A formatted string containing the top 5 most relevant documentation chunks
#     """
#     try:
#         # Get the embedding for the query
#         query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
#         # Query Supabase for relevant documents
#         result = ctx.deps.supabase.rpc(
#             'match_site_pages',
#             {
#                 'query_embedding': query_embedding,
#                 'match_count': 5,
#                 'filter': {'source': 'pydantic_ai_docs'}
#             }
#         ).execute()
        
#         if not result.data:
#             return "No relevant documentation found."
            
#         # Format the results
#         formatted_chunks = []
#         for doc in result.data:
#             chunk_text = f"""
# # {doc['title']}

# {doc['content']}
# """
#             formatted_chunks.append(chunk_text)
            
#         # Join all chunks with a separator
#         return "\n\n---\n\n".join(formatted_chunks)
        
#     except Exception as e:
#         print(f"Error retrieving documentation: {e}")
#         return f"Error retrieving documentation: {str(e)}"

# @pydantic_ai_expert.tool
# async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
#     """
#     Retrieve a list of all available Pydantic AI documentation pages.
    
#     Returns:
#         List[str]: List of unique URLs for all documentation pages
#     """
#     try:
#         # Query Supabase for unique URLs where source is pydantic_ai_docs
#         result = ctx.deps.supabase.from_('site_pages') \
#             .select('url') \
#             .eq('metadata->>source', 'pydantic_ai_docs') \
#             .execute()
        
#         if not result.data:
#             return []
            
#         # Extract unique URLs
#         urls = sorted(set(doc['url'] for doc in result.data))
#         return urls
        
#     except Exception as e:
#         print(f"Error retrieving documentation pages: {e}")
#         return []

# @pydantic_ai_expert.tool
# async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
#     """
#     Retrieve the full content of a specific documentation page by combining all its chunks.
    
#     Args:
#         ctx: The context including the Supabase client
#         url: The URL of the page to retrieve
        
#     Returns:
#         str: The complete page content with all chunks combined in order
#     """
#     try:
#         # Query Supabase for all chunks of this URL, ordered by chunk_number
#         result = ctx.deps.supabase.from_('site_pages') \
#             .select('title, content, chunk_number') \
#             .eq('url', url) \
#             .eq('metadata->>source', 'pydantic_ai_docs') \
#             .order('chunk_number') \
#             .execute()
        
#         if not result.data:
#             return f"No content found for URL: {url}"
            
#         # Format the page with its title and all chunks
#         page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
#         formatted_content = [f"# {page_title}\n"]
        
#         # Add each chunk's content
#         for chunk in result.data:
#             formatted_content.append(chunk['content'])
            
#         # Join everything together
#         return "\n\n".join(formatted_content)
        
#     except Exception as e:
#         print(f"Error retrieving page content: {e}")
#         return f"Error retrieving page content: {str(e)}"


# =============================================================================
# Agentic RAG – Quick Notes (for this project)
# =============================================================================
# What is “Agentic RAG” here?
# - LLM + multiple tools + a short policy in the system prompt.
# - The LLM can PLAN → CALL a tool → READ its result → DECIDE next step → repeat.
# - You do NOT code the loop; the Agent runtime handles tool calls as the LLM requests them.
#
# Minimum pieces you already have:
# 1) LLM: OpenAIModel(llm) pointed at Ollama via env (OPENAI_BASE_URL=/v1, OPENAI_API_KEY).
# 2) Tools (with clear docstrings that explain what/when/returns):
#    - retrieve_relevant_documentation(query): semantic top-k chunks (basic RAG).
#    - list_documentation_pages(): list all doc URLs to widen search space.
#    - get_page_content(url): fetch full page (ordered chunks) to deep-read.
# 3) System prompt (the “policy”): a short playbook that tells the LLM how to use tools.
#
# Why docstrings matter:
# - The LLM reads tool docstrings to decide WHICH tool to use and WHEN.
# - Keep them action-oriented: “Use this first…”, “Use when initial retrieval is weak…”, etc.
#
# A tiny, effective policy to put in system_prompt (copy/paste & tweak):
# """
# You are a Pydantic AI documentation agent.
# Policy:
# 1) Start with retrieve_relevant_documentation(<user_query>).
# 2) If results look generic/irrelevant/missing code, call list_documentation_pages().
# 3) Pick 1–3 likely URLs and call get_page_content(url) to read full context.
# 4) If still uncertain, repeat 2–3 once with different candidates.
# 5) Never guess. If evidence is insufficient, say what you tried and what’s missing.
# 6) In the final answer, cite the URLs you actually used.
# """
#
# “Basic RAG” vs “Agentic RAG”:
# - Basic RAG: one shot → vector search → answer from top-k only.
# - Agentic RAG: multi-step → the LLM can call more tools, read more pages, then answer.
# - Note: retries=2 only retries failed model calls; it doesn’t make RAG “agentic”.
#
# When the LLM should loop (typical signals):
# - Top-k is empty or weak (“No relevant…” / vague text).
# - The question references exact API names or errors not in the chunks.
# - The answer requires info spread across multiple pages/sections.
#
# Optional upgrades (later):
# - Add a keyword_search tool (SQL ILIKE / full-text) for exact strings (API names, error codes).
# - Return JSON with similarity scores from retrieve_relevant_documentation so the LLM can judge quality.
# - Add a tiny “assess_answer” step (LLM self-check) to decide if more context is needed.
#
# Ollama + embeddings quick ref:
# - Chat model (via OpenAI compat): LLM_MODEL=llama3.2:1b
#   OPENAI_BASE_URL=http://localhost:11434/v1, OPENAI_API_KEY=ollama (any non-empty string)
# - Embeddings (direct Ollama): EMBED_MODEL=nomic-embed-text (768-dim)
# - DB: site_pages.embedding is vector(768); keep EMBED_SIZE=768 in code for zero-vector fallback.
# - Make sure `ollama serve` is running and models are pulled.
#
# TL;DR:
# - Tools + clear docstrings + short policy in system prompt = agentic behavior.
# - The Agent framework runs the plan-act-observe loop; you don’t hand-code it.
# - If the first retrieve is weak, the LLM will use your other tools to gather better context.

#Simple mental model
#RAG = “bring me the right pages.”
#LLM = “read them, understand what I need, decide if I need more, combine them, and produce the best possible answer.”

#Your data work (crawl → chunk → embed → metadata) ensures the LLM is grounded (no hallucinations).
#Your tools give the LLM capabilities (retrieve, list pages, get full page).
#Your system prompt + docstrings tell it how to think and act (start with RAG, widen if weak, deep-read, then answer honestly).
#The agent runtime runs the loop; the LLM makes the choices and synthesizes the final answer.

#MCP (Model Context Protocol) “lives” in the tools layer.
#An MCP server exposes tools/resources (e.g., wikipedia.search, wikipedia.page) and your agent can call them exactly like your current tools (retrieve_relevant_documentation, get_page_content, etc.). 
# So for “get data from Wiki,” you’d add a Wiki MCP tool and let the LLM decide when to use it.

#How it fits your Agentic RAG
# 1. LLM = the brain (decides what to do next)
# 2. RAG tools = your internal knowledge (Supabase vector search, full-page fetch)
# 3. MCP tools = external knowledge (e.g., Wikipedia, issue trackers, repos, calendars, crawl APIs)
# 4. System prompt = tiny policy telling the LLM when to use which tool
# =============================================================================

# Qdrant Database is for basic RAG as it stores only embeddings (Just a Vector DB) + a little metadata. 
# Supabase (Postgres + pgvector) is for Agentic RAG when you stores the full page content, titles, summaries, and metadata. -> a structured data (full page content, titles, summaries, and metadata) + embeddings + metadata.

from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import json
from typing import List

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client

load_dotenv()

# ==== Chat model via Ollama (OpenAI-compatible endpoint) ====
# .env should have:
#   OPENAI_BASE_URL=http://localhost:11434/v1
#   OPENAI_API_KEY=ollama
#   LLM_MODEL=llama3.2:1b
llm = os.getenv("LLM_MODEL", "llama3.2:1b")
model = OpenAIModel(llm)

logfire.configure(send_to_logfire="if-token-present")

# ==== Only Supabase in deps now ====
@dataclass
class PydanticAIDeps:
    supabase: Client

# ==== Embedding settings (Ollama) ====
# .env should have:
#   EMBED_MODEL=nomic-embed-text
#   EMBED_SIZE=768
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_SIZE = int(os.getenv("EMBED_SIZE", "768"))

system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2,
)

# ------ Embeddings with Ollama (keeps async signature & zero-vector fallback) ------
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    try:
        import ollama
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)  # sync call is fine here
        vec = resp.get("embedding", [])
        return vec if vec else [0.0] * EMBED_SIZE
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * EMBED_SIZE  # dimension matches your DB (768 after migration)

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query (no openai_client anymore)
        query_embedding = await get_embedding(user_query)

        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            "match_site_pages",
            {
                "query_embedding": query_embedding,
                "match_count": 5,
                "filter": {"source": "pydantic_ai_docs"},
            },
        ).execute()

        if not result.data:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.

    Args:
        ctx: The context including the Supabase client.

    Returns:
        List[str]: Sorted list of unique URLs for all documentation pages.
    """
    try:
        result = (
            ctx.deps.supabase.from_("site_pages")
            .select("url")
            .eq("metadata->>source", "pydantic_ai_docs")
            .execute()
        )

        if not result.data:
            return []

        urls = sorted(set(doc["url"] for doc in result.data))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client.
        url: The URL of the page to retrieve.

    Returns:
        str: The complete page content with all chunks combined in order.
    """
    try:
        result = (
            ctx.deps.supabase.from_("site_pages")
            .select("title, content, chunk_number")
            .eq("url", url)
            .eq("metadata->>source", "pydantic_ai_docs")
            .order("chunk_number")
            .execute()
        )

        if not result.data:
            return f"No content found for URL: {url}"

        page_title = result.data[0]["title"].split(" - ")[0]
        formatted_content = [f"# {page_title}\n"]

        for chunk in result.data:
            formatted_content.append(chunk["content"])

        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

