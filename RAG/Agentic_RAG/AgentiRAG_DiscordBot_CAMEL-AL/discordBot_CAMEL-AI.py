import os
import json
import nest_asyncio
import discord
from dotenv import load_dotenv

# CAMEL-AI
from camel.embeddings import OpenAIEmbedding
from camel.types import EmbeddingModelType, ModelPlatformType, ModelType, RoleType
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.bots import DiscordApp
from camel.messages import BaseMessage           # <-- use BaseMessage (not ChatMessage)
from camel.storages.qdrant_storage import QdrantStorage
from camel.retrievers import VectorRetriever

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")  # e.g., http://localhost:6333 for local
QDRANT_CLOUD_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY") or os.getenv("QDRANT_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Embeddings
embedding_instance = OpenAIEmbedding(
    model_type=EmbeddingModelType.TEXT_EMBEDDING_3_LARGE,
    api_key=OPENAI_API_KEY,
)

# Qdrant storage
collection_name = "qdrant-agent"
storage_instance = QdrantStorage(
    vector_dim=embedding_instance.get_output_dim(),
    url_and_api_key=(QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY),
    collection_name=collection_name,
)

# Retriever
vector_retriever = VectorRetriever(
    embedding_model=embedding_instance,
    storage=storage_instance,
)

# (Optional) seed URLs once
qdrant_urls = [
    "https://qdrant.tech/documentation/overview",
    "https://qdrant.tech/documentation/guides/installation",
    "https://qdrant.tech/documentation/concepts/filtering",
    "https://qdrant.tech/documentation/concepts/indexing",
    "https://qdrant.tech/documentation/guides/distributed_deployment",
    "https://qdrant.tech/documentation/guides/quantization",
]
for u in qdrant_urls:
    vector_retriever.process(content=u)

# LLM model
config = ChatGPTConfig(temperature=0.2).as_dict()
openai_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=config,
)

assistant_sys_msg = (
    "You are a helpful assistant. You will be given an Original Query and a "
    "Retrieved Context. Answer ONLY using the Retrieved Context. "
    "If the context is insufficient, reply: 'I don't know.'"
)
qdrant_agent = ChatAgent(system_message=assistant_sys_msg, model=openai_model)

nest_asyncio.apply()
discord_q_bot = DiscordApp(token=DISCORD_BOT_TOKEN)

def _format_context(retrieved) -> str:
    if isinstance(retrieved, str):
        return retrieved
    if isinstance(retrieved, list):
        parts = []
        for item in retrieved:
            if isinstance(item, dict):
                parts.append(
                    item.get("content")
                    or item.get("text")
                    or json.dumps(item, ensure_ascii=False)
                )
            else:
                parts.append(str(item))
        return "\n\n".join(p for p in parts if p)
    return str(retrieved)

@discord_q_bot.client.event
async def on_message(message: discord.Message):
    if message.author == discord_q_bot.client.user:
        return
    if message.type != discord.MessageType.default:
        return
    if message.author.bot:
        return

    user_input = message.content.strip()
    if not user_input:
        return

    try:
        retrieved_info = vector_retriever.query(
            query=user_input, top_k=6, similarity_threshold=0.6
        )
        context_text = _format_context(retrieved_info)

        prompt = (
            f"Original Query:\n{user_input}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            "Final Answer:"
        )

        # 👇 Use BaseMessage here
        user_msg = BaseMessage(
            role_name="User",
            role_type=RoleType.USER,
            meta_dict=None,
            content=prompt,
        )

        step_out = qdrant_agent.step(user_msg)
        reply = step_out.msgs[0].content if step_out.msgs else "I don't know."

        if len(reply) > 2000:
            for i in range(0, len(reply), 2000):
                await message.channel.send(reply[i:i+2000])
        else:
            await message.channel.send(reply)

    except Exception as e:
        await message.channel.send(f"⚠️ Error: {e}")

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise RuntimeError("Missing DISCORD_BOT_TOKEN in env.")
    if not QDRANT_CLOUD_URL:
        raise RuntimeError("Missing QDRANT_CLOUD_URL (use http://localhost:6333 for local).")
    discord_q_bot.run()
