import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langgraph import StateGraph, tool, ToolNode, ToolMessage
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from brave_search import BraveSearch

load_dotenv()
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
brave_key = os.getenv("BRAVE_API_KEY")

def preprocess_dataset(docs_list):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=50,
        disallowed_special=()
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits

hugging_face_doc = HuggingFaceDatasetLoader("m-ric/huggingface_doc","text")
transformers_doc = HuggingFaceDatasetLoader("m-ric/transformers_documentation_en","text")

hf_splits = preprocess_dataset(hugging_face_doc.load()[:number_of_docs])
transformer_splits = preprocess_dataset(transformers_doc.load()[:number_of_docs])

class State(TypedDict):
    messages: Annotated[list, add_messages]

def create_retriever(collection_name, doc_splits):
    vectorstore = QdrantVectorStore.from_documents(
        doc_splits,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        url=qdrant_url,
        api_key=qdrant_key,
        collection_name=collection_name,
    )
    return vectorstore.as_retriever()

hf_retriever_tool = create_retriever_tool(
    hf_retriever,
    "retriever_hugging_face_documentation",
    "Search and return information about hugging face documentation, it includes the guide and Python code.",
)

transformer_retriever_tool = create_retriever_tool(
    transformer_retriever,
    "retriever_transformer",
    "Search and return information specifically about transformers library",
)

@tool("web_search_tool")
def search_tool(query):
    search = BraveSearch.from_api_key(api_key=brave_key, search_kwargs={"count": 3})
    return search.run(query)

tools = [hf_retriever_tool, transformer_retriever_tool, search_tool]

tool_node = ToolNode(tools=tools)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

llm_with_tools = llm.bind_tools(tools)

class ToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": outputs}

def route(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    return END

graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "agent",
    route,
    {"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "agent")
graph_builder.add_edge(START, "agent")

def run_agent(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)