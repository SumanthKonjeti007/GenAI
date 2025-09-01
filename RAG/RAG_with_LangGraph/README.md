# RAG with LangGraph

![RAG with LangGraph Architecture](https://python.langchain.com/img/favicon.ico)

A sophisticated RAG (Retrieval-Augmented Generation) implementation using LangGraph for building stateful, multi-step AI applications with advanced query analysis and structured workflows.

## What is RAG with LangGraph?

This project demonstrates an advanced RAG system that combines:

- **LangGraph**: For building stateful, multi-step AI workflows
- **Query Analysis**: Intelligent query understanding and routing
- **Structured Retrieval**: Context-aware document retrieval
- **Multi-step Generation**: Complex reasoning chains

## 🚀 Quick Start

```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies
pip install langchain langchain-ollama langgraph

# Run the notebook
jupyter notebook rag.ipynb
```

## 📁 Project Structure

```
RAGwithLangGraph/
├── rag.ipynb          # Main RAG with LangGraph implementation
├── .env               # Environment variables (API keys)
└── README.md          # This file
```

## 🔧 Key Components

### 1. **State Management**
```python
class State(TypedDict):
    question: str      # User's question
    query: Search      # Structured query analysis
    context: List[Document]  # Retrieved documents
    answer: str        # Final answer
```

### 2. **Query Analysis**
- **Intelligent Query Understanding**: Analyzes user questions to determine intent
- **Structured Output**: Converts natural language to structured queries
- **Section Filtering**: Routes queries to specific document sections

### 3. **Multi-step Workflow**
```
User Question → Query Analysis → Retrieval → Generation → Answer
```

### 4. **Advanced Features**
- **Streaming**: Real-time response generation
- **Metadata Filtering**: Context-aware document retrieval
- **Structured Outputs**: Type-safe query processing

## 💡 Usage

```python
# Initialize the graph
graph = build_rag_graph()

# Run a query
response = graph.invoke({
    "question": "What does the end of the post say about Task Decomposition?"
})

print(response["answer"])
```

## 🔍 How It Works

### **Step 1: Query Analysis**
```python
def analyze_query(state: State):
    # Convert natural language to structured query
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}
```

### **Step 2: Intelligent Retrieval**
```python
def retrieve(state: State):
    query = state["query"]
    # Retrieve with metadata filtering
    docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"]
    )
    return {"context": docs}
```

### **Step 3: Context-Aware Generation**
```python
def generate(state: State):
    # Generate answer using retrieved context
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    response = llm.invoke(prompt.format(context=docs_content, question=state["question"]))
    return {"answer": response.content}
```

## 🎯 Features

- **🔄 Stateful Workflows**: Maintains context across multiple steps
- **🧠 Query Analysis**: Intelligent understanding of user intent
- **📊 Structured Retrieval**: Metadata-aware document filtering
- **⚡ Streaming**: Real-time response generation
- **🔍 Observability**: Full traceability with LangSmith
- **🎨 Modular Design**: Easy to extend and customize

## 📚 Resources

This implementation is based on:

- **[LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)**: Core RAG concepts and implementation
- **[LangChain QA with Chat History](https://python.langchain.com/docs/tutorials/qa_chat_history/)**: Advanced conversation handling
- **LangGraph Documentation**: Stateful workflow management

## 🚀 Next Steps

- **Conversation Memory**: Add chat history support
- **Multi-step Reasoning**: Implement complex reasoning chains
- **Tool Integration**: Add external tool calling capabilities
- **Custom Retrievers**: Implement specialized retrieval strategies

## 🧪 Testing

```python
# Test the complete workflow
for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates"
):
    print(f"{step}\n\n----------------\n")
```

That's it! Advanced RAG with stateful workflows and intelligent query analysis.
