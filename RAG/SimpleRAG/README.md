# Simple RAG

![RAG Architecture Diagram](https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/diagram_2_mermaid-423723682-light-mermaid.svg)

A clean, production-ready RAG (Retrieval-Augmented Generation) implementation that's simple to use and easy to understand.

## What is RAG?

RAG combines three main pieces:

- **Embedding Model**: Converts text into vectors (numbers that represent meaning) so we can find similar text
- **Vector Database**: Stores text and its vectors so we can search through knowledge quickly  
- **Chatbot**: Generates responses using the retrieved knowledge

## 🚀 Quick Start

```bash
# Activate your virtual environment
source venv/bin/activate

# Run the example
python example.py
```

## 📁 Project Structure

```
SimpleRAG/
├── simple_rag.py      # Main RAG class
├── config.py          # Configuration and settings
├── example.py         # Interactive example
├── test_simple_rag.py # Basic tests
├── cat-facts.txt      # Sample knowledge base
└── README.md          # This file
```

## 💡 Usage

```python
from simple_rag import SimpleRAG

# Initialize
rag = SimpleRAG()

# Add knowledge
rag.add_knowledge("Your fact here")
rag.add_knowledge_from_file("your_file.txt")

# Ask questions
response = rag.query("Your question here")
print(response)
```

## 🔧 Features

- **Simple API**: Easy to use with just a few lines of code
- **File Loading**: Load knowledge from text files automatically
- **Streaming**: Get responses in real-time
- **Configurable**: Easy to change models and settings
- **Error Handling**: Robust error handling and validation
- **Statistics**: Monitor your system's performance

## 📚 Learn More

For a complete tutorial on building RAG from scratch, check out this excellent resource:

**[Make Your Own RAG from Scratch](https://huggingface.co/blog/ngxson/make-your-own-rag)**

## 🧪 Testing

```bash
python test_simple_rag.py
```

That's it! Simple but complete.