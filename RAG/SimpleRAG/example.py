#!/usr/bin/env python3
"""
Simple example demonstrating the SimpleRAG system
"""

from simple_rag import SimpleRAG
import os

def main():
    # Initialize the RAG system
    print("🚀 Initializing SimpleRAG...")
    rag = SimpleRAG()
    
    # Check if cat-facts.txt exists
    if os.path.exists('cat-facts.txt'):
        print("📚 Loading knowledge from cat-facts.txt...")
        chunks_added = rag.add_knowledge_from_file('cat-facts.txt')
        print(f"✅ Added {chunks_added} chunks to the database")
    else:
        print("⚠️  cat-facts.txt not found. Adding some sample knowledge...")
        sample_knowledge = [
            "Cats are mammals that belong to the Felidae family.",
            "Domestic cats have been living with humans for over 10,000 years.",
            "Cats have excellent night vision and can see in near darkness.",
            "A group of cats is called a clowder.",
            "Cats spend 70% of their lives sleeping."
        ]
        for fact in sample_knowledge:
            rag.add_knowledge(fact)
        print(f"✅ Added {len(sample_knowledge)} sample facts")
    
    # Show system stats
    stats = rag.get_stats()
    print(f"\n📊 System Stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Embedding model: {stats['embedding_model']}")
    print(f"   Language model: {stats['language_model']}")
    
    # Interactive Q&A
    print("\n🤔 Ask me anything about cats! (type 'quit' to exit)")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
                
            print("🔍 Searching for relevant information...")
            
            # Get response (streaming)
            response_stream = rag.query(question, top_k=3, stream=True)
            
            print("💬 Answer:")
            if isinstance(response_stream, str):
                print(response_stream)
            else:
                # Handle streaming response
                for chunk in response_stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        print(chunk['message']['content'], end='', flush=True)
                print()  # New line after streaming
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
