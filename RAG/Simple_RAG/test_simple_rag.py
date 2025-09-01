#!/usr/bin/env python3
"""
Simple test for the SimpleRAG system
"""

from simple_rag import SimpleRAG

def test_basic_functionality():
    """Test basic RAG functionality"""
    print("🧪 Testing SimpleRAG...")
    
    # Initialize
    rag = SimpleRAG()
    
    # Add knowledge
    test_knowledge = [
        "Python is a programming language.",
        "Python was created by Guido van Rossum.",
        "Python is known for its simplicity and readability."
    ]
    
    for fact in test_knowledge:
        rag.add_knowledge(fact)
    
    print(f"✅ Added {len(test_knowledge)} test facts")
    
    # Test query
    question = "Who created Python?"
    print(f"❓ Testing question: {question}")
    
    try:
        response = rag.query(question, top_k=2)
        print(f"💬 Response: {response}")
        print("✅ Test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test stats
    stats = rag.get_stats()
    print(f"📊 Stats: {stats['total_chunks']} chunks in database")

if __name__ == "__main__":
    test_basic_functionality()
