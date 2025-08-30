import ollama
import numpy as np
from typing import List, Tuple, Dict, Any
import json
import os

class SimpleRAG:
    def __init__(self, embedding_model: str = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf', 
                 language_model: str = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'):
        """
        Initialize the SimpleRAG system
        
        Args:
            embedding_model: Model name for generating embeddings
            language_model: Model name for generating responses
        """
        self.embedding_model = embedding_model
        self.language_model = language_model
        self.vector_db = []
        self.chunk_count = 0
        
    def add_knowledge(self, text: str) -> None:
        """
        Add a piece of knowledge to the vector database
        
        Args:
            text: Text chunk to add
        """
        if not text.strip():
            return
            
        embedding = self._get_embedding(text)
        self.vector_db.append({
            'id': self.chunk_count,
            'text': text.strip(),
            'embedding': embedding,
            'metadata': {'source': 'manual_input'}
        })
        self.chunk_count += 1
        
    def add_knowledge_from_file(self, file_path: str) -> int:
        """
        Add knowledge from a text file (one chunk per line)
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Number of chunks added
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        chunks_added = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    self.add_knowledge(line)
                    chunks_added += 1
                    
        return chunks_added
        
    def query(self, question: str, top_k: int = 3, stream: bool = False) -> Any:
        """
        Query the RAG system
        
        Args:
            question: User's question
            top_k: Number of top relevant chunks to retrieve
            stream: Whether to stream the response
            
        Returns:
            Response from the language model
        """
        if not self.vector_db:
            return "No knowledge available. Please add some knowledge first."
            
        # Retrieve relevant chunks
        relevant_chunks = self._retrieve(question, top_k)
        
        # Generate response
        if stream:
            return self._generate_response_stream(question, relevant_chunks)
        else:
            return self._generate_response(question, relevant_chunks)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for given text"""
        try:
            response = ollama.embed(model=self.embedding_model, input=text)
            return response['embeddings'][0]
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding: {e}")
    
    def _retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks"""
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for chunk in self.vector_db:
            similarity = self._cosine_similarity(query_embedding, chunk['embedding'])
            similarities.append((chunk, similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in similarities[:top_k]]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _generate_response(self, question: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate response using retrieved chunks"""
        context = '\n'.join([f"- {chunk['text']}" for chunk in relevant_chunks])
        
        prompt = f"""You are a helpful chatbot. Use only the following context to answer the question. Don't make up information:

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = ollama.chat(
                model=self.language_model,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant that answers questions based on provided context.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _generate_response_stream(self, question: str, relevant_chunks: List[Dict[str, Any]]):
        """Generate streaming response"""
        context = '\n'.join([f"- {chunk['text']}" for chunk in relevant_chunks])
        
        prompt = f"""You are a helpful chatbot. Use only the following context to answer the question. Don't make up information:

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            return ollama.chat(
                model=self.language_model,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant that answers questions based on provided context.'},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True
            )
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_chunks': len(self.vector_db),
            'embedding_model': self.embedding_model,
            'language_model': self.language_model,
            'database_size_mb': sum(len(str(chunk)) for chunk in self.vector_db) / (1024 * 1024)
        }
    
    def clear_database(self) -> None:
        """Clear all knowledge from the database"""
        self.vector_db.clear()
        self.chunk_count = 0
