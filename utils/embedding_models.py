import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_providers.ollama_provider import OllamaProvider

# Initialize ollama provider
ollama_client = OllamaProvider()
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

def get_embedding(text):
    """Get embedding for the provided text"""
    return ollama_client.embed(text, model=EMBEDDING_MODEL)

def add_chunk_to_database(VECTOR_DB, chunk):
    embedding = get_embedding(chunk)
    VECTOR_DB.append((chunk, embedding))

