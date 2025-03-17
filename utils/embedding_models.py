import ollama


# Initialize ollama provider
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'


def get_embedding(chunk):
    """Get embedding for the provided text"""
    return ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]


def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)


def retrieve_n_closest_vectors(query, VECTOR_DB, top_n=3):
  # TODO: we get it as text in the first place, so translate it first.
  query_embedding = get_embedding(query)
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]