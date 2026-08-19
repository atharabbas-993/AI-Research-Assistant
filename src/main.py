from embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator()

question = "What optimizer was used for training?"
question_embedding = embedder.embed_query(question)

print("Question embedding length:", len(question_embedding))
print("First few numbers:", question_embedding[:5])