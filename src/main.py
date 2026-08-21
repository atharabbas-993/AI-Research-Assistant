from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(top_k=3)

questions = [
    "What is Embeddings?",
    "why we use embeddings in RAG",
    "What is the difference between Embeddings and Vectors?"
]

for q in questions:
    result = pipeline.ask(q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")