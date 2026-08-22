from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

result = pipeline.ask("what is machine learning?")

print("Answer:", result["answer"])
print("\nSources (with rerank scores):")
for s in result["sources"]:
    print(f"- {s['source_filename']}, Page {s['page_number']} (score: {s['rerank_score']:.4f})")