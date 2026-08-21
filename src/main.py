# main.py

from rag_pipeline import RAGPipeline

# Create the pipeline once
pipeline = RAGPipeline(top_k=3)

# Ask as many questions as you want — reuses the same loaded models/connections
question = "What is Embeddings?"
result = pipeline.ask(question)

print("Question:", result["question"])
print("\nAnswer:", result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"- {source['source_filename']}, Page {source['page_number']} (distance: {source['distance']:.4f})")