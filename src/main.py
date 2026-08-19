from loader import load_pdf
from cleaner import clean_pages
from splitter import chunk_pages
from embeddings import EmbeddingGenerator
from vectorstore import VectorStore

# Run the full ingestion pipeline
raw_pages = load_pdf(r"D:\Workspace\AI_Research_Assistant\data\raw_pdfs\AI_domains.pdf")
cleaned_pages = clean_pages(raw_pages)
chunks = chunk_pages(cleaned_pages)

embedder = EmbeddingGenerator()
chunks_with_embeddings = embedder.embed_chunks(chunks)

# Save to vector database
store = VectorStore()
store.add_chunks(chunks_with_embeddings, source_filename="paper1.pdf")

print(f"Total chunks stored: {store.count()}")