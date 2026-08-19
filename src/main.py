from loader import load_pdf
from cleaner import clean_pages
from splitter import chunk_pages
from embeddings import EmbeddingGenerator

raw_pages = load_pdf(r"D:\Workspace\AI_Research_Assistant\data\raw_pdfs\AI_domains.pdf")
cleaned_pages = clean_pages(raw_pages)
chunks = chunk_pages(cleaned_pages)

embedder = EmbeddingGenerator()
sample_chunks = embedder.embed_chunks(chunks[:2])

print("Embedding length:", len(sample_chunks[0]["embedding"]))
print("First few numbers:", sample_chunks[0]["embedding"][:5])