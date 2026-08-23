# src/ingestion_pipeline.py


from src.loader import load_pdf
from src.cleaner import clean_pages
from src.splitter import chunk_pages
from src.embeddings import EmbeddingGenerator
from src.vectorstore import VectorStore


class IngestionPipeline:
    """
    Orchestrates the full document ingestion flow:
    load -> clean -> chunk -> embed -> store.

    Same reasoning as RAGPipeline: one class, load expensive
    components once, expose one simple method to the rest of the app.
    """

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vectorstore = VectorStore()

    def ingest(self, file_path: str, source_filename: str) -> int:
        """
        Runs the full ingestion pipeline on one PDF file.

        Args:
            file_path (str): Path to the PDF file on disk.
            source_filename (str): Name to store as metadata (used for
                                    citations and filtering later).

        Returns:
            int: Number of chunks stored.
        """
        raw_pages = load_pdf(file_path)
        cleaned_pages = clean_pages(raw_pages)
        chunks = chunk_pages(cleaned_pages)
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)

        self.vectorstore.add_chunks(chunks_with_embeddings, source_filename=source_filename)

        return len(chunks_with_embeddings)