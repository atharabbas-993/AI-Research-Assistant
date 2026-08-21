# question embedding + search
# 	Only job: take a question → return relevant chunks	Feeds into rag_pipeline.py



from typing import List, Dict

from embeddings import EmbeddingGenerator
from vectorstore import VectorStore


class Retriever:
    """
    Handles turning a user question into relevant chunks
    by combining question embedding + vector similarity search.
    """

    def __init__(self):
        # Reuse our existing classes instead of duplicating logic —
        # this is why we built them as reusable classes earlier.
        self.embedder = EmbeddingGenerator()
        self.vectorstore = VectorStore()

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Finds the most relevant chunks for a given question.

        Args:
            question (str): The user's question.
            top_k (int): How many chunks to retrieve. Default 3.

        Returns:
            List[Dict]: Each item has "text", "page_number", "source_filename", "distance"
        """

        # Step 1: Convert the question into a vector (same model as chunks)
        question_embedding = self.embedder.embed_query(question)

        # Step 2: Ask ChromaDB to find the top_k closest chunk vectors.
        # query() does the cosine similarity math internally for us.
        results = self.vectorstore.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )

        # Step 3: ChromaDB returns results in a nested-list format
        # (because it supports multiple queries at once — we only send one).
        # We simplify it into a clean list of dicts for easier use later.
        retrieved_chunks = []

        documents = results["documents"][0]      # the actual chunk text
        metadatas = results["metadatas"][0]       # page_number, source_filename
        distances = results["distances"][0]       # how close each match is (lower = more similar)

        for doc, meta, dist in zip(documents, metadatas, distances):
            retrieved_chunks.append({
                "text": doc,
                "page_number": meta["page_number"],
                "source_filename": meta["source_filename"],
                "distance": dist
            })

        return retrieved_chunks