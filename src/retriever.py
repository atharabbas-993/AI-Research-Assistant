

from typing import List, Dict, Optional

from embeddings import EmbeddingGenerator
from vectorstore import VectorStore


class Retriever:
    """
    Handles turning a user question into relevant chunks
    by combining question embedding + vector similarity search.
    """

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vectorstore = VectorStore()

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        source_filename: Optional[str] = None
    ) -> List[Dict]:
        """
        Finds the most relevant chunks for a given question.

        Args:
            question (str): The user's question.
            top_k (int): How many chunks to retrieve. Default 3.
            source_filename (Optional[str]): If provided, only search
                                              within this specific PDF.
                                              If None, search all documents.

        Returns:
            List[Dict]: Each item has "text", "page_number", "source_filename", "distance"
        """

        # Step 1: Convert the question into a vector (same model as chunks)
        question_embedding = self.embedder.embed_query(question)

        # Step 2: Build query parameters dynamically.
        # We only add "where" if a filename filter was actually given —
        # ChromaDB expects no "where" key at all when we want to search everything,
        # not an empty dict.
        query_params = {
            "query_embeddings": [question_embedding],
            "n_results": top_k
        }

        if source_filename:
            query_params["where"] = {"source_filename": source_filename}

        # Step 3: Run the search (filtered or unfiltered depending on above)
        results = self.vectorstore.collection.query(**query_params)

        # Step 4: Reformat ChromaDB's nested output into a clean list
        retrieved_chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            retrieved_chunks.append({
                "text": doc,
                "page_number": meta["page_number"],
                "source_filename": meta["source_filename"],
                "distance": dist
            })

        return retrieved_chunks