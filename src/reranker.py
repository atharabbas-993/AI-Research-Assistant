# src/reranker.py

import cohere
from typing import List, Dict

from src.config import COHERE_API_KEY, RERANK_MODEL_NAME


class Reranker:
    """
    Re-scores retrieved chunks using Cohere's rerank API,
    which compares the question and each chunk together for
    more accurate relevance scoring than embedding similarity alone.
    """

    def __init__(self):
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found. Check your .env file.")

        # Cohere's client — reused across calls, same "load once" pattern
        self.client = cohere.ClientV2(api_key=COHERE_API_KEY)

    def rerank(self, question: str, chunks: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Re-orders chunks by true relevance to the question.

        Args:
            question (str): The user's question.
            chunks (List[Dict]): Chunks from retriever.retrieve()
                                  (list of {"text", "page_number", "source_filename", "distance"})
            top_n (int): How many top chunks to keep after reranking.

        Returns:
            List[Dict]: Same chunk dicts, reordered by relevance,
                        with an added "rerank_score" field, trimmed to top_n.
        """

        # Cohere needs just the plain text of each chunk to score against the question
        documents = [chunk["text"] for chunk in chunks]

        # Call the rerank API — it returns indices sorted by relevance,
        # each with a relevance_score (higher = more relevant)
        response = self.client.rerank(
            model=RERANK_MODEL_NAME,
            query=question,
            documents=documents,
            top_n=top_n
        )

        # Rebuild our chunk list using Cohere's new relevance ordering
        reranked_chunks = []
        for result in response.results:
            original_chunk = chunks[result.index]   # map back to our original chunk dict
            original_chunk["rerank_score"] = result.relevance_score
            reranked_chunks.append(original_chunk)

        return reranked_chunks