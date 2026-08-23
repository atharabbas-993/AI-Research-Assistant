from typing import Dict, Optional

from src.retriever import Retriever
from src.reranker import Reranker
from src.prompt import build_prompt
from src.llm import LLM
from src.config import RETRIEVE_TOP_K, RERANK_TOP_N, MIN_RELEVANCE_SCORE


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = LLM()

    def ask(self, question: str, source_filename: Optional[str] = None) -> Dict:
        """
        Runs the full RAG flow, with a relevance check to prevent
        answering when nothing useful was actually found.

        Returns:
            Dict: {
                "question": str,
                "answer": str,
                "sources": list of {source_filename, page_number, rerank_score},
                "answered_from_context": bool   # False if we skipped the LLM
            }
        """

        # Step 1: Retrieve a wide candidate pool
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=RETRIEVE_TOP_K,
            source_filename=source_filename
        )

        # Handle the edge case: no chunks at all (empty database, or filter matched nothing)
        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "I couldn't find this information in the provided documents.",
                "sources": [],
                "answered_from_context": False
            }

        # Step 2: Rerank to find the TRUE best matches
        reranked_chunks = self.reranker.rerank(
            question,
            retrieved_chunks,
            top_n=RERANK_TOP_N
        )

        # Step 3: SAFETY CHECK — is the best match actually relevant enough?
        # reranked_chunks is already sorted best-first by Cohere, so index 0
        # is our best candidate.
        best_score = reranked_chunks[0]["rerank_score"]

        if best_score < MIN_RELEVANCE_SCORE:
            # Don't even call the LLM — we already know the context is too weak.
            # This saves an API call AND guarantees we don't hallucinate.
            return {
                "question": question,
                "answer": "I couldn't find this information in the provided documents.",
                "sources": [],
                "answered_from_context": False
            }

        # Step 4: Build prompt and generate answer, since we have relevant context
        prompt = build_prompt(question, reranked_chunks)
        answer = self.llm.generate_answer(prompt)

        sources = [
            {
                "source_filename": chunk["source_filename"],
                "page_number": chunk["page_number"],
                "rerank_score": round(chunk["rerank_score"], 4)
            }
            for chunk in reranked_chunks
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "answered_from_context": True
        }