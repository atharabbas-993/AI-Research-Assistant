# src/rag_pipeline.py

from typing import Dict, Optional

from src.retriever import Retriever
from src.reranker import Reranker
from src.prompt import build_prompt
from src.llm import LLM
from src.config import RETRIEVE_TOP_K, RERANK_TOP_N, MIN_RELEVANCE_SCORE
from src.logger import setup_logger

logger = setup_logger(__name__)


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = LLM()
        logger.info("RAGPipeline initialized successfully.")

    def ask(self, question: str, source_filename: Optional[str] = None) -> Dict:
        logger.info(f"Received question: '{question}' (filter: {source_filename})")

        try:
            retrieved_chunks = self.retriever.retrieve(
                question,
                top_k=RETRIEVE_TOP_K,
                source_filename=source_filename
            )
        except Exception as e:
            # Log the full error with traceback, then re-raise so the
            # API layer (FastAPI) can turn it into a proper HTTP error.
            logger.error(f"Retrieval failed for question '{question}': {e}", exc_info=True)
            raise

        if not retrieved_chunks:
            logger.warning(f"No chunks retrieved for question: '{question}'")
            return {
                "question": question,
                "answer": "I couldn't find this information in the provided documents.",
                "sources": [],
                "answered_from_context": False
            }

        try:
            reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_n=RERANK_TOP_N)
        except Exception as e:
            logger.error(f"Reranking failed for question '{question}': {e}", exc_info=True)
            raise

        best_score = reranked_chunks[0]["rerank_score"]
        logger.info(f"Best rerank score: {best_score:.4f} (threshold: {MIN_RELEVANCE_SCORE})")

        if best_score < MIN_RELEVANCE_SCORE:
            logger.warning(f"Best score {best_score:.4f} below threshold — refusing to answer.")
            return {
                "question": question,
                "answer": "I couldn't find this information in the provided documents.",
                "sources": [],
                "answered_from_context": False
            }

        prompt = build_prompt(question, reranked_chunks)

        try:
            answer = self.llm.generate_answer(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed for question '{question}': {e}", exc_info=True)
            raise

        logger.info(f"Successfully answered question: '{question}'")

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