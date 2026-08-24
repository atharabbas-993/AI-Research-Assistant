# src/rag_pipeline.py

from typing import Dict, Optional

from src.retriever import Retriever
from src.reranker import Reranker
from src.prompt import build_prompt
from src.llm import LLM
from src.cache import SimpleCache
from src.config import RETRIEVE_TOP_K, RERANK_TOP_N, MIN_RELEVANCE_SCORE
from src.logger import setup_logger

logger = setup_logger(__name__)


class RAGPipeline:
    """
    The orchestrator — combines retrieval, reranking, prompt building,
    caching, and LLM generation into a single, reusable pipeline.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = LLM()
        self.cache = SimpleCache(max_size=100)
        logger.info("RAGPipeline initialized successfully.")

    def ask(self, question: str, source_filename: Optional[str] = None) -> Dict:
        """
        Runs the full RAG flow for a single question (non-streaming),
        with caching and a relevance check to prevent hallucination.
        """
        logger.info(f"Received question: '{question}' (filter: {source_filename})")

        # Check cache first
        cached_result = self.cache.get(question, source_filename)
        if cached_result is not None:
            logger.info(f"Cache HIT for question: '{question}'")
            return cached_result

        logger.info(f"Cache MISS for question: '{question}' — running full pipeline.")

        try:
            retrieved_chunks = self.retriever.retrieve(
                question, top_k=RETRIEVE_TOP_K, source_filename=source_filename
            )
        except Exception as e:
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

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "answered_from_context": True
        }

        # Only cache successful, confident answers
        self.cache.set(question, source_filename, result)
        logger.info(f"Cached result for question: '{question}'")

        return result

    def ask_stream(self, question: str, source_filename: Optional[str] = None):
        """
        Same flow as ask(), but streams the final answer instead of
        returning it all at once. Retrieval, reranking, and the
        relevance threshold check happen normally BEFORE streaming
        starts — only LLM generation is streamed. Not cached.
        """
        logger.info(f"Received streaming question: '{question}' (filter: {source_filename})")

        try:
            retrieved_chunks = self.retriever.retrieve(
                question, top_k=RETRIEVE_TOP_K, source_filename=source_filename
            )
        except Exception as e:
            logger.error(f"Retrieval failed for question '{question}': {e}", exc_info=True)
            yield "An error occurred while retrieving information."
            return

        if not retrieved_chunks:
            yield "I couldn't find this information in the provided documents."
            return

        try:
            reranked_chunks = self.reranker.rerank(question, retrieved_chunks, top_n=RERANK_TOP_N)
        except Exception as e:
            logger.error(f"Reranking failed for question '{question}': {e}", exc_info=True)
            yield "An error occurred while ranking results."
            return

        best_score = reranked_chunks[0]["rerank_score"]

        if best_score < MIN_RELEVANCE_SCORE:
            yield "I couldn't find this information in the provided documents."
            return

        prompt = build_prompt(question, reranked_chunks)

        try:
            for chunk in self.llm.generate_answer_stream(prompt):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming LLM generation failed for question '{question}': {e}", exc_info=True)
            yield "\n\n[An error occurred while generating the answer.]"