# src/rag_pipeline.py

from typing import Dict, Optional

from retriever import Retriever
from reranker import Reranker
from prompt import build_prompt
from llm import LLM
from config import RETRIEVE_TOP_K, RERANK_TOP_N


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = LLM()

    def ask(self, question: str, source_filename: Optional[str] = None) -> Dict:
        # Step 1: Retrieve a WIDER set of candidates (e.g. top 10)
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=RETRIEVE_TOP_K,
            source_filename=source_filename
        )

        # Step 2: Rerank down to the BEST few (e.g. top 3)
        reranked_chunks = self.reranker.rerank(
            question,
            retrieved_chunks,
            top_n=RERANK_TOP_N
        )

        # Step 3: Build prompt using the reranked (higher quality) chunks
        prompt = build_prompt(question, reranked_chunks)

        # Step 4: Generate the answer
        answer = self.llm.generate_answer(prompt)

        sources = [
            {
                "source_filename": chunk["source_filename"],
                "page_number": chunk["page_number"],
                "rerank_score": chunk.get("rerank_score")
            }
            for chunk in reranked_chunks
        ]

        return {"question": question, "answer": answer, "sources": sources}