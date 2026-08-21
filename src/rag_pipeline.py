# connects everything together
# 	Orchestrator — calls all the above in order	Used by app/main.py


# src/rag_pipeline.py

from typing import Dict

from retriever import Retriever
from prompt import build_prompt
from llm import LLM


class RAGPipeline:
    """
    The orchestrator — combines retrieval, prompt building, and LLM generation
    into a single, reusable pipeline.

    This is the ONE class the rest of the app (API, CLI, tests) will talk to.
    """

    def __init__(self, top_k: int = 3):
        """
        Args:
            top_k (int): How many chunks to retrieve per question. Default 3.
        """
        # Initialize each component ONCE here.
        # Why here and not inside ask()? Same reasoning as before —
        # loading models/connections is "expensive" (time/resources),
        # so we do it once when the pipeline is created, not on every question.
        self.retriever = Retriever()
        self.llm = LLM()
        self.top_k = top_k

    def ask(self, question: str) -> Dict:
        """
        Runs the full RAG flow for a single question.

        Args:
            question (str): The user's question.

        Returns:
            Dict: {
                "question": str,
                "answer": str,
                "sources": list of {source_filename, page_number, distance}
            }
        """

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question, top_k=self.top_k)

        # Step 2: Build the prompt using retrieved context
        prompt = build_prompt(question, retrieved_chunks)

        # Step 3: Generate the answer from the LLM
        answer = self.llm.generate_answer(prompt)

        # Step 4: Package everything into one clean result.
        # We return sources too — the app can show "answer + citations"
        # to the user, which is a core RAG feature (Step 17 builds on this).
        sources = [
            {
                "source_filename": chunk["source_filename"],
                "page_number": chunk["page_number"],
                "distance": chunk["distance"]
            }
            for chunk in retrieved_chunks
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }