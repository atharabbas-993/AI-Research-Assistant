# src/llm.py

from langchain_groq import ChatGroq
from typing import Iterator

from src.config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE


class LLM:
    """
    Wraps our LLM connection so we initialize it once and reuse it
    across multiple questions.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")

        self.model = ChatGroq(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY,
        )

    def generate_answer(self, prompt: str) -> str:
        """Non-streaming: returns the complete answer at once."""
        response = self.model.invoke(prompt)
        return response.content

    def generate_answer_stream(self, prompt: str) -> Iterator[str]:
        """
        Streaming version: yields chunks of the answer as they're
        generated, instead of waiting for the full response.
        """
        for chunk in self.model.stream(prompt):
            if chunk.content:
                yield chunk.content