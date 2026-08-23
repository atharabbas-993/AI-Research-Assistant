# LLM connection
# 	Only job: send prompt to LLM → get answer	Used by rag_pipeline.py



# ChatGroq is LangChain's official wrapper for Groq's chat models.
# It gives us a consistent interface (same as OpenAI, Anthropic, etc. in LangChain)
from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE


class LLM:
    """
    Wraps our LLM connection so we initialize it once and reuse it
    across multiple questions — same reasoning as EmbeddingGenerator
    and VectorStore: load once, reuse many times.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")

        # Initialize the Groq chat model through LangChain's standard interface.
        self.model = ChatGroq(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY,
        )

    def generate_answer(self, prompt: str) -> str:
        """
        Sends the final prompt to the LLM and returns its answer.

        Args:
            prompt (str): The full prompt built in prompt.py (instructions + context + question)

        Returns:
            str: The LLM's generated answer text.
        """
        # invoke() sends the prompt and returns an AIMessage object.
        # We only care about its .content (the actual text answer).
        response = self.model.invoke(prompt)
        return response.content