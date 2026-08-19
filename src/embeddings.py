# src/embeddings.py

from huggingface_hub import InferenceClient
from typing import List, Dict

from config import HUGGINGFACE_API_KEY, EMBEDDING_MODEL_NAME


class EmbeddingGenerator:
    """
    Calls HuggingFace's Inference Providers API to generate embeddings,
    using the official huggingface_hub client (handles routing internally,
    so it won't break if HF changes their endpoint URLs again).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name

        if not HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACE_API_KEY not found. Check your .env file.")

        # provider="hf-inference" tells HuggingFace to route this
        # to their own free serverless inference backend.
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=HUGGINGFACE_API_KEY,
        )

    def _get_embedding(self, text: str) -> List[float]:
        """Sends ONE piece of text to the API and returns its embedding."""
        result = self.client.feature_extraction(
            text,
            model=self.model_name,
        )
        # result is a numpy array — convert to a plain list for easy storage
        return result.tolist()

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generates embeddings for a list of chunks."""
        for chunk in chunks:
            chunk["embedding"] = self._get_embedding(chunk["chunk_text"])
        return chunks