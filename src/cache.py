# src/cache.py

import hashlib
from typing import Optional, Dict


class SimpleCache:
    """
    A basic in-memory cache that stores question -> answer results.
    """

    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size (int): Maximum number of cached items before we
                             start evicting the oldest ones.
        """
        self._cache: Dict[str, Dict] = {}
        self.max_size = max_size

    def _make_key(self, question: str, source_filename: Optional[str]) -> str:
        """Builds a unique, normalized cache key from question + filter."""
        raw_key = f"{question.lower().strip()}|{source_filename or 'all'}"
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def get(self, question: str, source_filename: Optional[str] = None) -> Optional[Dict]:
        """Returns the cached result if it exists, else None."""
        key = self._make_key(question, source_filename)
        return self._cache.get(key)

    def set(self, question: str, source_filename: Optional[str], result: Dict) -> None:
        """Stores a result in the cache, evicting the oldest entry if full."""
        key = self._make_key(question, source_filename)

        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result