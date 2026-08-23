# vector DB setup + storage
# 	Only job: save/search vectors in ChromaDB	Used by retriever.py


# src/vectorstore.py

# chromadb is our vector database library
import chromadb
from typing import List, Dict

from src.config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

class VectorStore:
    """
    Wraps ChromaDB so we can save and search embeddings easily.
    Uses persistent storage — data survives across script restarts.
    """

    def __init__(self):
        # PersistentClient saves data to disk at CHROMA_DB_DIR.
        # Without this (using just chromadb.Client()), data would
        # disappear when the script ends — bad for our use case.
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        # get_or_create_collection: if "research_papers" already exists,
        # reuse it. If not, create it. This avoids errors on repeated runs.
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

    def add_chunks(self, chunks: List[Dict], source_filename: str):
        """
        Saves chunks (with their embeddings) into the vector database.

        Args:
            chunks (List[Dict]): list of {"page_number", "chunk_text", "embedding"}
            source_filename (str): name of the PDF this chunk came from —
                                    important once you have multiple PDFs.
        """

        # ChromaDB needs 4 separate lists, all in matching order:
        # ids, embeddings, documents (the actual text), and metadatas (extra info)

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            # Every chunk needs a UNIQUE id. We combine filename + page + index
            # so IDs never collide, even across multiple PDFs.
            chunk_id = f"{source_filename}_page{chunk['page_number']}_chunk{i}"

            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["chunk_text"])

            # Metadata lets us filter/track WHERE this chunk came from later
            # (used in Step 15: metadata filtering, and for citations)
            metadatas.append({
                "source_filename": source_filename,
                "page_number": chunk["page_number"]
            })

        # add() saves everything into the collection in one batch call —
        # much faster than adding one chunk at a time in a loop.
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def count(self) -> int:
        """Returns how many chunks are currently stored — useful for debugging."""
        return self.collection.count()