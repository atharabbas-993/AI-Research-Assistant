# all settings/constants in one place
# Central place for settings (chunk size, model names, paths)

# src/config.py

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# ------------------------
# API Keys
# ------------------------
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ------------------------
# Embedding Model Settings
# ------------------------
# Paste any HuggingFace "Feature Extraction" model link here.
# Just copy the part after huggingface.co/ from the model page URL.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------
# Chunking Settings
# ------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ------------------------
# File Paths
# ------------------------
RAW_PDF_DIR = "data/raw_pdfs"

# ------------------------
# Vector Database Settings
# ------------------------
CHROMA_DB_DIR = "data/chroma_db"       # folder where ChromaDB saves its files
CHROMA_COLLECTION_NAME = "research_papers"  # name of our "table" of vectors