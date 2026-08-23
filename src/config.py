# all settings/constants in one place
# Central place for settings (chunk size, model names, paths)

# src/config.py

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()


# ------------------------
# Base directory (MUST be defined first — everything else depends on it)
# ------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


# ------------------------
# LLM Settings
# ------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "openai/gpt-oss-20b"   # good balance of quality + speed on Groq
LLM_TEMPERATURE = 0.0   # 0 = focused/deterministic answers, less creative guessing

# ------------------------
# Reranking Settings
# ------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANK_MODEL_NAME = "rerank-v3.5"   # Cohere's current rerank model
RETRIEVE_TOP_K = 10   # cast a wider net at retrieval stage...
RERANK_TOP_N = 3       # ...then narrow down to the best 3 after reranking

# ------------------------
# Hallucination Prevention Settings
# ------------------------
# Cohere rerank scores range 0-1 (higher = more relevant).
# Chunks scoring below this are considered "not relevant enough" to answer from.
MIN_RELEVANCE_SCORE = 0.3

# ------------------------
# Authentication Settings
# ------------------------
import secrets

# SECRET_KEY signs our JWT tokens — if this leaks, anyone could forge valid tokens.
# We generate one automatically if not set in .env (fine for learning; in real
# production, always set this explicitly and keep it secret).
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # tokens expire after 1 hour

DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'data', 'app.db')}"