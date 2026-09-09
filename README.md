# AI Research Assistant

A production-structured RAG (Retrieval-Augmented Generation) application that lets users upload research papers (PDFs) and ask questions answered directly from that content, with citations.


**Live demo:** https://ai-research-assistant-6ov1.onrender.com/

## Features
- PDF upload and ingestion (chunking, embedding, vector storage)
- Semantic search with reranking (Cohere) for accurate retrieval
- LLM-powered answers (Groq) grounded strictly in uploaded documents
- Hallucination prevention via relevance thresholding
- Source citations (filename + page number) on every answer
- JWT-based authentication
- In-memory response caching
- Streaming responses
- Full evaluation suite (retrieval accuracy + answer accuracy)
- Dockerized for consistent deployment

## Tech Stack
- **Backend:** FastAPI, LangChain
- **Embeddings:** HuggingFace Inference API (`all-MiniLM-L6-v2`)
- **Vector DB:** ChromaDB (persistent, local)
- **Reranking:** Cohere Rerank API
- **LLM:** Groq (`openai/gpt-oss-120b`)
- **Auth:** JWT (python-jose) + bcrypt password hashing
- **Database:** SQLite (SQLAlchemy ORM)
- **Deployment:** Docker, Render

## Architecture
\`\`\`
PDF Upload → Load → Clean → Chunk → Embed → Store (ChromaDB)
                                                    ↓
Question → Embed → Retrieve (top 10) → Rerank (top 3) → Relevance Check → Prompt → LLM → Answer + Sources
\`\`\`

## Setup

1. Clone the repo and create a virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
\`\`\`

2. Copy \`.env.example\` to \`.env\` and fill in your API keys:
\`\`\`bash
cp .env.example .env
\`\`\`
Required keys: \`HUGGINGFACE_API_KEY\`, \`GROQ_API_KEY\`, \`COHERE_API_KEY\`, \`SECRET_KEY\`

3. Run locally:
\`\`\`bash
uvicorn app.main:app --reload
\`\`\`

4. Or run via Docker:
\`\`\`bash
docker build -t ai-research-assistant .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data ai-research-assistant
\`\`\`

5. Visit \`http://127.0.0.1:8000/docs\` for interactive API docs, or open \`frontend/index.html\` for the UI.

## Evaluation
Run \`python main.py\` (with the evaluation script) to see retrieval accuracy and answer accuracy against \`data/eval_dataset.json\`.

## Known Limitations
- Free-tier cloud deployment uses an ephemeral filesystem — uploaded data resets on redeploy
- In-memory cache resets on server restart and doesn't scale across multiple instances
- No rate limiting on auth endpoints yet