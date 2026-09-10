# 🔬 AI Research Assistant

A production-structured **RAG (Retrieval-Augmented Generation)** application that allows users to upload research papers (PDFs) and ask questions based directly on their content — with reliable **source citations**.

🌐 **Live Demo:** https://ai-research-assistant-6ov1.onrender.com/

---

## ✨ Features

* 📄 **PDF Upload & Ingestion** — Load, clean, chunk, embed, and store research papers
* 🔎 **Semantic Search** — Retrieve relevant document chunks using embeddings
* 🎯 **Reranking** — Cohere Rerank improves retrieval accuracy
* 🤖 **Grounded LLM Answers** — Groq generates answers strictly from uploaded documents
* 🛡️ **Hallucination Prevention** — Relevance thresholding prevents unsupported answers
* 📚 **Source Citations** — Every answer includes filename and page number
* 🔐 **JWT Authentication** — Secure user registration and login
* ⚡ **Response Caching** — Reduces repeated LLM calls and improves response speed
* 🌊 **Streaming Responses** — Answers are delivered progressively
* 📊 **Evaluation Suite** — Measure retrieval and answer performance
* 🐳 **Dockerized** — Consistent development and deployment environment
* ☁️ **Cloud Deployment** — Deployed using Render

---

## 🛠️ Tech Stack

| Component           | Technology                                     |
| ------------------- | ---------------------------------------------- |
| 🧩 Backend          | FastAPI, LangChain                             |
| 🧠 Embeddings       | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| 🗄️ Vector Database | ChromaDB                                       |
| 🎯 Reranking        | Cohere Rerank API                              |
| 🤖 LLM              | Groq (`openai/gpt-oss-120b`)                   |
| 🔐 Authentication   | JWT (`python-jose`) + bcrypt                   |
| 💾 Database         | SQLite + SQLAlchemy ORM                        |
| 🐳 Containerization | Docker                                         |
| ☁️ Deployment       | Render                                         |

---

## 🏗️ Architecture

### 📥 Document Ingestion

```text
PDF Upload
    ↓
Load
    ↓
Clean
    ↓
Chunk
    ↓
Generate Embeddings
    ↓
Store in ChromaDB
```

### 💬 Question Answering

```text
Question
    ↓
Generate Query Embedding
    ↓
Retrieve Top 10
    ↓
Cohere Reranking
    ↓
Select Top 3
    ↓
Relevance Check
    ↓
Build Prompt
    ↓
Groq LLM
    ↓
Answer + Source Citations
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/atharabbas-993/AI-Research-Assistant.git
cd AI-Research-Assistant
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then add your API keys:

```env
HUGGINGFACE_API_KEY=your_key
GROQ_API_KEY=your_key
COHERE_API_KEY=your_key
SECRET_KEY=your_secret_key
```

> ⚠️ Never commit your `.env` file or expose your API keys publicly.

---

## ▶️ Run Locally

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 📖 API Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

You can also open the frontend:

```text
frontend/index.html
```

---

## 🐳 Run with Docker

### Build the Image

```bash
docker build -t ai-research-assistant .
```

### Run the Container

```bash
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  ai-research-assistant
```

For Windows PowerShell, use:

```powershell
docker run -p 8000:8000 --env-file .env -v ${PWD}/data:/app/data ai-research-assistant
```

---

## 📊 Evaluation

The project includes an evaluation pipeline for measuring **retrieval quality** and **answer quality**.

Run:

```bash
python main.py
```

The evaluation uses:

```text
data/eval_dataset.json
```

### 📈 Retrieval Metrics

* 🎯 Hit@K
* 🔍 Recall@K
* 📌 Precision@K
* 📊 Mean Reciprocal Rank (MRR)

This helps evaluate whether the RAG pipeline is retrieving the **right context** before generating an answer.

---

## 🔐 Security

The application includes:

* 🔑 JWT-based authentication
* 🔒 Password hashing with bcrypt
* 🌱 Environment-based secret management
* 🚫 Protected API endpoints

API keys and secrets should always be stored in environment variables.

---

## ⚠️ Known Limitations

* ☁️ **Ephemeral Cloud Storage** — The free Render deployment uses an ephemeral filesystem, so uploaded documents and local database/vector-store data can reset after redeployment or restart.
* ⚡ **In-Memory Cache** — Cache data is lost when the server restarts and does not scale across multiple instances.
* 🚦 **Rate Limiting** — Authentication endpoints do not currently have rate limiting.
* 🗄️ **Local Storage** — ChromaDB and SQLite are currently configured for local/persistent filesystem usage rather than a managed cloud database.

---

## 🔮 Future Improvements

* 🗄️ Move to a managed PostgreSQL database
* ☁️ Use managed/cloud vector storage
* ⚡ Replace in-memory cache with Redis
* 🚦 Add API rate limiting
* 📈 Add production monitoring and logging
* 🧪 Expand automated evaluation
* 🔄 Add background document processing
* 👥 Add multi-user document isolation
* 🚀 Improve production scalability

---

## 🎯 Project Goal

The goal of this project is to demonstrate a **real-world RAG system** rather than a simple chatbot.

It combines:

**RAG + Semantic Search + Reranking + LLMs + Authentication + Caching + Streaming + Evaluation + Docker + Cloud Deployment**

into one production-structured AI application.

---

## 👨‍💻 Author

**Athar Abbas**

AI & Machine Learning Engineer
Interested in **RAG, LLMs, Generative AI, Computer Vision, and AI Engineering**.

---

⭐ If you find this project useful, consider giving it a star!
