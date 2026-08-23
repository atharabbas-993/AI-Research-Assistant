# app/main.py

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from src.ingestion_pipeline import IngestionPipeline
from src.rag_pipeline import RAGPipeline
from src.config import RAW_PDF_DIR
from src.database import init_db, get_db, User
from src.auth import hash_password, verify_password, create_access_token, get_current_user
from src.logger import setup_logger

logger = setup_logger(__name__)

# ------------------------------------------------------------------
# Create the FastAPI app instance
# ------------------------------------------------------------------
app = FastAPI(
    title="AI Research Assistant API",
    description="Upload research papers and ask questions about them.",
    version="1.0.0"
)

# Create the users table on startup if it doesn't exist yet
init_db()

# Initialize pipelines ONCE, when the app starts
ingestion_pipeline = IngestionPipeline()
rag_pipeline = RAGPipeline()

logger.info("AI Research Assistant API started successfully.")


# ------------------------------------------------------------------
# Request/Response schemas
# ------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    question: str
    source_filename: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list
    answered_from_context: bool


# ------------------------------------------------------------------
# AUTH ENDPOINT 1: Register a new user
# ------------------------------------------------------------------
@app.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    logger.info(f"Registration attempt for username: '{request.username}'")

    existing_user = db.query(User).filter(User.username == request.username).first()
    if existing_user:
        logger.warning(f"Registration failed — username already exists: '{request.username}'")
        raise HTTPException(status_code=400, detail="Username already registered.")

    new_user = User(
        username=request.username,
        hashed_password=hash_password(request.password)
    )
    db.add(new_user)
    db.commit()

    logger.info(f"User registered successfully: '{request.username}'")

    return {"message": "User registered successfully."}


# ------------------------------------------------------------------
# AUTH ENDPOINT 2: Login (returns a JWT token)
# ------------------------------------------------------------------
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Login attempt for username: '{form_data.username}'")

    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for username: '{form_data.username}'")
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    access_token = create_access_token(data={"sub": user.username})
    logger.info(f"User logged in successfully: '{user.username}'")

    return {"access_token": access_token, "token_type": "bearer"}


# ------------------------------------------------------------------
# ENDPOINT: Upload a PDF — PROTECTED (requires login)
# ------------------------------------------------------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    logger.info(f"Upload request from user '{current_user.username}': {file.filename}")

    if not file.filename.endswith(".pdf"):
        logger.warning(f"Rejected non-PDF upload attempt: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(RAW_PDF_DIR, file.filename)
    os.makedirs(RAW_PDF_DIR, exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunk_count = ingestion_pipeline.ingest(save_path, source_filename=file.filename)
        logger.info(f"Successfully ingested '{file.filename}': {chunk_count} chunks stored.")
    except Exception as e:
        logger.error(f"Ingestion failed for '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {
        "filename": file.filename,
        "chunks_stored": chunk_count,
        "uploaded_by": current_user.username,
        "message": "PDF uploaded and processed successfully."
    }


# ------------------------------------------------------------------
# ENDPOINT: Ask a question — PROTECTED (requires login)
# ------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    current_user: User = Depends(get_current_user)
):
    logger.info(f"Question from user '{current_user.username}': {request.question}")

    try:
        result = rag_pipeline.ask(
            question=request.question,
            source_filename=request.source_filename
        )
    except Exception as e:
        logger.error(f"Failed to answer question '{request.question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

    return result


# ------------------------------------------------------------------
# ENDPOINT: Health check
# ------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}