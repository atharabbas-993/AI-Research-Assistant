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


app = FastAPI(
    title="AI Research Assistant API",
    description="Upload research papers and ask questions about them.",
    version="1.0.0"
)

# Create the users table on startup if it doesn't exist yet
init_db()

ingestion_pipeline = IngestionPipeline()
rag_pipeline = RAGPipeline()


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
    # Check if username is already taken
    existing_user = db.query(User).filter(User.username == request.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered.")

    # NEVER store the plain password — hash it first
    new_user = User(
        username=request.username,
        hashed_password=hash_password(request.password)
    )
    db.add(new_user)
    db.commit()

    return {"message": "User registered successfully."}


# ------------------------------------------------------------------
# AUTH ENDPOINT 2: Login (returns a JWT token)
# ------------------------------------------------------------------
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    OAuth2PasswordRequestForm expects standard 'username' and 'password'
    form fields — this is what makes our /login endpoint compatible
    with FastAPI's built-in "Authorize" button in the /docs UI.
    """
    user = db.query(User).filter(User.username == form_data.username).first()

    # Check BOTH that the user exists AND the password is correct,
    # using a generic error message for both cases — this avoids
    # revealing whether a username exists at all (security best practice).
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    access_token = create_access_token(data={"sub": user.username})

    return {"access_token": access_token, "token_type": "bearer"}


# ------------------------------------------------------------------
# ENDPOINT: Upload a PDF — now PROTECTED (requires login)
# ------------------------------------------------------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)   # <-- this line protects the endpoint
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(RAW_PDF_DIR, file.filename)
    os.makedirs(RAW_PDF_DIR, exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunk_count = ingestion_pipeline.ingest(save_path, source_filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {
        "filename": file.filename,
        "chunks_stored": chunk_count,
        "uploaded_by": current_user.username,
        "message": "PDF uploaded and processed successfully."
    }


# ------------------------------------------------------------------
# ENDPOINT: Ask a question — now PROTECTED (requires login)
# ------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    current_user: User = Depends(get_current_user)   # <-- protects this endpoint too
):
    try:
        result = rag_pipeline.ask(
            question=request.question,
            source_filename=request.source_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

    return result


@app.get("/health")
async def health_check():
    return {"status": "ok"}