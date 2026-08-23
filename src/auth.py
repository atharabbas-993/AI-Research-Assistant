from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from src.database import User, get_db
from src.config import SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES


# CryptContext handles password hashing — bcrypt is the algorithm used.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer tells FastAPI: "expect a token in the
# Authorization header as 'Bearer <token>'". tokenUrl points to our
# login endpoint (used by the auto-generated docs UI).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password(password: str) -> str:
    """Converts a plain password into a secure hash for storage."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Checks if a plain password matches a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    """
    Creates a signed JWT token containing the given data (e.g., username),
    with an expiration time baked in.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    # jwt.encode signs the token using our SECRET_KEY — this signature
    # is what makes the token tamper-proof: anyone can READ a JWT's
    # contents, but only someone with SECRET_KEY can create a VALID one.
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    A FastAPI dependency: extracts and verifies the JWT token from the
    request, and returns the corresponding User. If the token is missing,
    invalid, or expired, this raises a 401 error automatically —
    any endpoint using this dependency is now "protected."
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode and verify the token's signature + expiration
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        # Covers: invalid signature, expired token, malformed token
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception

    return user