
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import DATABASE_URL

# engine = the actual connection to our SQLite database file
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# SessionLocal = a factory for creating database sessions (like a "conversation"
# with the database — open, do some queries, close).
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = the parent class all our database table models inherit from.
Base = declarative_base()


class User(Base):
    """
    Represents the 'users' table in our database.
    Each attribute becomes a column.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)


# Creates the actual table in the database file if it doesn't exist yet.
def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    A dependency function that provides a database session per request,
    and guarantees it's closed afterward (even if an error occurs).
    FastAPI will call this automatically wherever we need DB access.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()