"""
Database engine + session management.

Uses SQLAlchemy's declarative ORM so the underlying engine (SQLite now,
Postgres/MySQL later) can be swapped by changing DATABASE_URL only.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from .config import DATABASE_URL

# check_same_thread=False is required for SQLite when used with FastAPI's
# multi-threaded request handling. Not needed once you move to Postgres.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    FastAPI dependency that yields a DB session and guarantees it's
    closed after the request, even if an exception is raised.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
