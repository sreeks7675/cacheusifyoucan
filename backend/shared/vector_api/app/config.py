"""
Central configuration for the EADIS backend.
Keeping all paths / constants in one place makes it trivial to move
from SQLite -> Postgres later, or change upload limits, without
hunting through the codebase.
"""

import os

# --- Base paths -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Database -----------------------------------------------------------
# SQLite for the hackathon MVP. Swapping to Postgres later only requires
# changing this URL (e.g. "postgresql://user:pass@host/dbname") since we
# use SQLAlchemy's ORM layer rather than raw SQL anywhere.
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'eadis.db')}"

# --- Upload constraints ---------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_SIZE_MB = 15

# --- Logging --------------------------------------------------------------
LOG_FILE = os.path.join(BASE_DIR, "eadis.log")

# --- RAG / Shared Knowledge & Memory (vector store) -----------------------
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_store")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

KNOWLEDGE_COLLECTION_NAME = "eadis_knowledge"

# Chunking defaults for ingested documents (chars, not tokens - keeps this
# dependency-free; swap for a tokenizer-aware chunker if needed later).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Embedding vector dimensionality for the default offline embedder.
EMBEDDING_DIM = 384
