"""
EADIS Backend - FastAPI entrypoint.

Explainable Autonomous Deepfake Investigation System
Backend & Database module: API layer, schema, agent-output integration,
report storage.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.database import Base, engine
from app.routes import upload, investigate, report, history, knowledge
from app.utils.logger import logger

# Creates tables on startup if they don't exist yet. For a hackathon MVP
# this is fine; for production, switch to Alembic migrations.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EADIS Backend API",
    description=(
        "Explainable Autonomous Deepfake Investigation System - "
        "Backend & Database module."
    ),
    version="1.0.0",
)

# Permissive CORS for hackathon demo purposes (Streamlit/Gradio/React
# frontends running on a different port). Tighten origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global error handlers -------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


# --- Routes ------------------------------------------------------------
app.include_router(upload.router)
app.include_router(investigate.router)
app.include_router(report.router)
app.include_router(history.router)
app.include_router(knowledge.router)


@app.get("/", tags=["Health"])
def root():
    return {"service": "EADIS Backend", "status": "running"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
