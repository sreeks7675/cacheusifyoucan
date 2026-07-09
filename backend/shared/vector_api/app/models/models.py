"""
SQLAlchemy models for EADIS.

Three tables:
  - Investigation : one row per uploaded image / investigation lifecycle.
  - AgentOutput    : one row per agent's contribution to an investigation
                     (Planner, Forensic, Semantic, Retrieval, Fusion,
                     Decision, Report). Kept separate from Investigation
                     so we can store each agent's raw JSON output
                     independently and query/debug them individually.
  - Report         : the final human-readable forensic report generated
                     for an investigation (1:1 with Investigation once
                     the pipeline completes).
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    Text,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from ..database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InvestigationStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Verdict(str, enum.Enum):
    REAL = "real"
    FAKE = "fake"
    UNCERTAIN = "uncertain"


class KnowledgeCategory(str, enum.Enum):
    """Maps directly to the six boxes in the 'Shared Knowledge & Memory
    (RAG)' architecture diagram."""

    DEEPFAKE_KB = "deepfake_knowledge_base"
    MANIPULATION_PATTERNS = "manipulation_patterns"
    MODEL_REGISTRY = "model_registry"
    FORENSIC_RULES = "forensic_rules"
    PAST_CASES = "past_cases"
    EXTERNAL_RETRIEVAL = "external_retrieval"


class SourceType(str, enum.Enum):
    FILE = "file"          # ingested PDF/txt document
    STRUCTURED = "structured"  # ingested JSON entry (pattern/rule/registry/case)


class Investigation(Base):
    __tablename__ = "investigations"

    id = Column(String, primary_key=True, default=gen_uuid)
    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    content_type = Column(String, nullable=True)

    status = Column(
        SAEnum(InvestigationStatus),
        default=InvestigationStatus.UPLOADED,
        nullable=False,
    )
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    agent_outputs = relationship(
        "AgentOutput", back_populates="investigation", cascade="all, delete-orphan"
    )
    report = relationship(
        "Report",
        back_populates="investigation",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AgentOutput(Base):
    __tablename__ = "agent_outputs"

    id = Column(String, primary_key=True, default=gen_uuid)
    investigation_id = Column(String, ForeignKey("investigations.id"), nullable=False)

    agent_name = Column(String, nullable=False)  # e.g. "forensic", "semantic"
    output_json = Column(Text, nullable=False)  # raw JSON string from the agent
    created_at = Column(DateTime(timezone=True), default=utcnow)

    investigation = relationship("Investigation", back_populates="agent_outputs")


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=gen_uuid)
    investigation_id = Column(
        String, ForeignKey("investigations.id"), nullable=False, unique=True
    )

    verdict = Column(SAEnum(Verdict), nullable=False)
    confidence_score = Column(Float, nullable=False)  # 0.0 - 1.0

    evidence_summary = Column(Text, nullable=True)
    explainable_reasoning = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    highlighted_regions_json = Column(Text, nullable=True)  # Grad-CAM/SHAP boxes
    full_report_json = Column(Text, nullable=False)  # complete fused agent output

    created_at = Column(DateTime(timezone=True), default=utcnow)

    investigation = relationship("Investigation", back_populates="report")


class KnowledgeDocument(Base):
    """
    Metadata registry for everything ingested into the Shared Knowledge
    & Memory (RAG) vector store. The actual embedded chunks live in
    ChromaDB (see app/rag/vector_store.py); this table is SQLite's
    "table of contents" for that store - what was ingested, when, into
    which category, and how many chunks it produced - so it can be
    listed, filtered, and deleted through the normal REST API without
    ever touching Chroma directly.
    """

    __tablename__ = "knowledge_documents"

    id = Column(String, primary_key=True, default=gen_uuid)
    category = Column(SAEnum(KnowledgeCategory), nullable=False)
    source_type = Column(SAEnum(SourceType), nullable=False)

    title = Column(String, nullable=False)
    original_filename = Column(String, nullable=True)  # only for SourceType.FILE
    chunk_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), default=utcnow)
