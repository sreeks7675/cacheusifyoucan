# EADIS Backend & Database Module

Backend for the **Explainable Autonomous Deepfake Investigation System**
(EADIS). This module owns: the FastAPI layer, database schema, agent
output integration, and report storage/retrieval.

## Quickstart

```bash
cd eadis_backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Server runs at `http://127.0.0.1:8000`. Interactive API docs (Swagger)
are auto-generated at `http://127.0.0.1:8000/docs`.

## Project Structure

```
eadis_backend/
├── app/
│   ├── main.py              # FastAPI app, middleware, global error handlers
│   ├── config.py            # paths, DB URL, upload constraints
│   ├── database.py          # SQLAlchemy engine/session
│   ├── models/
│   │   └── models.py        # Investigation, AgentOutput, Report ORM models
│   ├── schemas/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── routes/
│   │   ├── upload.py        # POST /upload
│   │   ├── investigate.py   # POST /investigate
│   │   ├── report.py        # GET /report/{id}
│   │   └── history.py       # GET /history
│   ├── services/
│   │   ├── agent_interface.py       # <-- plug your real agents in here
│   │   ├── pipeline_service.py      # orchestrates the agent pipeline
│   │   └── investigation_service.py # DB query/CRUD helpers
│   └── utils/
│       ├── file_handler.py  # upload validation + disk persistence
│       └── logger.py        # centralized logging
│   └── rag/                  # Shared Knowledge & Memory (RAG) module
│       ├── chunker.py        # splits text into overlapping chunks
│       ├── embedder.py       # <-- swap embedding model here
│       ├── vector_store.py   # ChromaDB wrapper
│       ├── knowledge_loader.py  # PDF/txt/JSON -> plain text
│       └── retriever.py      # ingest + search orchestration
├── uploads/                  # stored images (gitignored)
├── reports/                  # reserved for exported PDF/report files
├── vector_store/             # ChromaDB persistent store (gitignored)
├── requirements.txt
└── eadis.db                  # SQLite DB (auto-created on first run)
```

## Shared Knowledge & Memory (RAG) Module

This extends the backend with the "Shared Knowledge & Memory" box from
the architecture diagram — a vector-searchable knowledge base sitting
alongside the SQLite metadata store:

```
                Upload Image
                     │
                     ▼
              Investigation API
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   SQLite Database         Vector Database (ChromaDB)
 (metadata & reports)     (embeddings / RAG)
```

It covers all six categories from the diagram via one `category` field
(`KnowledgeCategory` enum): `deepfake_knowledge_base`,
`manipulation_patterns`, `model_registry`, `forensic_rules`,
`past_cases`, `external_retrieval`.

### API Reference

**`POST /knowledge/documents`** — ingest a PDF/.txt/.md document (multipart form: `category`, `title`, `file`). Extracts text, chunks it, embeds each chunk, stores in ChromaDB, and records metadata in SQLite. Use for the Deepfake Knowledge Base or External Retrieval categories.

**`POST /knowledge/entries`** — ingest one structured JSON record (a manipulation pattern, forensic rule, model registry entry, or past case):
```bash
curl -X POST http://127.0.0.1:8000/knowledge/entries \
  -H "Content-Type: application/json" \
  -d '{
    "category": "manipulation_patterns",
    "title": "Reflection mismatch",
    "entry": {"pattern": "Reflection mismatch", "description": "Reflection does not match lighting.", "severity": "High"}
  }'
```

**`POST /knowledge/search`** — semantic search, optionally scoped to one category:
```bash
curl -X POST http://127.0.0.1:8000/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "reflection lighting mismatch", "top_k": 5, "category": "manipulation_patterns"}'
```
This is what the Retrieval/Fusion agents would call to pull supporting evidence during an investigation.

**`GET /knowledge/documents`** — list ingested documents, optionally filtered by `category`.

**`DELETE /knowledge/documents/{id}`** — remove a document and all its embedded chunks.

### Embedding model

The default embedder (`app/rag/embedder.py`) is a deterministic,
fully-offline `HashingVectorizer` — it needs **no internet access or
model download**, which is why the whole module runs and tests
immediately in any environment. Its semantic quality is a step below a
real neural embedding model, but retrieval still ranks relevant
results correctly (verified in testing).

To upgrade to real semantic embeddings (recommended once you have
reliable internet for model downloads), swap `get_embedding_function()`
in `embedder.py` — nothing else in the module needs to change:

```python
from chromadb.utils import embedding_functions

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"  # or e5-small, all-MiniLM-L6-v2, etc.
    )
```

### Vector database choice

Using **ChromaDB** (local, persistent, no server needed) — the same
recommendation for hackathon use. If you later need a managed/cloud
option, only `vector_store.py` needs rewriting (Pinecone, Qdrant, and
Weaviate all have similar Python clients).

## Database Schema

Three tables (see `app/models/models.py`):

| Table | Purpose |
|---|---|
| `investigations` | One row per uploaded image. Tracks `status` (`uploaded` → `processing` → `completed`/`failed`) and file metadata. |
| `agent_outputs` | One row per agent per investigation. Stores each agent's raw JSON output (`planner`, `forensic`, `semantic`, `retrieval`, `fusion`, `decision`, `report`), so every step is independently inspectable — this is what makes the system *explainable* rather than a black box. |
| `reports` | Final forensic report per investigation: verdict, confidence score, evidence summary, explainable reasoning, recommendations, highlighted regions, and the full fused agent context as JSON. |

SQLite is used for the MVP. Because everything goes through the
SQLAlchemy ORM, moving to Postgres later is a one-line change in
`app/config.py` (`DATABASE_URL`).

## API Reference

### `POST /upload`
Upload an image (multipart/form-data, field name `file`). Validates
extension (`.jpg .jpeg .png .webp .bmp`) and size (≤15 MB), saves it to
disk under a UUID name, and creates an `Investigation` row.

```bash
curl -X POST http://127.0.0.1:8000/upload -F "file=@sample.jpg"
```
```json
{
  "investigation_id": "2086e2c8-...",
  "original_filename": "sample.jpg",
  "status": "uploaded",
  "message": "Image uploaded successfully. Call /investigate to start analysis."
}
```

### `POST /investigate`
Runs the full agent pipeline for a previously uploaded image:
`Planner → Forensic → Semantic → Retrieval → Fusion → Decision → Report`.
Each agent's output is stored individually; the final verdict +
confidence + explanation are stored as a `Report`.

```bash
curl -X POST http://127.0.0.1:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"investigation_id": "2086e2c8-..."}'
```
```json
{
  "investigation_id": "2086e2c8-...",
  "status": "completed",
  "report_id": "9e335da2-...",
  "verdict": "fake",
  "confidence_score": 0.854,
  "message": "Investigation completed successfully."
}
```

### `GET /report/{investigation_id}`
Returns the full report, including every individual agent's raw output
for transparency.

### `GET /history?skip=0&limit=50`
Paginated list of all past investigations with status/verdict summary.

## Plugging In Your Real Agents

This is the one thing your teammates need to know: **all real agent
logic goes in `app/services/agent_interface.py`.** Each function
(`run_forensic_agent`, `run_semantic_agent`, etc.) currently returns a
mock JSON dict so the whole system is runnable end-to-end today.
Replace the body of each function with your model/agent call — the
input/output contract (`(image_path, context) -> dict`) and everything
downstream (DB storage, API responses) stays unchanged.

```python
def run_forensic_agent(image_path: str, context: dict) -> dict:
    result = your_forensic_model.analyze(image_path)   # <- your real call
    return {"agent_name": "forensic", "data": result}
```

## Error Handling

- Invalid file type/size → `400` / `413`
- Unknown investigation/report id → `404`
- Re-investigating a completed investigation → `409`
- Pipeline exceptions → investigation marked `failed` with
  `error_message` stored, API returns `500` with details
- Unhandled exceptions and validation errors are caught globally in
  `main.py` and logged to both console and `eadis.log`

## Git Workflow Suggestion

Given the team's folder-based ownership pattern used on prior
projects (AccessAI), this module is self-contained under
`eadis_backend/`, so it can live on its own branch
(`feature/backend-database`) and merge cleanly once each agent's
folder (owned by other teammates) is ready to plug into
`agent_interface.py`.
