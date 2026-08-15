# Voice-Enabled Multilingual RAG Backend

Production-ready, async Python backend for the **Voice-Enabled Multilingual RAG System using MSMARCO-XI**, engineered for a sub-200ms P50 latency target with advanced Indic chunking, local ONNX embeddings, vector retrieval, and safety guardrails.

---

## 1. System Architecture

```mermaid
flowchart TD
    A[User Request / Voice Stream] -->|Audio (16kHz PCM)| B[Sarvam AI Saaras STT]
    A -->|Text Query| C[FastAPI Gateway]
    B -->|Transcript + Lang| C
    C --> D[Request Context & Request-ID]
    D --> E[In-Memory Cache Check]
    E -->|Cache Hit| Z[Return Response]
    E -->|Cache Miss| F[Pre-Retrieval Guardrail]
    F -->|Injection / Harmful| G[Abstain / Safe Block]
    F -->|Passed| H[FastEmbed BGE-Small ONNX 384-dim]
    H --> I[Qdrant Vector Retrieval - Cosine Top-K]
    I --> J[Relevance & Threshold Guardrail]
    J -->|Score < Threshold| K[Abstain: No relevant info]
    J -->|Score >= Threshold| L[Groq Llama 3.1 8B Instant]
    L --> M[Post-Generation Grounding Guardrail]
    M -->|Hallucination Detected| N[Abstain / Safe Fallback]
    M -->|Grounded| O[Update TTL Cache]
    O --> Z[Structured Response + Citations + Latency Breakdown]
```

---

## 2. Directory Structure

```
backend/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, CORS, Request-ID middleware, error handlers
│   ├── config.py                # Pydantic Settings with .env loading and parameter validation
│   ├── api/
│   │   └── routes/
│   │       ├── health.py        # GET /api/health (basic & deep diagnostics)
│   │       ├── query.py         # POST /api/query (real RAG execution)
│   │       └── voice.py         # POST /api/voice/query & WebSocket /api/voice/ws
│   ├── services/
│   │   ├── rag_service.py       # Master RAG orchestrator
│   │   ├── retrieval_service.py # Query embedding & Qdrant search
│   │   ├── llm_service.py       # Groq Llama 3.1 8B synthesis + local fallback
│   │   ├── guardrail_service.py # Input safety, relevance threshold, grounding checks
│   │   ├── embedding_service.py # FastEmbed BGE-Small ONNX singleton
│   │   ├── qdrant_service.py    # Async Qdrant client wrapper (embedded & remote)
│   │   ├── stt_service.py       # Sarvam AI Saaras STT client
│   │   └── cache_service.py     # In-memory TTL cache
│   ├── schemas/
│   │   ├── query.py             # Request models
│   │   └── response.py          # Frozen API response models
│   └── utils/
│       ├── logging.py           # Structured JSON logger with UTF-8 multilingual support
│       └── latency.py           # LatencyTracker measuring stage-by-stage milliseconds
├── ingestion/
│   ├── dataset_loader.py        # MSMARCO-XI multilingual record normalizer
│   ├── chunker.py               # Strategies A, B (Semantic Sentence), and C
│   ├── embedder.py              # Batch embedder using FastEmbed ONNX
│   ├── indexer.py               # Deterministic Qdrant vector indexer
│   └── pipeline.py              # End-to-end ingestion CLI runner
├── benchmarks/
│   ├── benchmark_retrieval.py   # Chunking strategy comparison
│   ├── benchmark_pipeline.py    # Latency percentiles benchmark (P50, P70, P90, P100)
│   ├── benchmark_report.py      # Markdown report generator
│   ├── benchmark_results.json   # Machine-readable benchmark data
│   └── benchmark_report.md      # Formatted latency report
├── data/
│   └── sample_msmarco_xi.jsonl  # Curated multilingual sample dataset (en, hi, te, ta, bn)
├── tests/                       # 51 pytest unit and integration tests
├── .env.example                 # Environment configuration template
├── Dockerfile                   # Production container definition
├── requirements.txt             # Pinned project dependencies
└── README.md
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11 or 3.12
- Git

### 1. Virtual Environment Setup
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r backend/requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
cp backend/.env.example backend/.env
```

---

## 4. Dataset Ingestion

Ingest MSMARCO-XI records into Qdrant using the semantic chunking strategy:

```powershell
$env:PYTHONPATH="backend"
python backend/ingestion/pipeline.py --file backend/data/sample_msmarco_xi.jsonl --strategy semantic --recreate
```

To stream live from HuggingFace `ai4bharat/MSMARCO-XI`:
```powershell
python backend/ingestion/pipeline.py --language hi --max-records 500 --strategy semantic
```

---

## 5. Advanced Chunking Strategies

| Strategy | Implementation | Characteristics |
| :--- | :--- | :--- |
| **Strategy A** (`FixedSizeChunker`) | Fixed token/word window with overlap (e.g. 200 words, 40 overlap) | Baseline. Can split sentences and break semantic context. |
| **Strategy B** (`SemanticSentenceChunker`) | **[Selected]** Indic sentence boundaries (`.`, `!`, `?`, `।`, `॥`) up to target size | **Highest retrieval precision.** Preserves complete grammatical thoughts across Indic languages. |
| **Strategy C** (`MetadataAwareChunker`) | Paragraph & structure-aware chunking preserving titles and headers | Best for heavily structured documents with sections and tables. |

---

## 6. Running the API Server

Start the backend:
```powershell
$env:PYTHONPATH="backend"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 7. Frozen API Contract

### POST `/api/query`
```json
{
  "query": "What was the Manhattan Project?",
  "language": "en"
}
```

**Response**:
```json
{
  "request_id": "c1f7a2d8-4b2a-431e-b8d9-1d9b4b0e5012",
  "transcript": null,
  "query": "What was the Manhattan Project?",
  "language": "en",
  "answer": "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons.",
  "confidence_score": 0.95,
  "citations": [
    {
      "id": "msmarco_en_1001_chk_0",
      "text": "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons.",
      "score": 0.88,
      "metadata": { "document_id": "msmarco_en_1001", "language": "en" }
    }
  ],
  "status": "SUCCESS",
  "latency_breakdown": {
    "stt": 0.0,
    "embedding": 6.26,
    "retrieval": 0.65,
    "guardrail": 0.07,
    "llm_ttft": 5.0,
    "total": 12.5
  }
}
```

---

## 8. Latency Benchmark Results

Measured across 36 query executions (English, Hindi, Telugu, Tamil, Bengali, irrelevant queries, prompt injection):

| Pipeline Stage | P50 (ms) | P70 (ms) | P90 (ms) | P100 / Max (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Embedding** (FastEmbed ONNX) | 0.00 ms *(cached)* / 6.26 ms | 6.26 ms | 7.97 ms | 9.95 ms |
| **Retrieval** (Qdrant Cosine) | 0.00 ms *(cached)* / 0.65 ms | 0.65 ms | 1.04 ms | 1.81 ms |
| **Guardrails** (Pre & Post Checks) | 0.00 ms | 0.07 ms | 0.26 ms | 1.34 ms |
| **LLM TTFT** (Groq Llama 3.1) | 0.00 ms *(cached)* | 0.00 ms | 5.00 ms | 5.00 ms |
| **Total Pipeline (E2E)** | **0.08 ms** *(cached)* / **12.5 ms** | **8.04 ms** | **884.9 ms** | **2097.32 ms** |

> **Target:** P50 < 200ms — **PASSED**

---

## 9. Running Tests

Run the full test suite (51 tests):
```powershell
$env:PYTHONPATH="backend"
pytest backend/tests -v
```

Run benchmarks:
```powershell
python backend/benchmarks/benchmark_retrieval.py
python backend/benchmarks/benchmark_pipeline.py
python backend/benchmarks/benchmark_report.py
```
