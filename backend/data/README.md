# MSMARCO-XI Dataset & Data Analyst Handoff Specification

This directory manages the ingested knowledge corpus for the **HH Goa 2026 Task 2 — Voice-Enabled Multilingual RAG System**.

---

## 1. Current State: Demo Data Mode (`DATA_MODE=demo`)

During the initial development phase, the backend operates on a curated **40-record multilingual demo dataset**:
* Path: [`backend/data/demo/demo_msmarco_xi.jsonl`](file:///d:/Voice-Enabled%20RAG/backend/data/demo/demo_msmarco_xi.jsonl)
* Target Qdrant Collection: `msmarco_demo`
* Benchmark Query Set: [`backend/benchmarks/demo_test_queries.json`](file:///d:/Voice-Enabled%20RAG/backend/benchmarks/demo_test_queries.json)

---

## 2. Future Production State: Real MSMARCO-XI Integration

When the Data Analyst team finalizes the MSMARCO-XI processing, the pipeline can be seamlessly switched to production with **zero backend code redesign**.

```text
       [ DEMO MODE ]                                    [ PRODUCTION MODE ]
  backend/data/demo/                                  backend/data/production/
  demo_msmarco_xi.jsonl                               msmarco_xi_processed.jsonl
          │                                                       │
          ▼                                                       ▼
  Qdrant (msmarco_demo)                               Qdrant (msmarco_xi_prod)
          │                                                       │
          └───────────────►  SAME FASTAPI BACKEND  ◄──────────────┘
                             SAME RAG PIPELINE
                             SAME GUARDRAILS
                             SAME HARNESS
```

---

## 3. Data Analyst Handoff Deliverables Checklist

The Data Analyst must supply the following 6 artifacts:

### 1. Processed MSMARCO-XI Dataset (`msmarco_xi_processed.jsonl`)
* **Format:** JSON Lines (`.jsonl`), UTF-8 encoded.
* **Target Path:** `backend/data/production/msmarco_xi_processed.jsonl` (or configured via environment).
* **Record Structure:**
  ```json
  {
    "id": "doc_001",
    "language": "en",
    "title": "Document Title",
    "query": "Associated training query (optional)",
    "answers": ["Ground truth answer (optional)"],
    "source": "ai4bharat/MSMARCO-XI",
    "passages": [
      {
        "passage_id": "p_001_0",
        "text": "Full text of the passage...",
        "is_selected": true
      }
    ]
  }
  ```

### 2. Supported Languages & Partitions
* Must include balanced subsets for:
  * English (`en`)
  * Hindi (`hi`)
  * Marathi (`mr`)
  * Bengali (`bn`), Telugu (`te`), Tamil (`ta`), Hinglish/Code-Switched

### 3. Recommended Chunking Parameters
* Semantic Sentence target word count: `150–200 words`
* Parent-Child sizes: Parent `~512 words`, Child `~128 words` (20 word overlap)
* Metadata attributes: `query_id`, `passage_id`, `parent_id`, `child_id`, `language`, `chunk_strategy`, `is_selected`, `source`

### 4. 100+ Evaluation Query Benchmark File (`test_queries.json`)
* **Target Path:** `backend/benchmarks/test_queries.json`
* **Categories Required:**
  * `in_domain` (factual grounded queries)
  * `ambiguous` (broad / multi-facet queries)
  * `no_answer` (ungrounded queries to verify low-relevance refusal)
  * `off_topic` (weather, general banter, etc.)
  * `prompt_injection` (adversarial jailbreak attempts)
  * `toxic_unsafe` (hostile input phrases)
* **Query Format:**
  ```json
  [
    {
      "id": "eval_001",
      "query": "What was the Manhattan Project?",
      "language": "en",
      "category": "in_domain",
      "expected_status": "SUCCESS",
      "expected_answer_keywords": ["nuclear", "weapons", "World War II"]
    }
  ]
  ```

### 5. Ground Truth Relevance Annotations
* Mapping of `query_id -> [relevant_passage_ids]` for automated Precision@K, Recall@K, and MRR calculations.

### 6. Vector Index Specification
* Distance Metric: `Cosine`
* Vector Dimension: `384` (for `BAAI/bge-small-en-v1.5`)
* Recommended HNSW parameters: `m=16`, `ef_construct=100`

---

## 4. How to Switch from Demo to Production

1. Place the new dataset in `backend/data/` or set `DATASET_PATH`.
2. Update `.env`:
   ```env
   DATA_MODE=production
   QDRANT_COLLECTION=msmarco_xi_bge_small
   ```
3. Run the ingestion pipeline:
   ```bash
   python backend/ingestion/indexer.py
   ```
4. Run the official benchmark suite:
   ```bash
   python backend/benchmarks/benchmark_pipeline.py
   ```
5. All FastAPI endpoints (`POST /api/query`, `POST /api/voice/query`, `WS /api/voice/ws`) automatically serve the production dataset without code modifications.
