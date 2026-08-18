# Multilingual MSMARCO-XI Data Pipeline & Benchmark Suite

Comprehensive Data Analyst pipeline for the **HH Goa 2026 Task 2 — Voice-Enabled Multilingual RAG System**.

---

## 1. Architecture Overview

```
[ AI4Bharat MSMARCO-XI / JSONL ]
              │
              ▼
   dataset_downloader.py
(Download, Clean, Normalize, Deduplicate)
              │
              ▼
   Clean Normalized Documents
              │
              ▼
   chunking_strategies.py
(A: Semantic | B: Hierarchical | C: Overlap)
              │
              ▼
       Structured Chunks
              │
              ▼
      vector_indexer.py
(FastEmbed ONNX 384d → Qdrant Vector DB)
              │
              ▼
     benchmark_suite.py
(100 Multilingual Queries → P50/P70/P100 Latency Reports)
```

---

## 2. Multilingual Embedding Model

* **Selected Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Embedding Dimension:** `384`
* **Distance Metric:** `Cosine`
* **Why Selected:**
  * Native cross-lingual support across **50+ languages**, specifically optimized for English, Hindi, Marathi, Bengali, Telugu, Tamil, Urdu, Gujarati, and Hinglish.
  * Compact 384-dimensional vector representation enabling sub-millisecond CPU ONNX inference.
  * Direct compatibility with FastEmbed and Qdrant without external API token latency.

---

## 3. Chunking Strategies

| Strategy | Target Budget | Description | Metadata Preserved |
|---|---|---|---|
| **Semantic Chunking** | ~180 words | Sentence-boundary aware using English (`.?!`) and Indic (`।॥\n`) terminators | `document_id`, `chunk_id`, `language`, `strategy`, `position`, `source` |
| **Hierarchical Chunking** | Parent ~512 words, Child ~128 words | Multi-granularity: Child chunks for vector search, Parent context for LLM synthesis | `parent_id`, `parent_text`, `child_id`, `parent_index`, `child_index` |
| **Overlap Chunking** | 200 words (40 word overlap) | Fixed sliding window with word-boundary preservation | `start_word_idx`, `end_word_idx`, `position` |

---

## 4. 100-Query Benchmark Suite (5 Categories)

The benchmark suite strictly validates exact category counts across 100 multilingual queries:

* `in_domain`: **50 queries** (English, Hindi, Marathi, Bengali, Telugu, Tamil, Hinglish)
* `ambiguous`: **15 queries** (Broad multi-concept queries)
* `no_answer`: **15 queries** (Ungrounded out-of-corpus queries)
* `off_topic`: **10 queries** (Weather, jokes, dating, casual banter)
* `prompt_injection`: **10 queries** (Jailbreak, system prompt reveal, developer mode)
* **Total:** **100 queries**

### Metrics Calculated:
* **P50 (Median Latency ms)**
* **P70 Latency ms**
* **P100 (Max Latency ms)**
* Min, Max, Mean, and Standard Deviation
* Stage breakdown: Query Embedding ms, Vector Search ms, Total Retrieval ms

---

## 5. Execution Commands

### A. Run Data Pipeline Unit & Integration Tests
```powershell
$env:PYTHONPATH="voice-rag-goa-2026"
pytest voice-rag-goa-2026/data_pipeline/tests -v
```

### B. Run End-to-End Pipeline
```powershell
$env:PYTHONPATH="voice-rag-goa-2026"
python -m data_pipeline.pipeline_runner --all
```

### C. Run Individual Stages
```powershell
# 1. Dataset Download & Preprocessing
python -m data_pipeline.dataset_downloader --languages en,hi,mr,bn,te,ta,hinglish --max-records 50

# 2. Vector Indexing into Qdrant
python -m data_pipeline.pipeline_runner --index --strategy semantic

# 3. 100-Query Benchmark Execution
python -m data_pipeline.benchmark_suite --mode local
```
