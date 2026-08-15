# DEMO DATA — NOT FINAL MSMARCO-XI DATA

> **IMPORTANT NOTICE:**  
> This directory contains a lightweight, curated **DEMO DATASET** used solely for backend pipeline verification, integration testing, and local benchmarking prior to final delivery from the Data Analyst team.  
> **DO NOT** claim benchmark numbers generated from this dataset as official project evaluation results.

---

## Dataset Overview

* **File:** [`demo_msmarco_xi.jsonl`](file:///d:/Voice-Enabled%20RAG/backend/data/demo/demo_msmarco_xi.jsonl)
* **Total Records:** 40
* **Languages Covered:**
  * English (`en`)
  * Hindi (`hi`)
  * Marathi (`mr`)
  * Hinglish / Code-Switched (`hi` / `en`)
* **Categories:**
  1. `in_domain`: Factual knowledge passages (ISRO, Manhattan Project, Photosynthesis, Qdrant, FastEmbed, Chandrayaan, Aryabhata, Machine Learning).
  2. `ambiguous`: Broad informational queries requiring semantic retrieval.
  3. `no_answer`: Intentionally out-of-domain / ungrounded queries to test low-score rejection.
  4. `off_topic`: Weather, game coding, and relationship advice to test Tier 2 scope guardrails.
  5. `prompt_injection`: Adversarial system prompt overrides, jailbreaks, and SQL injection tests for Tier 1 safety guardrails.
  6. `toxic_unsafe`: Safe synthetic test phrases for toxic input rejection.

---

## Record Schema

```json
{
  "id": "demo_en_001",
  "language": "en",
  "query": "What was the Manhattan Project?",
  "answer": "The Manhattan Project was a research program...",
  "title": "Manhattan Project Overview",
  "category": "in_domain",
  "data_mode": "DEMO DATA — NOT FINAL MSMARCO-XI DATA",
  "passages": [
    {
      "passage_id": "demo_en_p001",
      "text": "The Manhattan Project was a research and development undertaking...",
      "is_selected": true
    }
  ]
}
```
