"""Report generator translating benchmark results into clean Markdown."""

import json
import os
from typing import Any, Dict


def generate_markdown_report(results_json_path: str, output_md_path: str) -> str:
    """Generate Markdown report from benchmark JSON results."""
    if not os.path.exists(results_json_path):
        raise FileNotFoundError(f"Results file not found: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    stage_p = data.get("stage_percentiles_ms", {})
    tot = stage_p.get("total", {})
    emb = stage_p.get("embedding", {})
    ret = stage_p.get("retrieval", {})
    grd = stage_p.get("guardrail", {})
    llm = stage_p.get("llm_ttft", {})

    target_met = data.get("target_met", True)
    status_badge = "✅ **PASS (< 200ms)**" if target_met else "⚠️ **EXCEEDED TARGET**"

    lines = [
        "# RAG Pipeline Latency & Benchmark Report",
        "",
        f"**Generated:** {data.get('timestamp', 'N/A')}  ",
        f"**Target P50:** < 200ms  ",
        f"**Achieved P50:** **{data.get('achieved_p50_ms', 0.0)} ms** — {status_badge}  ",
        f"**Total Queries Executed:** {data.get('total_queries_executed', 0)} ({data.get('unique_queries_count', 0)} distinct queries)  ",
        "",
        "---",
        "",
        "## 1. Latency Percentiles Breakdown (ms)",
        "",
        "| Pipeline Stage | P50 (ms) | P70 (ms) | P90 (ms) | P100 / Max (ms) | Avg (ms) | Min (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Embedding** (FastEmbed ONNX) | {emb.get('p50', 0)} | {emb.get('p70', 0)} | {emb.get('p90', 0)} | {emb.get('p100', 0)} | {emb.get('avg', 0)} | {emb.get('min', 0)} |",
        f"| **Retrieval** (Qdrant Cosine) | {ret.get('p50', 0)} | {ret.get('p70', 0)} | {ret.get('p90', 0)} | {ret.get('p100', 0)} | {ret.get('avg', 0)} | {ret.get('min', 0)} |",
        f"| **Guardrails** (Pre & Post Checks) | {grd.get('p50', 0)} | {grd.get('p70', 0)} | {grd.get('p90', 0)} | {grd.get('p100', 0)} | {grd.get('avg', 0)} | {grd.get('min', 0)} |",
        f"| **LLM TTFT** (Groq Llama 3.1) | {llm.get('p50', 0)} | {llm.get('p70', 0)} | {llm.get('p90', 0)} | {llm.get('p100', 0)} | {llm.get('avg', 0)} | {llm.get('min', 0)} |",
        f"| **Total Pipeline (E2E)** | **{tot.get('p50', 0)}** | **{tot.get('p70', 0)}** | **{tot.get('p90', 0)}** | **{tot.get('p100', 0)}** | **{tot.get('avg', 0)}** | **{tot.get('min', 0)}** |",
        "",
        "---",
        "",
        "## 2. Representative Query Performance",
        "",
        "| Language | Query | Expected Topic | Confidence | Citations | Avg Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for q in data.get("query_details", []):
        query_esc = q["query"].replace("|", "\\|")
        lines.append(
            f"| `{q['language']}` | {query_esc} | `{q['expected_topic']}` | {q['confidence_score']:.2f} | {q['citations_count']} | {q['avg_latency_ms']} ms |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Chunking Strategy Decision",
        "",
        "- **Selected Strategy**: `SemanticSentenceChunker` (Strategy B)",
        "- **Rationale**:",
        "  1. Preserves complete grammatical thoughts and avoids splitting critical facts across arbitrary token windows.",
        "  2. Handles Indic punctuation (including danda `।` and `॥`) across multilingual MSMARCO-XI data.",
        "  3. Delivers high retrieval precision and lowest query embedding noise.",
        "",
    ])

    report_md = "\n".join(lines)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return report_md


if __name__ == "__main__":
    res_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_report.md")
    if os.path.exists(res_path):
        generate_markdown_report(res_path, out_path)
        print("Generated benchmark report at:", out_path)
    else:
        print("No benchmark_results.json found. Run benchmark_pipeline.py first.")
