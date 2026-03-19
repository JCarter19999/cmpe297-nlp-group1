"""Core logic for eval rag."""
from __future__ import annotations


import re
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any, Dict, List, Sequence

from rag_local.app_core import answer_turn, cfg_int, init_embedder


@dataclass(frozen=True)
class EvalItem:
    """Represents eval item."""
    question: str
    reference_answer: str = ""


def default_eval_items() -> List[EvalItem]:
    """Default eval items."""
    return [
        EvalItem(
            question="What does perplexity represent in language modeling?",
            reference_answer=(
                "Perplexity measures how well a probability model predicts a sequence. "
                "Lower perplexity generally indicates better predictive performance."
            ),
        ),
        EvalItem(
            question="What are two pretraining reasons and one post-training reason for LLM hallucination?",
            reference_answer=(
                "Possible pretraining reasons include noisy or contradictory web data and incomplete coverage. "
                "A post-training reason is reward misalignment or over-optimization for plausible-sounding responses."
            ),
        ),
        EvalItem(
            question="How do skip-gram and CBOW differ?",
            reference_answer=(
                "Skip-gram predicts surrounding context words from a center word, while CBOW predicts the center word "
                "from surrounding context words."
            ),
        ),
    ]


# -------------------------------------------------------------------
# Heuristic scoring helpers
# -------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    """Internal helper for tokenize."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", (text or "").lower())


def _overlap_ratio(a: str, b: str) -> float:
    """Internal helper for overlap ratio."""
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def _contains_citation(text: str) -> bool:
    """Internal helper for contains citation."""
    return bool(re.search(r"\[S\d+\]", text or ""))


def _score_1_to_5(x: float) -> int:
    """Internal helper for score 1 to 5."""
    x = max(0.0, min(1.0, x))
    if x < 0.2:
        return 1
    if x < 0.4:
        return 2
    if x < 0.6:
        return 3
    if x < 0.8:
        return 4
    return 5


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Internal helper for cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = sum(float(x) * float(x) for x in a) ** 0.5
    nb = sum(float(y) * float(y) for y in b) ** 0.5
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def _safe_semantic_similarity(cfg: Any, a: str, b: str) -> float:
    """
    Embedding-based semantic similarity in [0, 1].
    Falls back to lexical overlap if embedding fails.
    """
    ta = (a or "").strip()
    tb = (b or "").strip()
    if not ta or not tb:
        return 0.0

    try:
        embedder = init_embedder(cfg)
        va = embedder.embed_query(ta)
        vb = embedder.embed_query(tb)
        cos = _cosine_similarity(va, vb)
        return (cos + 1.0) / 2.0
    except Exception:
        return _overlap_ratio(ta, tb)


def _score_correctness(answer: str, reference: str) -> int:
    """Internal helper for score correctness."""
    if not reference.strip():
        return 0  # N/A for live conversation mode
    return _score_1_to_5(_overlap_ratio(reference, answer))


def _score_semantic_correctness(cfg: Any, answer: str, reference: str) -> int:
    """Internal helper for score semantic correctness."""
    if not reference.strip():
        return 0
    return _score_1_to_5(_safe_semantic_similarity(cfg, reference, answer))


def _score_relevance(question: str, answer: str) -> int:
    """Internal helper for score relevance."""
    return _score_1_to_5(_overlap_ratio(question, answer))


def _score_groundedness(answer: str, sources: Sequence[Dict[str, Any]]) -> int:
    """Internal helper for score groundedness."""
    if not sources:
        return 1

    source_text = "\n".join((s.get("snippet", "") or "") for s in sources)
    overlap = _overlap_ratio(answer, source_text)

    # Slight bump if answer uses explicit source citations
    if _contains_citation(answer):
        overlap = min(1.0, overlap + 0.1)

    return _score_1_to_5(overlap)


def _score_retrieval_relevance(question: str, sources: Sequence[Dict[str, Any]]) -> int:
    """Internal helper for score retrieval relevance."""
    if not sources:
        return 1

    source_text = "\n".join((s.get("snippet", "") or "") for s in sources)
    return _score_1_to_5(_overlap_ratio(question, source_text))


def _std(vals: Sequence[float]) -> float:
    """Internal helper for std."""
    if len(vals) <= 1:
        return 0.0
    return float(stdev(vals))


def _percentile(vals: Sequence[float], p: float) -> float:
    """Internal helper for percentile."""
    if not vals:
        return 0.0
    xs = sorted(float(v) for v in vals)
    if len(xs) == 1:
        return xs[0]
    rank = (max(0.0, min(100.0, p)) / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _failure_tags(
    *,
    relevance: int,
    groundedness: int,
    retrieval_relevance: int,
    citation_coverage: int,
    sources_count: int,
) -> List[str]:
    """Internal helper for failure tags."""
    tags: List[str] = []
    if sources_count == 0 or retrieval_relevance <= 2:
        tags.append("retrieval_miss")
    if groundedness <= 2:
        tags.append("grounding_risk")
    if relevance <= 2:
        tags.append("off_topic")
    if sources_count > 0 and citation_coverage == 0:
        tags.append("missing_citations")
    return tags


# -------------------------------------------------------------------
# Benchmark / regression eval (existing style)
# -------------------------------------------------------------------
def run_rag_eval(cfg: Any, index: Any, items: Sequence[EvalItem]) -> Dict[str, Any]:
    """Run rag eval."""
    rows: List[Dict[str, Any]] = []
    repeats = max(1, cfg_int(cfg, "rag_eval_repeats", 1))

    for item in items:
        run_rows: List[Dict[str, Any]] = []

        for _ in range(repeats):
            answer, sources, trace = answer_turn(
                history=[],
                user_text=item.question,
                cfg=cfg,
                index=index,
                return_trace=True,
            )

            retrieval_s = float(trace.get("retrieval_s", 0.0) or 0.0)
            generation_s = float(trace.get("generation_s", 0.0) or 0.0)
            total_s = float(trace.get("total_s", retrieval_s + generation_s) or 0.0)

            correctness = _score_correctness(answer, item.reference_answer)
            semantic_correctness = _score_semantic_correctness(cfg, answer, item.reference_answer)
            relevance = _score_relevance(item.question, answer)
            groundedness = _score_groundedness(answer, sources)
            retrieval_relevance = _score_retrieval_relevance(item.question, sources)
            citation_coverage = 1 if _contains_citation(answer) else 0

            run_rows.append(
                {
                    "question": item.question,
                    "correctness": correctness,
                    "semantic_correctness": semantic_correctness,
                    "relevance": relevance,
                    "groundedness": groundedness,
                    "retrieval_relevance": retrieval_relevance,
                    "citation_coverage": citation_coverage,
                    "latency_s": round(total_s, 4),
                    "retrieval_s": round(retrieval_s, 4),
                    "generation_s": round(generation_s, 4),
                    "answer": answer,
                    "sources": list(sources),
                    "mode": "benchmark",
                    "failure_tags": _failure_tags(
                        relevance=relevance,
                        groundedness=groundedness,
                        retrieval_relevance=retrieval_relevance,
                        citation_coverage=citation_coverage,
                        sources_count=len(sources),
                    ),
                }
            )

        correctness_vals = [r["correctness"] for r in run_rows if r.get("correctness", 0) > 0]
        sem_corr_vals = [r["semantic_correctness"] for r in run_rows if r.get("semantic_correctness", 0) > 0]
        relevance_vals = [r["relevance"] for r in run_rows]
        groundedness_vals = [r["groundedness"] for r in run_rows]
        retrieval_vals = [r["retrieval_relevance"] for r in run_rows]
        citation_vals = [r.get("citation_coverage", 0) for r in run_rows]
        latency_vals = [r["latency_s"] for r in run_rows]
        retrieval_s_vals = [r.get("retrieval_s", 0.0) for r in run_rows]
        generation_s_vals = [r.get("generation_s", 0.0) for r in run_rows]

        union_failure_tags = sorted({t for r in run_rows for t in r.get("failure_tags", [])})
        row = {
            "question": item.question,
            "correctness": round(mean(correctness_vals), 2) if correctness_vals else 0.0,
            "semantic_correctness": round(mean(sem_corr_vals), 2) if sem_corr_vals else 0.0,
            "relevance": round(mean(relevance_vals), 2),
            "groundedness": round(mean(groundedness_vals), 2),
            "retrieval_relevance": round(mean(retrieval_vals), 2),
            "citation_coverage": round(mean(citation_vals), 2),
            "latency_s": round(mean(latency_vals), 4),
            "latency_std_s": round(_std(latency_vals), 4),
            "retrieval_s": round(mean(retrieval_s_vals), 4),
            "generation_s": round(mean(generation_s_vals), 4),
            "answer": run_rows[-1]["answer"] if run_rows else "",
            "sources": run_rows[-1]["sources"] if run_rows else [],
            "mode": "benchmark",
            "repeats": repeats,
            "failure_tags": union_failure_tags,
            "sample_runs": run_rows,
        }
        rows.append(row)

    return _summarize_rows(rows)


# -------------------------------------------------------------------
# Live GUI conversation eval
# -------------------------------------------------------------------
def run_conversation_eval(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates actual assistant turns already present in the GUI session.

    Expected message format:
      {"role": "user"|"assistant", "content": "...", ...}
    Assistant messages may contain:
      {"sources": [...], "latency_s": float}
    """
    rows: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        if i == 0:
            continue

        # Find the nearest previous user turn
        question = ""
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") == "user":
                question = messages[j].get("content", "")
                break

        if not question.strip():
            continue

        answer = msg.get("content", "") or ""
        sources = msg.get("sources", []) or []
        latency_s = float(msg.get("latency_s", 0.0) or 0.0)

        row = {
            "turn_index": i,
            "question": question,
            "correctness": 0,  # no gold answer in live chat mode
            "semantic_correctness": 0,
            "relevance": _score_relevance(question, answer),
            "groundedness": _score_groundedness(answer, sources),
            "retrieval_relevance": _score_retrieval_relevance(question, sources),
            "citation_coverage": 1 if _contains_citation(answer) else 0,
            "latency_s": round(latency_s, 4),
            "retrieval_s": round(float(msg.get("retrieval_s", 0.0) or 0.0), 4),
            "generation_s": round(float(msg.get("generation_s", 0.0) or 0.0), 4),
            "answer": answer,
            "sources": list(sources),
            "mode": "conversation",
        }
        row["failure_tags"] = _failure_tags(
            relevance=row["relevance"],
            groundedness=row["groundedness"],
            retrieval_relevance=row["retrieval_relevance"],
            citation_coverage=row["citation_coverage"],
            sources_count=len(sources),
        )
        rows.append(row)

    return _summarize_rows(rows)


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Internal helper for summarize rows."""
    if not rows:
        return {
            "rows": [],
            "summary": {
                "n": 0,
                "correctness_avg": 0.0,
                "relevance_avg": 0.0,
                "groundedness_avg": 0.0,
                "retrieval_relevance_avg": 0.0,
                "citation_coverage_avg": 0.0,
                "latency_avg_s": 0.0,
            },
        }

    correctness_vals = [r["correctness"] for r in rows if r.get("correctness", 0) > 0]
    sem_corr_vals = [r["semantic_correctness"] for r in rows if r.get("semantic_correctness", 0) > 0]
    relevance_vals = [r["relevance"] for r in rows]
    groundedness_vals = [r["groundedness"] for r in rows]
    retrieval_vals = [r["retrieval_relevance"] for r in rows]
    citation_vals = [r.get("citation_coverage", 0) for r in rows]
    latency_vals = [r["latency_s"] for r in rows]
    retrieval_s_vals = [r.get("retrieval_s", 0.0) for r in rows]
    generation_s_vals = [r.get("generation_s", 0.0) for r in rows]
    failure_tag_counts: Dict[str, int] = {}
    for r in rows:
        for t in r.get("failure_tags", []):
            failure_tag_counts[t] = failure_tag_counts.get(t, 0) + 1

    summary = {
        "n": len(rows),
        "correctness_avg": round(mean(correctness_vals), 2) if correctness_vals else 0.0,
        "semantic_correctness_avg": round(mean(sem_corr_vals), 2) if sem_corr_vals else 0.0,
        "relevance_avg": round(mean(relevance_vals), 2),
        "groundedness_avg": round(mean(groundedness_vals), 2),
        "retrieval_relevance_avg": round(mean(retrieval_vals), 2),
        "citation_coverage_avg": round(mean(citation_vals), 2),
        "latency_avg_s": round(mean(latency_vals), 2) if latency_vals else 0.0,
        "latency_p50_s": round(_percentile(latency_vals, 50), 2) if latency_vals else 0.0,
        "latency_p95_s": round(_percentile(latency_vals, 95), 2) if latency_vals else 0.0,
        "retrieval_avg_s": round(mean(retrieval_s_vals), 2) if retrieval_s_vals else 0.0,
        "generation_avg_s": round(mean(generation_s_vals), 2) if generation_s_vals else 0.0,
        "failure_tag_counts": dict(sorted(failure_tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    return {"rows": list(rows), "summary": summary}


def format_eval_report(result: Dict[str, Any]) -> str:
    """Format eval report."""
    rows = result.get("rows", [])
    summary = result.get("summary", {})

    lines = []
    mode = rows[0].get("mode", "unknown") if rows else "unknown"
    lines.append(f"[EVAL] mode={mode}")
    lines.append("[EVAL] # | corr sem rel grd ret cite | latency(s) | tags | question")
    lines.append("[EVAL] " + "-" * 80)

    for idx, row in enumerate(rows, start=1):
        q = row.get("question", "").replace("\n", " ").strip()
        if len(q) > 42:
            q = q[:39] + "..."
        tags = ",".join(row.get("failure_tags", [])) or "-"
        if len(tags) > 18:
            tags = tags[:15] + "..."
        lines.append(
            f"[EVAL] {idx:<2} | "
            f"{row.get('correctness', 0):<4} "
            f"{row.get('semantic_correctness', 0):<3} "
            f"{row.get('relevance', 0):<3} "
            f"{row.get('groundedness', 0):<3} "
            f"{row.get('retrieval_relevance', 0):<3} "
            f"{row.get('citation_coverage', 0):<4} | "
            f"{row.get('latency_s', 0.0):>8.2f} | "
            f"{tags:<18} | "
            f"{q}"
        )

    lines.append("")
    lines.append(
        "[EVAL] averages: "
        f"correctness={summary.get('correctness_avg', 0.0):.2f}, "
        f"semantic_correctness={summary.get('semantic_correctness_avg', 0.0):.2f}, "
        f"relevance={summary.get('relevance_avg', 0.0):.2f}, "
        f"groundedness={summary.get('groundedness_avg', 0.0):.2f}, "
        f"retrieval_relevance={summary.get('retrieval_relevance_avg', 0.0):.2f}, "
        f"citation_coverage={summary.get('citation_coverage_avg', 0.0):.2f}, "
        f"latency(avg/p50/p95)={summary.get('latency_avg_s', 0.0):.2f}/"
        f"{summary.get('latency_p50_s', 0.0):.2f}/"
        f"{summary.get('latency_p95_s', 0.0):.2f}s, "
        f"retrieval_avg={summary.get('retrieval_avg_s', 0.0):.2f}s, "
        f"generation_avg={summary.get('generation_avg_s', 0.0):.2f}s"
    )
    tag_counts = summary.get("failure_tag_counts", {}) or {}
    if tag_counts:
        tags_txt = ", ".join(f"{k}:{v}" for k, v in tag_counts.items())
        lines.append(f"[EVAL] failure_tags: {tags_txt}")
    return "\n".join(lines)