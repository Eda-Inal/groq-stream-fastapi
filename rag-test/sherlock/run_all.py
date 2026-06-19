"""
Full RAG retrieval test — runs questions from questions.json.

Usage:
    python rag-test/sherlock/run_all.py                    # all questions
    python rag-test/sherlock/run_all.py --range Q01-Q04    # Q01 through Q04
    python rag-test/sherlock/run_all.py --range Q05        # single question

Output:
  rag-test/sherlock/results/run_XXX/
    ├── results.json
    └── logs/
        ├── exploration_log_Q01.json
        ├── exploration_log_Q02.json
        └── ...
"""

import argparse
import httpx
import json
import time
from datetime import datetime
from pathlib import Path

DEBUG_URL = "http://localhost:8001/tools/rag_debug"
USER_ID = "sherlock-rag-test"
TOP_K = 5
RATE_LIMIT_S = 5

BASE_DIR = Path(__file__).parent
QUESTIONS_FILE = BASE_DIR / "questions.json"
RESULTS_DIR = BASE_DIR / "results"


def next_run_id() -> str:
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_"))
    if not existing:
        return "run_001"
    last_num = int(existing[-1].split("_")[1])
    return f"run_{last_num + 1:03d}"


def load_questions(q_range: str | None = None) -> list[dict]:
    all_qs = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if not q_range:
        return all_qs

    if "-" in q_range:
        start, end = q_range.split("-", 1)
        start_num = int(start.lstrip("Qq"))
        end_num = int(end.lstrip("Qq"))
        return [q for q in all_qs if start_num <= int(q["q_id"].lstrip("Q")) <= end_num]

    num = int(q_range.lstrip("Qq"))
    return [q for q in all_qs if int(q["q_id"].lstrip("Q")) == num]


def call_debug(query: str) -> dict:
    r = httpx.post(
        DEBUG_URL,
        json={"query": query, "user_id": USER_ID, "top_k": TOP_K},
        timeout=120.0,
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    return r.json()


def extract_result(q: dict, data: dict) -> dict:
    if "error" in data:
        return {
            "q_id": q["q_id"],
            "category": q["category"],
            "error": data["error"],
            "correct_chunk_found": None,
            "expected_chunks": [],
            "found_chunks": [],
            "dense_found": [],
            "sparse_found": [],
            "grep_found": [],
            "chunk_recall": None,
            "rank_before_rerank": None,
            "rank_after_rerank": None,
            "rerank_improved": None,
            "correct_chunk_score_dense": None,
            "correct_chunk_score_rerank": None,
            "top1_score_dense": None,
            "top1_score_rerank": None,
            "pre_rerank_top5": [],
            "post_rerank_top5": [],
        }

    breakdown = data.get("search_breakdown", {})
    pre_rerank = breakdown.get("pre_rerank", [])
    post_rerank = data.get("post_rerank") or []

    expected_idx = q.get("chunk_index")
    if expected_idx is None:
        expected_list = []
    elif isinstance(expected_idx, list):
        expected_list = expected_idx
    else:
        expected_list = [expected_idx]
    expected_set = set(expected_list)

    pre_indices = [e.get("chunk_index") for e in pre_rerank]
    post_indices = [e.get("chunk_index") for e in post_rerank]

    # Per-leg recall: which leg found expected chunks
    dense_leg = breakdown.get("dense_leg", [])
    sparse_leg = breakdown.get("sparse_leg", [])
    grep_leg = breakdown.get("grep_leg", [])

    dense_indices = {e.get("chunk_index") for e in dense_leg}
    sparse_indices = {e.get("chunk_index") for e in sparse_leg}
    grep_db_ids = {e.get("chunk_db_id") for e in grep_leg}
    # grep only has chunk_db_id, map via pre_rerank entries
    grep_chunk_indices = set()
    for entry in pre_rerank:
        if entry.get("grep_contrib", 0) > 0:
            grep_chunk_indices.add(entry.get("chunk_index"))

    dense_found = sorted(c for c in expected_list if c in dense_indices)
    sparse_found = sorted(c for c in expected_list if c in sparse_indices)
    grep_found = sorted(c for c in expected_list if c in grep_chunk_indices)

    # Multi-chunk recall: which expected chunks appeared anywhere in pre_rerank or post_rerank
    all_retrieved = set(pre_indices) | set(post_indices)
    found_chunks = sorted(c for c in expected_list if c in all_retrieved)
    chunk_recall = len(found_chunks) / len(expected_list) if expected_list else None

    # First match rank in pre-rerank
    rank_before = None
    correct_dense = None
    for i, entry in enumerate(pre_rerank):
        if entry.get("chunk_index") in expected_set:
            rank_before = i + 1
            correct_dense = entry.get("cosine_similarity")
            break

    # First match rank in post-rerank
    rank_after = None
    correct_rerank = None
    for i, entry in enumerate(post_rerank):
        if entry.get("chunk_index") in expected_set:
            rank_after = i + 1
            correct_rerank = entry.get("rerank_score")
            break

    found = len(found_chunks) > 0

    rerank_improved = None
    if rank_before is not None and rank_after is not None:
        rerank_improved = rank_after < rank_before

    top1_dense = pre_rerank[0].get("cosine_similarity") if pre_rerank else None
    top1_rerank = post_rerank[0].get("rerank_score") if post_rerank else None

    return {
        "q_id": q["q_id"],
        "category": q["category"],
        "correct_chunk_found": found,
        "expected_chunks": expected_list,
        "found_chunks": found_chunks,
        "dense_found": dense_found,
        "sparse_found": sparse_found,
        "grep_found": grep_found,
        "chunk_recall": chunk_recall,
        "rank_before_rerank": rank_before,
        "rank_after_rerank": rank_after,
        "rerank_improved": rerank_improved,
        "correct_chunk_score_dense": correct_dense,
        "correct_chunk_score_rerank": correct_rerank,
        "top1_score_dense": top1_dense,
        "top1_score_rerank": top1_rerank,
        "pre_rerank_top5": pre_indices[:5],
        "post_rerank_top5": post_indices[:5],
    }


def compute_summary(results: list[dict]) -> dict:
    with_expected = [r for r in results if r.get("expected_chunks")]
    found_any = [r for r in with_expected if r["correct_chunk_found"] is True]

    # Recall@5: % of questions where at least 1 expected chunk is in results
    recall_at_5 = len(found_any) / len(with_expected) if with_expected else 0.0

    # MRR: mean of 1/rank_before_rerank (only questions that have a rank)
    rr_values = []
    for r in with_expected:
        rb = r.get("rank_before_rerank")
        if rb is not None:
            rr_values.append(1.0 / rb)
    mrr_pre = sum(rr_values) / len(with_expected) if with_expected else 0.0

    # MRR post-rerank
    rr_post_values = []
    for r in with_expected:
        ra = r.get("rank_after_rerank")
        if ra is not None:
            rr_post_values.append(1.0 / ra)
    mrr_post = sum(rr_post_values) / len(with_expected) if with_expected else 0.0

    # Average cosine similarity of correct chunks
    cosine_values = [r["correct_chunk_score_dense"] for r in results if r.get("correct_chunk_score_dense") is not None]
    avg_correct_cosine = sum(cosine_values) / len(cosine_values) if cosine_values else None

    # Average chunk recall (multi-chunk)
    recall_values = [r["chunk_recall"] for r in results if r.get("chunk_recall") is not None]
    avg_chunk_recall = sum(recall_values) / len(recall_values) if recall_values else None

    # Per-leg recall: in how many questions (with expected) did each leg find at least 1 chunk
    dense_hit = sum(1 for r in with_expected if r.get("dense_found"))
    sparse_hit = sum(1 for r in with_expected if r.get("sparse_found"))
    grep_hit = sum(1 for r in with_expected if r.get("grep_found"))

    rerank_improved = sum(1 for r in results if r.get("rerank_improved") is True)
    rerank_same_or_worse = sum(1 for r in results if r.get("rerank_improved") is False)

    return {
        "total_questions": len(results),
        "total_with_expected": len(with_expected),
        "total_found": len(found_any),
        "recall_at_5": round(recall_at_5, 3),
        "mrr_pre_rerank": round(mrr_pre, 3),
        "mrr_post_rerank": round(mrr_post, 3),
        "avg_correct_cosine": round(avg_correct_cosine, 4) if avg_correct_cosine else None,
        "avg_chunk_recall": round(avg_chunk_recall, 3) if avg_chunk_recall else None,
        "dense_recall": f"{dense_hit}/{len(with_expected)}",
        "sparse_recall": f"{sparse_hit}/{len(with_expected)}",
        "grep_recall": f"{grep_hit}/{len(with_expected)}",
        "rerank_improved_count": rerank_improved,
        "rerank_same_or_worse_count": rerank_same_or_worse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", default=None, help="Question range (e.g. Q01-Q04, Q05)")
    args = parser.parse_args()

    questions = load_questions(args.range)
    run_id = next_run_id()
    run_dir = RESULTS_DIR / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_id}")
    print(f"Questions: {len(questions)}")
    print(f"Output: {run_dir}")
    print("=" * 70)

    run_meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {},
        "results": [],
    }
    results_file = run_dir / "results.json"

    total = len(questions)
    for i, q in enumerate(questions):
        q_id = q["q_id"]
        print(f"\n[{i+1}/{total}] {q_id} ({q['category']})")
        print(f"  Q: {q['question'][:70]}...")
        print(f"  Expected chunk_index: {q.get('chunk_index')}")

        t0 = time.monotonic()
        data = call_debug(q["question"])
        duration = round(time.monotonic() - t0, 1)

        # Save exploration log
        log_file = logs_dir / f"exploration_log_{q_id}.json"
        log_file.write_text(
            json.dumps({"question": q, "debug_response": data}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Update config from first successful response
        if not run_meta["config"] and "config" in data:
            run_meta["config"] = data["config"]

        # Extract summary
        result = extract_result(q, data)
        run_meta["results"].append(result)

        # Save after every question
        results_file.write_text(
            json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Print summary line
        found = result["correct_chunk_found"]
        rb = result["rank_before_rerank"]
        ra = result["rank_after_rerank"]
        recall = result["chunk_recall"]
        err = result.get("error")
        if err:
            print(f"  ERROR: {err}")
        else:
            print(f"  found={found}  recall={recall}  rank_before={rb}  rank_after={ra}  ({duration}s)")

        # Rate limit between questions
        if i < total - 1:
            time.sleep(RATE_LIMIT_S)

    # Compute and save summary
    summary = compute_summary(run_meta["results"])
    run_meta["summary"] = summary
    results_file.write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  Results: {results_file}")


if __name__ == "__main__":
    main()
