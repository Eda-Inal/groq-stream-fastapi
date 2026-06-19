"""
Single-question RAG retrieval debug — tests Q01 only.

Output:
  rag-test/techcorp/results/run_XXX/
    ├── results.json
    └── logs/
        └── exploration_log_Q01.json
"""

import httpx
import json
from datetime import datetime
from pathlib import Path

DEBUG_URL = "http://localhost:8001/tools/rag_debug"
USER_ID = "techcorp-test"
TOP_K = 5

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


def load_question(q_id: str = "Q01") -> dict:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    for q in questions:
        if q["q_id"] == q_id:
            return q
    raise ValueError(f"Question {q_id} not found")


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

    # Per-leg recall
    dense_leg = breakdown.get("dense_leg", [])
    sparse_leg = breakdown.get("sparse_leg", [])

    dense_indices = {e.get("chunk_index") for e in dense_leg}
    sparse_indices = {e.get("chunk_index") for e in sparse_leg}
    grep_chunk_indices = set()
    for entry in pre_rerank:
        if entry.get("grep_contrib", 0) > 0:
            grep_chunk_indices.add(entry.get("chunk_index"))

    dense_found = sorted(c for c in expected_list if c in dense_indices)
    sparse_found = sorted(c for c in expected_list if c in sparse_indices)
    grep_found = sorted(c for c in expected_list if c in grep_chunk_indices)

    all_retrieved = set(pre_indices) | set(post_indices)
    found_chunks = sorted(c for c in expected_list if c in all_retrieved)
    chunk_recall = len(found_chunks) / len(expected_list) if expected_list else None

    rank_before = None
    correct_dense = None
    for i, entry in enumerate(pre_rerank):
        if entry.get("chunk_index") in expected_set:
            rank_before = i + 1
            correct_dense = entry.get("cosine_similarity")
            break

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


def main():
    q = load_question("Q01")
    run_id = next_run_id()
    run_dir = RESULTS_DIR / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_id}")
    print(f"Output: {run_dir}\n")
    print(f"  q_id     : {q['q_id']}")
    print(f"  category : {q['category']}")
    print(f"  question : {q['question']}")
    print(f"  expected chunk_index: {q['chunk_index']}")

    print("\nCalling /tools/rag_debug...")
    data = call_debug(q["question"])

    # Save exploration log
    log_file = logs_dir / f"exploration_log_{q['q_id']}.json"
    log_file.write_text(
        json.dumps({"question": q, "debug_response": data}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Log saved: {log_file}")

    # Extract and save results.json
    result = extract_result(q, data)

    run_meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": data.get("config", {}),
        "results": [result],
    }

    results_file = run_dir / "results.json"
    results_file.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Results saved: {results_file}")

    # Print summary
    print(f"\n  correct_chunk_found : {result['correct_chunk_found']}")
    print(f"  expected_chunks     : {result['expected_chunks']}")
    print(f"  found_chunks        : {result['found_chunks']}")
    print(f"  chunk_recall        : {result['chunk_recall']}")
    print(f"  rank_before_rerank  : {result['rank_before_rerank']}")
    print(f"  rank_after_rerank   : {result['rank_after_rerank']}")
    print(f"  rerank_improved     : {result['rerank_improved']}")
    print(f"  pre_rerank_top5     : {result['pre_rerank_top5']}")
    print(f"  post_rerank_top5    : {result['post_rerank_top5']}")


if __name__ == "__main__":
    main()
