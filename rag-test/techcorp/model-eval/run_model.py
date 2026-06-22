"""
Model-based generation eval — tests the full pipeline including LLM generation.

Change MODEL below to switch models between runs.

Usage:
    python rag-test/techcorp/model-eval/run_model.py                  # all questions
    python rag-test/techcorp/model-eval/run_model.py --range Q31-Q34  # range
    python rag-test/techcorp/model-eval/run_model.py --range Q33      # single

Output:
  rag-test/techcorp/model-eval/results/run_XXX/
    ├── results.json
    └── logs/
        └── eval_log_Q01.json   (full answer + rag query + messages)
  rag-test/techcorp/model-eval/comparison_table.md  (appended after each run)
"""

import argparse
import httpx
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "openai/gpt-oss-120b"        # change this to switch models
USER_ID = "techcorp-test"           # must match the user who uploaded the document
RATE_LIMIT_S = 60                   # seconds between questions (1 question/min)
CHAT_URL = "http://localhost:8000/api/v1/chat/stream"
DB_CONTAINER = "groq_stream_db"
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
QUESTIONS_FILE = BASE_DIR.parent / "questions.json"
RESULTS_DIR = BASE_DIR / "results"
TABLE_FILE = BASE_DIR / "comparison_table.md"

NOT_FOUND_PHRASES = [
    "not mentioned", "not addressed", "not covered", "not described",
    "not in the", "not in this", "no mention", "does not mention",
    "does not address", "does not cover", "no information about",
    "handbook does not", "policy does not", "not available",
    "not found in", "not specified", "not provided", "not discussed",
    "not included", "no details", "cannot find", "could not find",
    "this information is not", "no such policy", "no policy",
    "not part of", "not stated",
]


# ── File helpers ──────────────────────────────────────────────────────────────

def next_run_id() -> str:
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted(
        p.name for p in RESULTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )
    if not existing:
        return "run_001"
    return f"run_{int(existing[-1].split('_')[1]) + 1:03d}"


def load_questions(q_range: str | None = None) -> list[dict]:
    all_qs = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if not q_range:
        return all_qs
    if "-" in q_range:
        start, end = q_range.split("-", 1)
        lo, hi = int(start.lstrip("Qq")), int(end.lstrip("Qq"))
        return [q for q in all_qs if lo <= int(q["q_id"].lstrip("Q")) <= hi]
    num = int(q_range.lstrip("Qq"))
    return [q for q in all_qs if int(q["q_id"].lstrip("Q")) == num]


# ── Chat call ─────────────────────────────────────────────────────────────────

def call_chat(question: str, conversation_id: str) -> tuple[str, float]:
    """Call the streaming chat endpoint and collect the full response."""
    payload = {
        "messages": [{"role": "user", "content": question}],
        "user_id": USER_ID,
        "conversation_id": conversation_id,
        "model": MODEL,
    }
    parts: list[str] = []
    t0 = time.monotonic()

    with httpx.Client(timeout=180.0) as client:
        with client.stream("POST", CHAT_URL, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content") or ""
                    )
                    parts.append(content)
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass

    return "".join(parts), round(time.monotonic() - t0, 1)


# ── DB helpers ────────────────────────────────────────────────────────────────

def fetch_chat_log(conversation_id: str) -> dict:
    """
    Query chat_logs for this conversation and return
    {model_name, messages} or empty dict on failure.
    """
    safe = conversation_id.replace("'", "''")
    sql = (
        f"SELECT model_name, messages::text FROM chat_logs "
        f"WHERE conversation_id = '{safe}' "
        f"ORDER BY created_at DESC LIMIT 1;"
    )
    result = subprocess.run(
        [
            "docker", "exec", DB_CONTAINER,
            "psql", "-U", "app", "-d", "app", "-t", "-A", "-c", sql,
        ],
        capture_output=True, text=True, timeout=15, encoding="utf-8",
    )
    raw = result.stdout.strip()
    # Strip PostgreSQL warning lines
    lines = [l for l in raw.splitlines() if not l.startswith("WARNING")]
    clean = "\n".join(lines).strip()
    if not clean:
        return {}
    # psql -A separates columns with |
    parts = clean.split("|", 1)
    model_name = parts[0].strip() if parts else ""
    messages_raw = parts[1].strip() if len(parts) > 1 else ""
    try:
        messages = json.loads(messages_raw) if messages_raw else []
    except json.JSONDecodeError:
        messages = []
    return {"model_name": model_name, "messages": messages}


def extract_rag_query(messages: list[dict]) -> str | None:
    """Find the query string the model sent to rag_search."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") == "rag_search":
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                    return args.get("query")
                except (json.JSONDecodeError, AttributeError):
                    return str(raw)
    return None


def rag_was_called(messages: list[dict]) -> bool:
    return extract_rag_query(messages) is not None


# ── Evaluation ────────────────────────────────────────────────────────────────

def detect_not_in_document(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in NOT_FOUND_PHRASES)


def evaluate(q: dict, answer: str, messages: list[dict]) -> dict:
    category = q.get("category", "")
    expected_keywords = q.get("expected_keywords") or []
    is_absent = category == "absent_plausible"

    rag_query = extract_rag_query(messages)

    if is_absent:
        not_found = detect_not_in_document(answer)
        return {
            "q_id": q["q_id"],
            "difficulty": q.get("difficulty"),
            "category": category,
            "rag_called": rag_query is not None,
            "rag_query_used": rag_query,
            "keywords_found": [],
            "keywords_missing": [],
            "keywords_score": None,
            "not_in_document_said": not_found,
            "pass": not_found,
        }

    # Normalize hyphens/dashes to spaces so "four-year" matches "four years",
    # "four‑year" (U+2011), "four–year" (en-dash) etc.
    def _norm(s: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFKC", s).lower()
        for ch in "-\u2011\u2012\u2013\u2014\u2015\u2010":
            s = s.replace(ch, " ")
        return s

    answer_norm = _norm(answer)
    found = [kw for kw in expected_keywords if _norm(kw) in answer_norm]
    missing = [kw for kw in expected_keywords if _norm(kw) not in answer_norm]
    score = round(len(found) / len(expected_keywords), 2) if expected_keywords else None
    passed = score is not None and score >= 0.8

    return {
        "q_id": q["q_id"],
        "difficulty": q.get("difficulty"),
        "category": category,
        "rag_called": rag_query is not None,
        "rag_query_used": rag_query,
        "keywords_found": found,
        "keywords_missing": missing,
        "keywords_score": score,
        "not_in_document_said": None,
        "pass": passed,
    }


def compute_summary(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("pass") is True)
    failed = sum(1 for r in results if r.get("pass") is False)
    rag_called = sum(1 for r in results if r.get("rag_called") is True)

    scores = [r["keywords_score"] for r in results if r.get("keywords_score") is not None]
    avg_score = round(sum(scores) / len(scores), 3) if scores else None

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0}
        by_category[cat]["total"] += 1
        if r.get("pass") is True:
            by_category[cat]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "rag_called_count": rag_called,
        "avg_keywords_score": avg_score,
        "by_category": by_category,
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def _trunc(s: str | None, n: int) -> str:
    if not s:
        return "—"
    s = s.replace("\n", " ").replace("|", "\\|")
    return s[:n] + ("…" if len(s) > n else "")


def append_to_table(
    run_id: str,
    results: list[dict],
    summary: dict,
    logs_dir: Path,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    pass_pct = round(summary["pass_rate"] * 100)

    header = [
        f"\n## {run_id} — {timestamp} | {MODEL}\n",
        f"**Pass: {summary['passed']}/{summary['total']} ({pass_pct}%) | "
        f"RAG called: {summary['rag_called_count']}/{summary['total']} | "
        f"Avg keyword score: {summary['avg_keywords_score']}**\n",
        "",
        "| Q | Question | Diff | Category | Pass | Score | RAG Query | Model Answer | Expected Answer |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    rows: list[str] = []
    for r in results:
        q_id = r["q_id"]
        diff = r.get("difficulty") or "—"
        cat = r.get("category") or "—"
        passed = r.get("pass")
        status = "✅" if passed is True else ("❌" if passed is False else "❓")

        if r.get("keywords_score") is not None:
            score_str = str(r["keywords_score"])
        elif r.get("not_in_document_said") is not None:
            nf = r["not_in_document_said"]
            score_str = f"not_found={'yes' if nf else 'no'}"
        else:
            score_str = "—"

        rag_q = _trunc(r.get("rag_query_used"), 55)

        # Pull answers from the log file
        log_file = logs_dir / f"eval_log_{q_id}.json"
        model_ans = "—"
        exp_ans = "—"
        if log_file.exists():
            try:
                log_data = json.loads(log_file.read_text(encoding="utf-8"))
                model_ans = _trunc(log_data.get("model_answer"), 100)
                exp_ans = _trunc(log_data.get("expected_answer"), 80)
            except (json.JSONDecodeError, KeyError):
                pass

        # Pull question text from log file
        question_text = "—"
        if log_file.exists():
            try:
                log_data_q = json.loads(log_file.read_text(encoding="utf-8"))
                question_text = _trunc(log_data_q.get("question", {}).get("question"), 70)
            except (json.JSONDecodeError, KeyError, AttributeError):
                pass

        rows.append(
            f"| {q_id} | {question_text} | {diff} | {cat} | {status} | {score_str} "
            f"| {rag_q} | {model_ans} | {exp_ans} |"
        )

    if not TABLE_FILE.exists():
        TABLE_FILE.write_text(
            "# TechCorp — Model Generation Eval\n\n"
            "Score = keyword match ratio (pass ≥ 0.8). "
            "For `absent_plausible`: not_found=yes means model correctly said "
            "'not in document'.\n",
            encoding="utf-8",
        )

    with open(TABLE_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(header + rows) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", default=None, help="e.g. Q31-Q34 or Q33")
    args = parser.parse_args()

    questions = load_questions(args.range)
    run_id = next_run_id()
    run_dir = RESULTS_DIR / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run:       {run_id}")
    print(f"Model:     {MODEL}")
    print(f"Questions: {len(questions)}")
    print(f"Output:    {run_dir}")
    print("=" * 70)

    run_meta: dict = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "results": [],
    }
    results_file = run_dir / "results.json"

    total = len(questions)
    for i, q in enumerate(questions):
        q_id = q["q_id"]
        conv_id = f"{run_id}-{q_id}"

        print(f"\n[{i + 1}/{total}] {q_id} ({q.get('difficulty', '?')} / {q.get('category', '?')})")
        print(f"  Q: {q['question'][:80]}")

        # ── Chat call ────────────────────────────────────────────────────────
        try:
            answer, latency = call_chat(q["question"], conv_id)
        except Exception as exc:
            print(f"  ERROR (chat): {exc}")
            answer, latency = f"[ERROR: {exc}]", 0.0

        # ── DB fetch (give the server time to commit the chat log) ───────────
        time.sleep(2.0)
        log_data = fetch_chat_log(conv_id)
        messages = log_data.get("messages") or []
        actual_model = log_data.get("model_name") or MODEL

        # ── Evaluate ─────────────────────────────────────────────────────────
        result = evaluate(q, answer, messages)
        result["latency_s"] = latency
        result["conversation_id"] = conv_id
        result["actual_model"] = actual_model

        # ── Save log (full detail) ────────────────────────────────────────────
        log_file = logs_dir / f"eval_log_{q_id}.json"
        log_file.write_text(
            json.dumps(
                {
                    "question": q,
                    "model_answer": answer,
                    "expected_answer": q.get("expected_answer"),
                    "rag_query_used": result.get("rag_query_used"),
                    "db_messages_count": len(messages),
                    "result": result,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        run_meta["results"].append(result)
        results_file.write_text(
            json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── Print summary line ────────────────────────────────────────────────
        passed = result.get("pass")
        status = "✅" if passed else "❌"
        score = result.get("keywords_score")
        nf = result.get("not_in_document_said")
        rag_q = result.get("rag_query_used") or "—"

        if score is not None:
            print(f"  {status} score={score}  missing={result.get('keywords_missing')}")
        else:
            print(f"  {status} not_found_said={nf}")
        print(f"  rag_query : {rag_q}")
        print(f"  answer    : {answer[:140]}")
        print(f"  ({latency}s | model: {actual_model})")

        if i < total - 1:
            print(f"\n  Waiting {RATE_LIMIT_S}s...")
            time.sleep(RATE_LIMIT_S)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = compute_summary(run_meta["results"])
    run_meta["summary"] = summary
    results_file.write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    append_to_table(run_id, run_meta["results"], summary, logs_dir)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  Results : {results_file}")
    print(f"  Table   : {TABLE_FILE}")


if __name__ == "__main__":
    main()
