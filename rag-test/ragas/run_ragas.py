"""
RAGAS evaluation — LLM-as-judge metrics for the TechCorp RAG pipeline.

Collects model answers + retrieved contexts via the chat API, then runs
RAGAS (Faithfulness, Answer Correctness, Context Recall, Context Precision)
with gpt-oss-120b as the judge LLM.

Usage:
    python rag-test/ragas/run_ragas.py                              # all questions
    python rag-test/ragas/run_ragas.py --range Q01-Q10              # range
    python rag-test/ragas/run_ragas.py --range Q05                  # single
    python rag-test/ragas/run_ragas.py --evaluate-only results/run_001/dataset.json

Output:
  rag-test/ragas/results/run_XXX/
    ├── dataset.json          (collected data, replayable)
    ├── results.json          (RAGAS scores + summary)
    └── logs/
        └── ragas_log_Q01.json
  rag-test/ragas/comparison_table.md
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
JUDGE_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "llama-3.3-70b-versatile"
CHAT_URL = "http://localhost:8000/api/v1/chat/stream"
DB_CONTAINER = "groq_stream_db"
USER_ID = "techcorp-test"
RATE_LIMIT_S = 30
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
QUESTIONS_FILE = BASE_DIR.parent / "techcorp" / "questions.json"
RESULTS_DIR = BASE_DIR / "results"
TABLE_FILE = BASE_DIR / "comparison_table.md"

SKIP_CATEGORIES = {"absent_plausible", "trap"}

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


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    if "," in q_range:
        nums = {int(p.strip().lstrip("Qq")) for p in q_range.split(",")}
        return [q for q in all_qs if int(q["q_id"].lstrip("Q")) in nums]
    if "-" in q_range:
        lo, hi = q_range.split("-", 1)
        lo_n, hi_n = int(lo.lstrip("Qq")), int(hi.lstrip("Qq"))
        return [q for q in all_qs if lo_n <= int(q["q_id"].lstrip("Q")) <= hi_n]
    n = int(q_range.lstrip("Qq"))
    return [q for q in all_qs if int(q["q_id"].lstrip("Q")) == n]


def _trunc(s: str | None, n: int) -> str:
    if not s:
        return "—"
    s = s.replace("\n", " ").replace("|", "\\|")
    return s[:n] + ("…" if len(s) > n else "")


# ── Chat API ─────────────────────────────────────────────────────────────────

def call_chat(question: str, conversation_id: str) -> tuple[str, float]:
    payload = {
        "messages": [{"role": "user", "content": question}],
        "user_id": USER_ID,
        "conversation_id": conversation_id,
        "model": CHAT_MODEL,
    }
    parts: list[str] = []
    t0 = time.monotonic()
    with httpx.Client(timeout=180.0) as client:
        with client.stream("POST", CHAT_URL, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
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


# ── DB helpers ───────────────────────────────────────────────────────────────

def fetch_chat_log(conversation_id: str) -> dict:
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
    lines = [l for l in raw.splitlines() if not l.startswith("WARNING")]
    clean = "\n".join(lines).strip()
    if not clean:
        return {}
    parts = clean.split("|", 1)
    model_name = parts[0].strip() if parts else ""
    messages_raw = parts[1].strip() if len(parts) > 1 else ""
    try:
        messages = json.loads(messages_raw) if messages_raw else []
    except json.JSONDecodeError:
        messages = []
    return {"model_name": model_name, "messages": messages}


def extract_rag_query(messages: list[dict]) -> str | None:
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


def extract_rag_contexts(messages: list[dict]) -> list[str]:
    for m in messages:
        if m.get("role") != "tool" or m.get("name") != "rag_search":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if "No relevant information" in content:
            return []
        return parse_contexts(content)
    return []


def parse_contexts(raw: str) -> list[str]:
    blocks = [b.strip() for b in raw.split("\n---\n") if b.strip()]
    contexts: list[str] = []
    for block in blocks:
        match = re.search(r'Content:\s*"(.+)"', block, re.DOTALL)
        if match:
            contexts.append(match.group(1).strip())
        else:
            lines = block.split("\n")
            text_lines = [
                l for l in lines
                if not l.startswith("Source:") and not l.startswith("Similarity:")
                and not l.startswith("Rerank-score:")
            ]
            text = "\n".join(text_lines).strip()
            if text:
                contexts.append(text)
    return contexts


def detect_not_in_document(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in NOT_FOUND_PHRASES)


# ── Data collection ──────────────────────────────────────────────────────────

def collect_data(questions: list[dict], run_id: str) -> list[dict]:
    collected: list[dict] = []
    total = len(questions)

    for i, q in enumerate(questions):
        q_id = q["q_id"]
        category = q.get("category", "")
        conv_id = f"ragas-{run_id}-{q_id}"
        should_skip = category in SKIP_CATEGORIES

        print(f"\n[{i + 1}/{total}] {q_id} ({q.get('difficulty', '?')} / {category})")
        print(f"  Q: {q['question'][:80]}")

        if should_skip:
            print(f"  SKIP (category={category}) — will evaluate separately")
            collected.append({
                "q_id": q_id,
                "question": q["question"],
                "answer": "",
                "contexts": [],
                "ground_truth": q.get("expected_answer", ""),
                "rag_query": None,
                "rag_called": False,
                "category": category,
                "difficulty": q.get("difficulty"),
                "ragas_skipped": True,
                "skip_reason": "absent_plausible/trap",
                "not_in_document_said": None,
            })
            continue

        try:
            answer, latency = call_chat(q["question"], conv_id)
        except Exception as exc:
            print(f"  ERROR (chat): {exc}")
            answer, latency = f"[ERROR: {exc}]", 0.0

        time.sleep(2.0)
        log_data = fetch_chat_log(conv_id)
        messages = log_data.get("messages") or []

        rag_query = extract_rag_query(messages)
        contexts = extract_rag_contexts(messages)
        rag_called = rag_query is not None

        skip_ragas = not rag_called or len(contexts) == 0
        skip_reason = None
        if not rag_called:
            skip_reason = "rag_not_called"
        elif len(contexts) == 0:
            skip_reason = "empty_contexts"

        item = {
            "q_id": q_id,
            "question": q["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": q.get("expected_answer", ""),
            "rag_query": rag_query,
            "rag_called": rag_called,
            "category": category,
            "difficulty": q.get("difficulty"),
            "ragas_skipped": skip_ragas,
            "skip_reason": skip_reason,
            "not_in_document_said": None,
        }
        collected.append(item)

        print(f"  rag_called={rag_called}  contexts={len(contexts)}  skip={skip_ragas}")
        print(f"  answer: {answer[:120]}")
        print(f"  ({latency}s)")

        if i < total - 1:
            print(f"\n  Waiting {RATE_LIMIT_S}s...")
            time.sleep(RATE_LIMIT_S)

    return collected


# ── RAGAS evaluation ─────────────────────────────────────────────────────────

def run_ragas_eval(collected: list[dict]) -> dict[str, dict]:
    from openai import AsyncOpenAI
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory
    from ragas.metrics import (
        Faithfulness,
        AnswerCorrectness,
        ContextRecall,
        ContextPrecision,
    )

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    groq_client = AsyncOpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
    judge_llm = llm_factory(JUDGE_MODEL, provider="openai", client=groq_client)

    emb_base = os.environ.get("EMBEDDING_BASE_URL")
    emb_key = os.environ.get("EMBEDDING_API_KEY")
    judge_embeddings = None
    if emb_base and emb_key:
        emb_client = AsyncOpenAI(api_key=emb_key, base_url=emb_base)
        judge_embeddings = embedding_factory("openai", model="nomic-embed-text", client=emb_client)

    metrics = [Faithfulness(), ContextRecall(), ContextPrecision()]
    if judge_embeddings:
        metrics.append(AnswerCorrectness(embeddings=judge_embeddings))
    else:
        metrics.append(AnswerCorrectness())

    evaluable = [item for item in collected if not item["ragas_skipped"]]
    if not evaluable:
        print("No evaluable questions — all skipped.")
        return {}

    print(f"\nRunning RAGAS evaluation on {len(evaluable)} questions (one at a time)...")
    print(f"  Judge: {JUDGE_MODEL}")
    print(f"  Metrics: {[m.name for m in metrics]}")

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=1, timeout=180, max_retries=5)

    scores_by_qid: dict[str, dict] = {}
    for i, item in enumerate(evaluable):
        q_id = item["q_id"]
        print(f"  [{i + 1}/{len(evaluable)}] Evaluating {q_id}...", end=" ", flush=True)

        sample = SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            reference=item["ground_truth"],
        )
        dataset = EvaluationDataset(samples=[sample])

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            run_config=run_config,
        )

        df = result.to_pandas()
        row = df.iloc[0]
        scores = {
            "faithfulness": _safe_float(row.get("faithfulness")),
            "answer_correctness": _safe_float(row.get("answer_correctness")),
            "context_recall": _safe_float(row.get("context_recall")),
            "context_precision": _safe_float(row.get("context_precision")),
        }
        scores_by_qid[q_id] = scores
        print(
            f"faith={scores['faithfulness']}  correct={scores['answer_correctness']}  "
            f"recall={scores['context_recall']}  precision={scores['context_precision']}"
        )

        if i < len(evaluable) - 1:
            time.sleep(RATE_LIMIT_S)

    return scores_by_qid


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        import math
        f = float(val)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ── Summary ──────────────────────────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    evaluated = [r for r in results if not r.get("ragas_skipped")]
    skipped = [r for r in results if r.get("ragas_skipped")]

    def _avg(key: str) -> float | None:
        vals = [
            r["scores"][key] for r in evaluated
            if r.get("scores") and r["scores"].get(key) is not None
        ]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_category: dict[str, dict] = {}
    for r in evaluated:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"count": 0, "faithfulness": [], "answer_correctness": []}
        by_category[cat]["count"] += 1
        if r.get("scores"):
            f = r["scores"].get("faithfulness")
            a = r["scores"].get("answer_correctness")
            if f is not None:
                by_category[cat]["faithfulness"].append(f)
            if a is not None:
                by_category[cat]["answer_correctness"].append(a)

    by_cat_summary = {}
    for cat, data in by_category.items():
        by_cat_summary[cat] = {
            "count": data["count"],
            "avg_faithfulness": (
                round(sum(data["faithfulness"]) / len(data["faithfulness"]), 4)
                if data["faithfulness"] else None
            ),
            "avg_answer_correctness": (
                round(sum(data["answer_correctness"]) / len(data["answer_correctness"]), 4)
                if data["answer_correctness"] else None
            ),
        }

    absent_results = [r for r in skipped if r.get("category") in SKIP_CATEGORIES]
    absent_passed = sum(1 for r in absent_results if r.get("not_in_document_said") is True)

    return {
        "total": len(results),
        "ragas_evaluated": len(evaluated),
        "ragas_skipped": len(skipped),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_answer_correctness": _avg("answer_correctness"),
        "avg_context_recall": _avg("context_recall"),
        "avg_context_precision": _avg("context_precision"),
        "absent_plausible_total": len(absent_results),
        "absent_plausible_passed": absent_passed,
        "by_category": by_cat_summary,
    }


# ── Comparison table ─────────────────────────────────────────────────────────

def append_to_table(run_id: str, results: list[dict], summary: dict) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    header = [
        f"\n## {run_id} — {timestamp} | chat: {CHAT_MODEL} | judge: {JUDGE_MODEL}",
        (
            f"**Avg: Faith={summary['avg_faithfulness']} | "
            f"Correct={summary['avg_answer_correctness']} | "
            f"Recall={summary['avg_context_recall']} | "
            f"Precision={summary['avg_context_precision']}**\n"
        ),
        "| Q | Diff | Category | Faith | Correct | Recall | Precision | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]

    rows: list[str] = []
    for r in results:
        q_id = r["q_id"]
        diff = r.get("difficulty") or "—"
        cat = r.get("category") or "—"

        if r.get("ragas_skipped"):
            reason = r.get("skip_reason", "skipped")
            nf = r.get("not_in_document_said")
            if r.get("category") in SKIP_CATEGORIES:
                status = "✅" if nf else ("❌" if nf is False else "⏭️")
                nf_str = f"not_found={'yes' if nf else 'no'}" if nf is not None else "—"
                rows.append(
                    f"| {q_id} | {diff} | {cat} | — | — | — | — | {status} {nf_str} |"
                )
            else:
                rows.append(
                    f"| {q_id} | {diff} | {cat} | — | — | — | — | ⏭️ {reason} |"
                )
            continue

        s = r.get("scores") or {}
        faith = s.get("faithfulness")
        correct = s.get("answer_correctness")
        recall = s.get("context_recall")
        precision = s.get("context_precision")

        def _fmt(v: float | None) -> str:
            return f"{v:.2f}" if v is not None else "—"

        is_good = (
            (faith is not None and faith >= 0.8)
            and (correct is not None and correct >= 0.6)
        )
        status = "✅" if is_good else "❌"

        rows.append(
            f"| {q_id} | {diff} | {cat} "
            f"| {_fmt(faith)} | {_fmt(correct)} | {_fmt(recall)} | {_fmt(precision)} "
            f"| {status} |"
        )

    if not TABLE_FILE.exists():
        TABLE_FILE.write_text(
            "# TechCorp — RAGAS Evaluation\n\n"
            "LLM-as-judge metrics. Faith = Faithfulness, "
            "Correct = Answer Correctness, "
            "Recall = Context Recall, Precision = Context Precision.\n",
            encoding="utf-8",
        )

    with open(TABLE_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(header + rows) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for TechCorp RAG pipeline")
    parser.add_argument("--range", default=None, help="Question range (e.g. Q01-Q10, Q05)")
    parser.add_argument(
        "--evaluate-only", default=None, metavar="DATASET_JSON",
        help="Skip data collection; evaluate an existing dataset.json",
    )
    args = parser.parse_args()

    run_id = next_run_id()
    run_dir = RESULTS_DIR / run_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = run_dir / "dataset.json"
    results_file = run_dir / "results.json"

    # ── Phase 1: Collect or load ─────────────────────────────────────────
    if args.evaluate_only:
        src = Path(args.evaluate_only)
        if not src.exists():
            src = RESULTS_DIR / args.evaluate_only
        if not src.exists():
            print(f"ERROR: dataset not found: {args.evaluate_only}")
            sys.exit(1)
        collected = json.loads(src.read_text(encoding="utf-8"))
        print(f"Loaded {len(collected)} items from {src}")
        dataset_file.write_text(
            json.dumps(collected, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        questions = load_questions(args.range)
        print(f"Run:       {run_id}")
        print(f"Chat:      {CHAT_MODEL}")
        print(f"Judge:     {JUDGE_MODEL}")
        print(f"Questions: {len(questions)}")
        print(f"Output:    {run_dir}")
        print("=" * 70)

        collected = collect_data(questions, run_id)
        dataset_file.write_text(
            json.dumps(collected, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nDataset saved: {dataset_file}")

    # ── Phase 2: Evaluate with RAGAS ─────────────────────────────────────
    scores_by_qid = run_ragas_eval(collected)

    # ── Phase 3: Merge scores + save results ─────────────────────────────
    final_results: list[dict] = []
    for item in collected:
        q_id = item["q_id"]
        entry = {
            "q_id": q_id,
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "ragas_skipped": item.get("ragas_skipped", False),
            "skip_reason": item.get("skip_reason"),
            "rag_called": item.get("rag_called"),
            "scores": scores_by_qid.get(q_id),
        }

        if item.get("category") in SKIP_CATEGORIES:
            if item.get("answer"):
                entry["not_in_document_said"] = detect_not_in_document(item["answer"])
            else:
                entry["not_in_document_said"] = None
            item["not_in_document_said"] = entry["not_in_document_said"]

        final_results.append(entry)

        log_file = logs_dir / f"ragas_log_{q_id}.json"
        log_file.write_text(
            json.dumps(
                {
                    "q_id": q_id,
                    "question": item["question"],
                    "model_answer": item["answer"],
                    "ground_truth": item["ground_truth"],
                    "contexts": item["contexts"],
                    "rag_query": item.get("rag_query"),
                    "scores": entry.get("scores"),
                    "ragas_skipped": entry["ragas_skipped"],
                    "skip_reason": entry.get("skip_reason"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    summary = compute_summary(final_results)

    run_meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "chat_model": CHAT_MODEL,
        "judge_model": JUDGE_MODEL,
        "results": final_results,
        "summary": summary,
    }
    results_file.write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    append_to_table(run_id, final_results, summary)

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Chat model:             {CHAT_MODEL}")
    print(f"  Judge model:            {JUDGE_MODEL}")
    print(f"  Total questions:        {summary['total']}")
    print(f"  RAGAS evaluated:        {summary['ragas_evaluated']}")
    print(f"  RAGAS skipped:          {summary['ragas_skipped']}")
    print(f"  Avg Faithfulness:       {summary['avg_faithfulness']}")
    print(f"  Avg Answer Correctness: {summary['avg_answer_correctness']}")
    print(f"  Avg Context Recall:     {summary['avg_context_recall']}")
    print(f"  Avg Context Precision:  {summary['avg_context_precision']}")
    if summary["absent_plausible_total"] > 0:
        print(
            f"  Absent/Trap pass rate:  "
            f"{summary['absent_plausible_passed']}/{summary['absent_plausible_total']}"
        )
    print(f"\n  Results: {results_file}")
    print(f"  Table:   {TABLE_FILE}")


if __name__ == "__main__":
    main()
