"""
Eval script: send each dataset question to the routing endpoint, get the tool
selected directly, compare with the expected tool, and write results to LangSmith.

Uses POST /api/v1/eval/route — a lightweight endpoint that runs only the routing
LLM call (no tool execution, no final answer). ~3x cheaper than /chat/stream.

Run from project root:
    .venv\Scripts\python eval/run_eval.py
"""

import os
import time
import httpx

# ── env & settings ────────────────────────────────────────────────────────────

def _load_env() -> None:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://placeholder:x@localhost/x")

from langsmith import evaluate  # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────

DATASET_NAME  = "tool-routing-eval-v2"
ROUTE_URL     = "http://localhost:8000/api/v1/eval/route"
HEALTH_URL    = "http://localhost:8000/api/v1/chat/models"
# Set MODEL to any model in AVAILABLE_MODELS. Uses a separate provider so
# eval runs don't consume the chat app's Groq token quota.
MODEL         = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── rate-limit guard ──────────────────────────────────────────────────────────
# llama-4-scout on Groq: ~30 RPM limit. At 2s delay = max 15 req/min — safe.
REQUEST_DELAY = 2.0   # seconds between every question
GROUP_SIZE    = 5     # questions per group
GROUP_PAUSE   = 15.0  # extra seconds after each group

RETRY_MAX          = 4
RETRY_INITIAL_WAIT = 15.0

# ── Filter ────────────────────────────────────────────────────────────────────
# Leave empty to run all questions.
# Add substrings to run only matching questions (case-insensitive).
FILTER: list[str] = []

_results_log: list[dict] = []
_call_count   = 0

# ── helpers ───────────────────────────────────────────────────────────────────

def _call_route(question: str) -> str:
    """
    POST /api/v1/eval/route and return the tool name selected by the routing LLM.
    Returns empty string on HTTP error or rate-limit so caller can retry.
    """
    with httpx.Client(timeout=30) as client:
        r = client.post(ROUTE_URL, json={"question": question, "model": MODEL})
        if r.status_code == 429 or r.status_code >= 500:
            return ""
        r.raise_for_status()
        return r.json().get("tool", "none")


# ── LangSmith target & evaluator ─────────────────────────────────────────────

def target(inputs: dict) -> dict:
    """Call the routing endpoint and return which tool was selected."""
    global _call_count
    _call_count += 1

    question = inputs["question"]

    wait = RETRY_INITIAL_WAIT
    tool_used = ""
    for attempt in range(1, RETRY_MAX + 1):
        tool_used = _call_route(question)
        if tool_used:
            break
        if attempt < RETRY_MAX:
            print(
                f"\n  [rate-limited]  retry {attempt}/{RETRY_MAX - 1} in {wait:.0f}s...",
                flush=True,
            )
            time.sleep(wait)
            wait = min(wait * 2, 60.0)
        else:
            print(
                f"\n  [GIVE UP] {question[:55]!r} — no response after {RETRY_MAX} attempts",
                flush=True,
            )
            tool_used = "none"

    time.sleep(REQUEST_DELAY)

    if _call_count % GROUP_SIZE == 0:
        print(f"\n  [group pause {GROUP_PAUSE:.0f}s after question {_call_count}]", flush=True)
        time.sleep(GROUP_PAUSE)

    return {"question": question, "tool_used": tool_used}


def tool_selection_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs.get("expected_tool", "none")
    actual   = outputs.get("tool_used", "none")
    correct  = (expected == actual)
    question = outputs.get("question", "")[:60]

    icon = "OK  " if correct else "FAIL"
    print(f"  [{icon}]  {question:<62}  expected={expected:<12}  got={actual}")

    _results_log.append({"expected": expected, "actual": actual, "correct": correct})

    return {
        "key": "tool_selection_correct",
        "score": 1 if correct else 0,
        "comment": f"expected={expected} | got={actual}",
    }

# ── main ──────────────────────────────────────────────────────────────────────

def _warmup(retries: int = 3, delay: float = 3.0) -> None:
    """Verify the API is reachable via GET /chat/models. Zero tokens."""
    for attempt in range(1, retries + 1):
        try:
            print(f"Checking API health (attempt {attempt}/{retries})...", end=" ", flush=True)
            httpx.get(HEALTH_URL, timeout=10).raise_for_status()
            print("OK.")
            return
        except Exception as e:
            print(f"failed: {e}")
            if attempt < retries:
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
    raise RuntimeError(f"API health check failed after {retries} attempts. Eval aborted.")


if __name__ == "__main__":
    from langsmith import Client

    _warmup()

    if FILTER:
        client = Client()
        all_examples = list(client.list_examples(dataset_name=DATASET_NAME))
        data = [
            e for e in all_examples
            if any(f.lower() in e.inputs["question"].lower() for f in FILTER)
        ]
        if not data:
            print(f"No questions matched FILTER={FILTER}. Exiting.")
            raise SystemExit(1)
        print("=" * 60)
        print(f"Dataset : {DATASET_NAME}  (filtered: {len(data)}/{len(all_examples)})")
    else:
        data = DATASET_NAME
        print("=" * 60)
        print(f"Dataset : {DATASET_NAME}  (all questions)")

    print(f"Model   : {MODEL}")
    print(f"API     : {ROUTE_URL}")
    print("=" * 60)

    evaluate(
        target,
        data=data,
        evaluators=[tool_selection_evaluator],
        experiment_prefix="tool-routing",
        max_concurrency=1,
    )

    total   = len(_results_log)
    correct = sum(1 for r in _results_log if r["correct"])

    print("\n" + "=" * 60)
    print(f"Overall: {correct}/{total} correct  ({100 * correct // total if total else 0}%)")
    print()
    print(f"  {'Tool':<12}  {'Correct'}")
    print(f"  {'-'*12}  {'-'*10}")
    for tool in ["calculator", "web_search", "rag_search", "none"]:
        subset = [r for r in _results_log if r["expected"] == tool]
        if subset:
            ok = sum(1 for r in subset if r["correct"])
            bar = "#" * ok + "." * (len(subset) - ok)
            print(f"  {tool:<12}  {ok}/{len(subset)}  {bar}")

    print()
    print("View experiment: https://smith.langchain.com > Projects > groq-stream-fastapi")
