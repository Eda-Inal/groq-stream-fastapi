"""
Eval script: send each dataset question to the API, detect which tool was
selected, compare with the expected tool, and write results to LangSmith.

Run from project root:
    .venv\Scripts\python eval/run_eval.py

Detection logic (response text → tool):
  1. Response mentions "not found in your documents"  → rag  (tried, empty result)
  2. Source: calculator                               → calculator
  3. Source: https?://...                             → web_search
  4. Source: [file.ext] / filename.pdf               → rag
  5. No Source line at all                           → none
"""

import os
import re
import json
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
# Settings requires DATABASE_URL even when we don't touch the DB.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://placeholder:x@localhost/x")

from langsmith import evaluate  # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────

DATASET_NAME = "tool-routing-eval-v1"
API_URL      = "http://localhost:8000/api/v1/chat/stream"
MODEL        = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Filter ────────────────────────────────────────────────────────────────────
# Leave empty to run all 18 questions.
# Add substrings to run only matching questions (case-insensitive).
# Example: FILTER = ["Capital of France", "World War II", "1250"]
FILTER: list[str] = []

_results_log: list[dict] = []

# ── helpers ───────────────────────────────────────────────────────────────────

def _call_api(question: str) -> str:
    """Stream a question to the API and return the full response text."""
    text = ""
    with httpx.Client(timeout=90) as client:
        with client.stream(
            "POST",
            API_URL,
            json={
                "messages": [{"role": "user", "content": question}],
                "model": MODEL,
            },
        ) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        d = json.loads(line[6:])
                        chunk = (
                            d.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if chunk:
                            text += chunk
                    except Exception:
                        pass
    return text


def _detect_tool(response_text: str) -> str:
    """
    Infer the tool used from the model's response text.

    Priority:
    1. RAG miss marker — model was directed to rag_search but found nothing.
    2. Source: calculator
    3. Source: http(s)://...  → web_search
    4. Source: [filename.ext] → rag
    5. No Source line         → none
    """
    text = response_text.strip()

    # RAG was called but returned no results
    if "not found in your documents" in text.lower():
        return "rag"

    match = re.search(r"(?i)source:\s*(.+?)\.?\s*$", text, re.MULTILINE)
    if not match:
        return "none"

    source = match.group(1).strip().rstrip(".")

    if source.lower() == "calculator":
        return "calculator"

    if (
        source.startswith("http://")
        or source.startswith("https://")
        or "[http" in source
    ):
        return "web_search"

    # Looks like a filename (has file extension or is wrapped in brackets)
    if re.search(r"\.(pdf|docx?|txt|csv|xlsx?)(\]|$)", source, re.IGNORECASE):
        return "rag"
    if source.startswith("[") and source.endswith("]") and "http" not in source:
        return "rag"

    return "none"

# ── LangSmith target & evaluator ─────────────────────────────────────────────

def target(inputs: dict) -> dict:
    """Send a question to the API and return the response + detected tool."""
    question = inputs["question"]
    response_text = _call_api(question)
    tool_used = _detect_tool(response_text)
    return {
        "question": question,
        "response_preview": response_text[:300],
        "tool_used": tool_used,
    }


def tool_selection_evaluator(outputs: dict, reference_outputs: dict) -> dict:
    """
    Compare the detected tool against the expected tool.
    Prints question + result in a single line to avoid interleaving with other threads.
    """
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
    """
    Send a throwaway request before eval starts.
    Retries up to `retries` times with `delay` seconds between attempts.
    Raises RuntimeError if all attempts fail so eval never starts on a broken API.
    """
    import time

    for attempt in range(1, retries + 1):
        try:
            print(f"Warming up API (attempt {attempt}/{retries})...", end=" ", flush=True)
            _call_api("hello")
            print("done.")
            return
        except Exception as e:
            print(f"failed: {e}")
            if attempt < retries:
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)

    raise RuntimeError(f"API warm-up failed after {retries} attempts. Eval aborted.")


if __name__ == "__main__":
    from langsmith import Client

    _warmup()

    # Resolve the dataset (or a filtered subset of it).
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
    print(f"API     : {API_URL}")
    print("=" * 60)

    evaluate(
        target,
        data=data,
        evaluators=[tool_selection_evaluator],
        experiment_prefix="tool-routing",
        max_concurrency=1,  # sequential — one question at a time
    )

    # ── summary ───────────────────────────────────────────────────────────
    total   = len(_results_log)
    correct = sum(1 for r in _results_log if r["correct"])

    print("\n" + "=" * 60)
    print(f"Overall: {correct}/{total} correct  ({100 * correct // total if total else 0}%)")
    print()
    print(f"  {'Tool':<12}  {'Correct'}")
    print(f"  {'-'*12}  {'-'*10}")
    for tool in ["calculator", "web_search", "rag", "none"]:
        subset = [r for r in _results_log if r["expected"] == tool]
        if subset:
            ok = sum(1 for r in subset if r["correct"])
            bar = "#" * ok + "." * (len(subset) - ok)
            print(f"  {tool:<12}  {ok}/{len(subset)}  {bar}")

    print()
    print("View experiment: https://smith.langchain.com > Projects > groq-stream-fastapi")
