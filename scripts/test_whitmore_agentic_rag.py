"""
Whitmore estate RAG test — 10 independent questions.

Each question is sent with a unique conversation_id so there is no
shared history between questions. The Whitmore document is already
in the DB (user_id=test_user); no upload step needed.

TEST MODE must be active in chat_service.py:
  - documents are scoped by user_id only (not conversation_id)
  - so a new conv_id per question still finds the same documents

Rate: 3 questions/minute (20s delay) — safe for openai/gpt-oss-120b:free
  limits: 30 RPM | 1K RPD | 8K TPM | 200K TPD

Usage:
  .venv\\Scripts\\python scripts\\test_whitmore_rag.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── config ────────────────────────────────────────────────────────────────────

BASE_URL  = "http://localhost:8000/api/v1"
CHAT_URL  = f"{BASE_URL}/chat/stream"
MODEL     = "openai/gpt-oss-120b"
USER_ID   = "test_user"
DELAY_S   = 20.0  # 3 questions/minute

RESPONSES_DIR = Path(__file__).parent / "responses"

# ── questions ─────────────────────────────────────────────────────────────────

QUESTIONS: list[str] = [
    "If Diane Kowalski had been Harold's daughter instead of a hired caregiver, which elements of the undue influence test would be harder to prove and why?",
]

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _health_check() -> None:
    print("API health check...", end=" ", flush=True)
    try:
        httpx.get(f"{BASE_URL}/chat/models", timeout=10).raise_for_status()
        print("OK\n")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)


def _ask(question: str, conv_id: str) -> str:
    payload = {
        "messages": [{"role": "user", "content": question}],
        "model": MODEL,
        "user_id": USER_ID,
        "conversation_id": conv_id,
        "temperature": 0,
    }
    parts: list[str] = []
    with httpx.stream("POST", CHAT_URL, json=payload, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
                delta = chunk["choices"][0]["delta"].get("content", "")
                parts.append(delta)
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
    return "".join(parts).strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    questions = [q for q in QUESTIONS if q.strip()]
    if not questions:
        print("ERROR: QUESTIONS list is empty. Add questions and re-run.")
        sys.exit(1)

    _health_check()

    slug = MODEL.replace("/", "_").replace(":", "_")
    run_id = uuid4().hex[:8]
    out_path = RESPONSES_DIR / f"whitmore_rag_{slug}_{run_id}.json"

    print("=" * 70)
    print(f"  Whitmore RAG Test")
    print(f"  Model     : {MODEL}")
    print(f"  User ID   : {USER_ID}")
    print(f"  Questions : {len(questions)}")
    print(f"  Delay     : {DELAY_S}s between questions")
    print(f"  Output    : {out_path.name}")
    print("=" * 70)
    print()

    results: list[dict] = []

    for i, question in enumerate(questions, start=1):
        conv_id = f"whitmore-rag-{uuid4().hex[:8]}"
        print(f"{'─' * 70}")
        print(f"Q{i:02d} | conv_id={conv_id}")
        print(f"Q    : {question}")
        print()

        try:
            answer = _ask(question, conv_id)
        except Exception as e:
            answer = f"ERROR: {e}"

        print(f"A    : {answer}")
        print()

        results.append({
            "index": i,
            "conv_id": conv_id,
            "question": question,
            "answer": answer,
        })

        if i < len(questions):
            print(f"[waiting {DELAY_S:.0f}s...]\n")
            time.sleep(DELAY_S)

    print("=" * 70)
    print(f"Done — {len(results)}/{len(questions)} questions answered")

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model": MODEL,
                "user_id": USER_ID,
                "timestamp": datetime.utcnow().isoformat(),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
