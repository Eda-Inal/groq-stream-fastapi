"""
Whitmore estate ReAct agent test.

Sends questions to /api/v1/agent/stream and parses the custom SSE
event stream (thought / action / observation / chunk / done).

The Whitmore document must already be in the DB (user_id=test_user).
Agent endpoint only has rag_search available — no web_search, no calculator.

Usage:
  .venv\\Scripts\\python scripts\\test_whitmore_agent.py
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
AGENT_URL = f"{BASE_URL}/agent/stream"
MODEL     = "openai/gpt-oss-120b"
USER_ID   = "test_user"
DELAY_S   = 5.0

RESPONSES_DIR = Path(__file__).parent / "responses"

# ── questions ─────────────────────────────────────────────────────────────────
#.venv\Scripts\python scripts\test_whitmore_agent.py
QUESTIONS: list[str] = [
    "What independent verification steps were skipped before Harold put pen to paper on September 3, and why does their absence matter legally?",
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


def _ask(question: str, conv_id: str) -> dict:
    """
    Stream one question through the agent endpoint.

    Returns a dict with:
      answer      - final answer text (from "chunk" events)
      thoughts    - list of thought strings
      actions     - list of {tool, args} dicts
      observations- list of {tool, result} dicts
      error       - error message string or None
    """
    payload = {
        "messages": [{"role": "user", "content": question}],
        "model": MODEL,
        "user_id": USER_ID,
        "conversation_id": conv_id,
        "temperature": 0,
    }

    answer_parts: list[str] = []
    thoughts: list[str] = []
    actions: list[dict] = []
    observations: list[dict] = []
    error: str | None = None

    with httpx.stream("POST", AGENT_URL, json=payload, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "thought":
                text = event.get("text", "")
                thoughts.append(text)
                print(f"  [Thought] {text}")

            elif etype == "action":
                tool = event.get("tool", "")
                args = event.get("args", {})
                actions.append({"tool": tool, "args": args})
                print(f"  [Action]  {tool}({json.dumps(args, ensure_ascii=False)[:120]})")

            elif etype == "observation":
                tool = event.get("tool", "")
                result = event.get("result", "")
                observations.append({"tool": tool, "result": result})
                preview = result[:200].replace("\n", " ")
                print(f"  [Obs]     {tool} → {preview}{'...' if len(result) > 200 else ''}")

            elif etype == "chunk":
                answer_parts.append(event.get("text", ""))

            elif etype == "error":
                error = event.get("message", "Unknown error")
                print(f"  [ERROR]   {error}")

            elif etype == "done":
                break

    return {
        "answer": "".join(answer_parts).strip(),
        "thoughts": thoughts,
        "actions": actions,
        "observations": observations,
        "error": error,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    questions = [q for q in QUESTIONS if q.strip()]
    if not questions:
        print("ERROR: QUESTIONS list is empty.")
        sys.exit(1)

    _health_check()

    slug = MODEL.replace("/", "_").replace(":", "_")
    run_id = uuid4().hex[:8]
    out_path = RESPONSES_DIR / f"whitmore_agent_{slug}_{run_id}.json"

    print("=" * 70)
    print(f"  Whitmore ReAct Agent Test")
    print(f"  Model     : {MODEL}")
    print(f"  User ID   : {USER_ID}")
    print(f"  Endpoint  : {AGENT_URL}")
    print(f"  Questions : {len(questions)}")
    print(f"  Output    : {out_path.name}")
    print("=" * 70)
    print()

    results: list[dict] = []

    for i, question in enumerate(questions, start=1):
        conv_id = f"whitmore-agent-{uuid4().hex[:8]}"
        print(f"{'─' * 70}")
        print(f"Q{i:02d} | conv_id={conv_id}")
        print(f"Q    : {question}")
        print()

        try:
            result = _ask(question, conv_id)
        except Exception as e:
            result = {
                "answer": f"ERROR: {e}",
                "thoughts": [],
                "actions": [],
                "observations": [],
                "error": str(e),
            }

        print()
        print(f"A    : {result['answer']}")
        print(f"       ({len(result['thoughts'])} thought(s), "
              f"{len(result['actions'])} action(s), "
              f"{len(result['observations'])} observation(s))")
        print()

        results.append({
            "index": i,
            "conv_id": conv_id,
            "question": question,
            **result,
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
                "endpoint": AGENT_URL,
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
