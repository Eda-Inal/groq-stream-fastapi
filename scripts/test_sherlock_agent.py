"""
Sherlock Holmes RAG test — /api/v1/agent/stream endpoint (ReAct loop).

Sends all questions from sherlock_questions.md (baseline, multi-retrieval,
hallucination) and saves full results including thoughts/actions/observations
to scripts/responses/.

Rate limit: 2 questions per minute (30s enforced between each call).
Model: openai/gpt-oss-120b
user_id: sherlock-rag-test  (document_id: 103, 104 chunk upload)
Each question gets its own conversation_id so no history leaks between them.
"""

import httpx
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

API_URL = "http://localhost:8000/api/v1/agent/stream"
USER_ID = "sherlock-rag-test"
MODEL = "llama-3.3-70b-versatile"
RATE_LIMIT_S = 30  # 2 questions per minute


def parse_chunks_from_observation(observation_text: str) -> list[dict]:
    """Split a raw rag_search observation into individual chunk records."""
    blocks = [b.strip() for b in observation_text.split("\n---\n") if b.strip()]
    return [
        {"index": i, "preview": b[:120], "full_text": b}
        for i, b in enumerate(blocks)
    ]

QUESTIONS = [
    # ── Category 1: Baseline ─────────────────────────────────────────────
    {
        "id": "Q1", "category": "baseline",
        "text": "What specific physical clues does Holmes use to deduce that Watson has recently returned to medical practice?",
    },
    {
        "id": "Q2", "category": "baseline",
        "text": "How does Holmes identify the anonymous note's author as German before the visitor even arrives?",
    },
    {
        "id": "Q3", "category": "baseline",
        "text": "What is the exact role Watson is asked to play in Holmes's plan at Briony Lodge?",
    },
    {
        "id": "Q4", "category": "baseline",
        "text": "In what disguise does Holmes arrive at Briony Lodge, and what does Watson say would have been lost to the stage?",
    },
    {
        "id": "Q5", "category": "baseline",
        "text": "Who founded the League of the Red-Headed Men, what was his nationality, and what condition did he set for eligibility?",
    },

    # ── Category 2: Multi-Retrieval ──────────────────────────────────────
    {
        "id": "Q6", "category": "multi_retrieval",
        "text": (
            "Holmes deduces Watson's profession from objects on his person, and later does the same for Jabez Wilson. "
            "What specific object on Watson reveals he is a doctor, and what object on Wilson's body — combined with "
            "a second object on his watch chain — reveals he has been to China?"
        ),
    },
    {
        "id": "Q7", "category": "multi_retrieval",
        "text": (
            "The King of Bohemia arrives wearing a mask, claiming to be Count Von Kramm. Holmes disguises himself as "
            "a Nonconformist clergyman for the Briony Lodge operation. What is the stated purpose of each person's "
            "disguise, and in which case does the disguise actually succeed?"
        ),
    },
    {
        "id": "Q8", "category": "multi_retrieval",
        "text": (
            "Watson describes Holmes between cases as alternating between two contrasting states. When Jabez Wilson "
            "brings the Red-Headed League problem, Holmes reacts very differently to having a puzzle in front of him. "
            "What are the two states Watson describes Holmes cycling through in the Scandal in Bohemia opening, and "
            "what specific phrase does Holmes use in the Red-Headed League case to signal he is now fully engaged?"
        ),
    },
    {
        "id": "Q9", "category": "multi_retrieval",
        "text": (
            "Holmes lectures Watson that 'You see, but you do not observe,' using the Baker Street stairs as his example. "
            "How does this same principle apply when Holmes questions Wilson about Vincent Spaulding — what physical detail "
            "does Wilson openly admit he noticed, and what is Holmes's reaction that signals its true importance?"
        ),
    },
    {
        "id": "Q10", "category": "multi_retrieval",
        "text": (
            "Holmes states a methodological principle when examining the anonymous note before the King of Bohemia arrives, "
            "and restates a related but different principle after Jabez Wilson leaves. What does Holmes say about theorizing "
            "in the first instance, and what does he say about bizarre cases in the second?"
        ),
    },
    {
        "id": "Q11", "category": "multi_retrieval",
        "text": (
            "Two different characters in this text introduce themselves under false names. What false name and title does "
            "the King of Bohemia give when he first enters Holmes's rooms, and what false name does the man who managed "
            "the Red-Headed League office give to the building's landlord when later confronted by Wilson?"
        ),
    },
    {
        "id": "Q12", "category": "multi_retrieval",
        "text": (
            "Holmes says after the Briony Lodge operation 'I know where it is' about the photograph, but does not yet "
            "have it. When Wilson arrives with a completely different case, Holmes tells him 'graver issues hang from it "
            "than might at first sight appear.' What does Wilson consider to be the gravest consequence of the League "
            "affair, and how does Holmes's framing of it differ?"
        ),
    },
    {
        "id": "Q13", "category": "multi_retrieval",
        "text": (
            "Holmes makes two deductions that prove wrong or incomplete within this text. He deduces Watson has put on "
            "'seven and a half pounds' but Watson corrects him to seven. He also tells the King he already knew who he was. "
            "What does the first moment reveal about how Holmes handles being corrected, and what does the second reveal "
            "about the value he places on surprise?"
        ),
    },
    {
        "id": "Q14", "category": "multi_retrieval",
        "text": (
            "Watson attempts to read Jabez Wilson using Holmes's deductive method before Holmes speaks, and had done the "
            "same earlier with the anonymous note sent by the King of Bohemia's agent. What conclusion does Watson reach "
            "in each attempt, and how does the text signal whether he succeeds or fails each time?"
        ),
    },
    {
        "id": "Q15", "category": "multi_retrieval",
        "text": (
            "Holmes demonstrates his deductive method twice in quick succession when Wilson visits: first without "
            "explanation, then step by step when asked. What does Holmes say immediately after explaining everything "
            "that reveals his ambivalence about transparency, and how does Watson's earlier reaction to Holmes's "
            "explanations foreshadow this?"
        ),
    },

    # ── Category 3: Hallucination Traps ─────────────────────────────────
    {
        "id": "H1", "category": "hallucination",
        "text": "How does Holmes ultimately retrieve the photograph from Irene Adler at the end of the Scandal in Bohemia case?",
    },
    {
        "id": "H2", "category": "hallucination",
        "text": "What is the physical description of Irene Adler's appearance — her hair color, height, and facial features — as given in the text?",
    },
    {
        "id": "H3", "category": "hallucination",
        "text": "In The Adventure of the Speckled Band, what does Holmes identify as the murder weapon and how is it delivered to the victim?",
    },
]


def ask_agent_stream(question: str, conversation_id: str) -> tuple[str, list, list, list, float]:
    """
    Send one question to /agent/stream.
    Returns (final_answer, thoughts, actions, observations, duration_s).
    """
    payload = {
        "messages": [{"role": "user", "content": question}],
        "model": MODEL,
        "user_id": USER_ID,
        "conversation_id": conversation_id,
        "temperature": 0.0,
    }

    thoughts: list[str] = []
    actions: list[dict] = []
    observations: list[str] = []
    answer_parts: list[str] = []
    t0 = time.monotonic()

    with httpx.stream("POST", API_URL, json=payload, timeout=180.0) as r:
        if r.status_code >= 400:
            body = r.read().decode()
            raise RuntimeError(f"HTTP {r.status_code}: {body[:300]}")

        for line in r.iter_lines():
            line = line.strip()
            if not line or line == "data: [DONE]":
                continue
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            if etype == "thought":
                thoughts.append(event.get("text", ""))
            elif etype == "action":
                actions.append({
                    "tool": event.get("tool"),
                    "args": event.get("args", {}),
                })
            elif etype == "observation":
                observations.append(event.get("result", ""))
            elif etype == "chunk":
                text = event.get("text", "")
                if text:
                    answer_parts.append(text)
            elif etype == "error":
                raise RuntimeError(event.get("message", "Agent error"))

    return (
        "".join(answer_parts),
        thoughts,
        actions,
        observations,
        round(time.monotonic() - t0, 2),
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions", default=None,
        help="Comma-separated question IDs to run, e.g. Q1,Q6,H1. Runs all if omitted.",
    )
    args = parser.parse_args()

    selected = (
        [q.strip().upper() for q in args.questions.split(",")]
        if args.questions else None
    )
    questions = [q for q in QUESTIONS if selected is None or q["id"] in selected]

    if not questions:
        print(f"No questions matched: {args.questions}")
        return

    out_dir = Path(__file__).parent / "responses"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"sherlock_agent_{timestamp}.json"

    run = {
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now().isoformat(),
        "endpoint": API_URL,
        "model": MODEL,
        "user_id": USER_ID,
        "results": [],
    }

    total = len(questions)
    for i, q in enumerate(questions):
        conv_id = uuid.uuid4().hex
        print(f"\n[{i+1}/{total}] {q['id']} ({q['category']})")
        print(f"  Q: {q['text'][:90]}...")

        t_start = time.monotonic()
        error = None
        answer = ""
        thoughts: list[str] = []
        actions: list[dict] = []
        observations: list[str] = []
        duration = 0.0

        retrieved_chunks: list[dict] = []
        try:
            answer, thoughts, actions, observations, duration = ask_agent_stream(
                q["text"], conv_id
            )

            # Parse chunks out of each rag_search observation.
            chunk_idx = 0
            for obs_i, obs_text in enumerate(observations):
                parsed = parse_chunks_from_observation(obs_text)
                for c in parsed:
                    retrieved_chunks.append({
                        "rag_call": obs_i,
                        "index": chunk_idx,
                        "preview": c["preview"],
                        "full_text": c["full_text"],
                    })
                    chunk_idx += 1

            rag_count = sum(1 for a in actions if a.get("tool") == "rag_search")
            print(f"  Thoughts    : {len(thoughts)}")
            print(f"  RAG calls   : {rag_count}")
            print(f"  Chunks total: {len(retrieved_chunks)}")
            for c in retrieved_chunks:
                print(f"    [call {c['rag_call']} / chunk {c['index']}] {c['preview']}")
            print(f"  A: {answer[:120]}...")
            print(f"  ({duration}s)")
        except Exception as e:
            error = str(e)
            duration = round(time.monotonic() - t_start, 2)
            print(f"  ERROR: {error}")

        run["results"].append({
            "id": q["id"],
            "category": q["category"],
            "question": q["text"],
            "conversation_id": conv_id,
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "rag_search_count": sum(1 for a in actions if a.get("tool") == "rag_search"),
            "duration_s": duration,
            "error": error,
        })

        # Save after every question.
        out_file.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

        if i < total - 1:
            elapsed = time.monotonic() - t_start
            wait = max(0.0, RATE_LIMIT_S - elapsed)
            if wait > 0:
                print(f"  Waiting {wait:.0f}s (rate limit)...")
                time.sleep(wait)

    print(f"\nDone. {total} questions. Results saved to {out_file}")


if __name__ == "__main__":
    main()
