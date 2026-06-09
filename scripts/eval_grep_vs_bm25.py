"""
eval_grep_vs_bm25.py
────────────────────
End-to-end RAG evaluation script for comparing BM25 hybrid search with
grep-based sparse search as a third RRF leg.

The document is uploaded once; each question is sent with an independent
conversation_id (no history, chat context stays clean).

Requires TEST MODE active (chat_service.py):
  - conversation_id filter disabled → documents scoped by user_id only
  - Each question gets its own uuid4() conv_id

Outputs (scripts/responses/):
  eval_grep_vs_bm25_state.json            — document_id saved after upload
  eval_grep_vs_bm25_results_<model>.json  — raw answers
  eval_grep_vs_bm25_<model>.md            — evaluation table ready to fill in

Usage:
  # First run — uploads the document:
  .venv\\Scripts\\python scripts\\eval_grep_vs_bm25.py

  # Subsequent runs — reuse existing document:
  .venv\\Scripts\\python scripts\\eval_grep_vs_bm25.py --skip-upload
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import uuid
from pathlib import Path

import httpx

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── config ────────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000/api/v1"
CHAT_URL = f"{BASE_URL}/chat/stream"
DOCS_URL = f"{BASE_URL}/documents"

MODEL    = "llama-3.3-70b-versatile"
USER_ID  = "eval-grep-vs-bm25"

PAUSE_BETWEEN_QUESTIONS = 30.0  # 2 soru/dakika — 4 soru = ~2 dakika

RESPONSES_DIR = Path(__file__).parent / "responses"
STATE_FILE    = RESPONSES_DIR / "eval_grep_vs_bm25_state.json"

# ── belge ─────────────────────────────────────────────────────────────────────

DOCUMENT_FILENAME = "orion_platform_v2.4.1.txt"

DOCUMENT_TEXT = """
ORION PLATFORM — INTERNAL TECHNICAL DOCUMENTATION v2.4.1

Overview
The Orion Platform is a distributed event processing system designed for high-throughput financial transaction monitoring. It was first deployed in production on 2019-03-14 and has undergone three major architectural revisions since then. The system currently handles approximately 4.2 million events per day across 17 regional nodes.
At its core, Orion relies on a pipeline of microservices that communicate via an internal message broker. Each service is identified by a unique alphanumeric tag following the format SVC-[A-Z]{3}-[0-9]{4}, for example SVC-TXN-0041 handles transaction normalization, while SVC-AUD-0088 is responsible for audit trail generation. These identifiers are immutable once assigned and serve as the primary key in all internal routing tables.

Transaction Lifecycle
When a transaction enters the system, it is first assigned a globally unique identifier using the format TXN-[YYYY]-[MM]-[HH]-[0-9]{6}, such as TXN-2024-07-14-882341. This ID is stamped at ingestion time and propagates through every downstream service unchanged. If a transaction is flagged for manual review, its status code changes from STATUS:PENDING to STATUS:FLAGGED_MR and a secondary identifier REV-[0-9]{8} is attached, for example REV-00293847.
The normalization service SVC-TXN-0041 applies a series of transformation rules before forwarding the payload. Rule identifiers follow the pattern RULE_[A-Z]+_[0-9]{3}, and the most commonly triggered ones during Q3 2023 were RULE_CURRENCY_019, RULE_DECIMAL_004, and RULE_WHITESPACE_031. If none of the standard rules apply, the transaction is passed to the fallback handler, which logs the event under error class ERR::UNMATCHED_RULE before routing to a dead-letter queue.

Error Taxonomy
Orion uses a structured error taxonomy with three severity levels. Level 1 errors (format: E1-[0-9]{4}) are recoverable and handled automatically. Level 2 errors (E2-[0-9]{4}) trigger an alert to the on-call engineer but do not halt processing. Level 3 errors (E3-[0-9]{4}) cause a full pipeline pause and require manual intervention.
The most frequently observed errors in the 2023 fiscal year were E1-0023 (malformed timestamp), E1-0077 (duplicate transaction hash), E2-0145 (downstream service timeout), E2-0302 (schema version mismatch), and E3-0019 (consensus failure across regional nodes). Error E3-0019 was responsible for the major outage on 2023-11-08 that lasted 4 hours and 17 minutes, affecting nodes in the EU-WEST-2 and AP-SOUTH-1 regions.

Code Internals
The core routing logic lives in a module called orion_router.py. The main dispatch function is defined as follows:

def dispatch_event(event_id: str, payload: dict, flags: list[str]) -> RoutingResult:
    if "BYPASS_AUDIT" in flags:
        return route_direct(event_id, payload)

    normalized = normalize_payload(payload, rule_set="DEFAULT_V3")

    if normalized.get("status") == "STATUS:FLAGGED_MR":
        audit_log(event_id, level="E2", code="E2-0302")
        return route_to_review_queue(event_id, normalized)

    result = consensus_check(event_id, normalized, quorum=0.67)

    if not result.passed:
        raise ConsensusError(f"E3-0019: quorum not reached for {event_id}")

    return route_standard(event_id, normalized)

The consensus_check function requires a quorum of 0.67 (i.e., at least 67% of active regional nodes must agree) before a transaction is routed to standard processing. This threshold was raised from 0.51 following the 2023-11-08 incident. The previous value of 0.51 was deemed insufficient for Byzantine fault tolerance under high network partition scenarios.
The normalize_payload function references a rule set identifier. Currently three rule sets are in use: DEFAULT_V3, LEGACY_V1, and STRICT_MODE_V2. The LEGACY_V1 rule set is deprecated and scheduled for removal in the Q2 2025 release cycle. Any service still referencing LEGACY_V1 will receive a warning logged under WARN::DEPRECATED_RULESET starting from version 2.3.0.

Configuration
Each regional node is configured via a YAML file located at /etc/orion/node_config.yaml. A representative excerpt:

node_id: EU-WEST-2
max_throughput: 85000
consensus_quorum: 0.67
audit_service: SVC-AUD-0088
fallback_queue: dlq://orion-dead-letter-eu
retry_policy:
  max_attempts: 3
  backoff_ms: 250
  jitter: true
dead_letter_threshold: 50

The dead_letter_threshold parameter defines the maximum number of messages that can accumulate in the dead-letter queue before E2-0145 is automatically triggered. In the EU-WEST-2 node this was set to 50 until 2024-01-15, when it was increased to 120 following load testing that revealed false positive threshold breaches during peak hours.

Audit Trail
All audit events are written by SVC-AUD-0088 to an append-only log stored at /var/log/orion/audit.log. Each entry follows a strict format:
[TIMESTAMP_ISO8601] | [EVENT_ID] | [SVC_TAG] | [ACTION] | [OUTCOME]

For example:
2024-03-22T14:55:03Z | TXN-2024-03-14-004412 | SVC-TXN-0041 | NORMALIZE | OK
2024-03-22T14:55:04Z | TXN-2024-03-14-004412 | SVC-AUD-0088 | AUDIT_WRITE | OK
2024-03-22T14:55:07Z | TXN-2024-03-14-004412 | SVC-TXN-0041 | CONSENSUS | FAIL:E3-0019

If SVC-AUD-0088 itself fails, the fallback mechanism writes a partial record tagged with AUDIT_PARTIAL to a secondary log at /var/log/orion/audit_fallback.log. These partial records are reconciled during the nightly maintenance window that runs at 02:00 UTC every day.

Regional Node Registry
Node ID | Region | Status | Max Throughput | Primary Service
EU-WEST-2 | Europe West | Active | 85,000/s | SVC-TXN-0041
EU-CENTRAL-1 | Europe Central | Active | 72,000/s | SVC-TXN-0041
AP-SOUTH-1 | Asia Pacific South | Degraded | 40,000/s | SVC-TXN-0041
AP-EAST-3 | Asia Pacific East | Active | 91,000/s | SVC-TXN-0041
US-EAST-1 | US East | Active | 110,000/s | SVC-TXN-0041
US-WEST-4 | US West | Maintenance | 0/s | —

Node AP-SOUTH-1 has been in a degraded state since 2024-02-19 due to a hardware failure in the underlying infrastructure. Its throughput cap was halved from 80,000/s to 40,000/s as a temporary mitigation. Full restoration is tracked under internal ticket INFRA-TKT-20240219-003.

Deprecation & Migration Notes
As of version 2.4.0, the following are formally deprecated:
- Rule set LEGACY_V1 — migrate to DEFAULT_V3 before Q2 2025
- Flag BYPASS_AUDIT — this flag disables audit trail generation entirely and will be removed in v3.0.0; use AUDIT_DEFERRED instead
- Config key retry_policy.backoff_ms — replaced by retry_policy.backoff_strategy which accepts linear, exponential, or jitter_only

Services still using BYPASS_AUDIT as of 2024-06-01 will have it silently replaced with AUDIT_DEFERRED by the platform migration script migrate_flags.py. This script is idempotent and can be safely run multiple times.
"""

# ── questions ────────────────────────────────────────────────────────────────
# expected_behavior:
#   ANSWER_EXACT        → answer must contain a specific value/code/path
#   ANSWER_WITH_CONTEXT → answer synthesises multiple chunks
#   REFUSE              → model must say the information is not in the document

QUESTIONS: list[dict] = [
    {
        "question_id": "A10",
        "category": "EXACT",
        "text": "What error class is logged when no transformation rule matches a transaction?",
        "expected_answer": "ERR::UNMATCHED_RULE",
        "retrieval_hint": "ERR::UNMATCHED_RULE",
        "expected_behavior": "ANSWER_EXACT",
    },
    {
        "question_id": "A11",
        "category": "EXACT",
        "text": "What warning is logged for services still referencing the deprecated rule set?",
        "expected_answer": "WARN::DEPRECATED_RULESET",
        "retrieval_hint": "WARN::DEPRECATED_RULESET",
        "expected_behavior": "ANSWER_EXACT",
    },
    {
        "question_id": "A1",
        "category": "EXACT",
        "text": "What is the service tag responsible for audit trail generation?",
        "expected_answer": "SVC-AUD-0088",
        "retrieval_hint": "SVC-AUD-0088",
        "expected_behavior": "ANSWER_EXACT",
    },
    {
        "question_id": "A9",
        "category": "EXACT",
        "text": "What is the internal ticket number tracking the AP-SOUTH-1 node restoration?",
        "expected_answer": "INFRA-TKT-20240219-003",
        "retrieval_hint": "INFRA-TKT-20240219-003",
        "expected_behavior": "ANSWER_EXACT",
    },
]

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _health_check(retries: int = 3, delay: float = 3.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            print(f"API health check ({attempt}/{retries})...", end=" ", flush=True)
            httpx.get(f"{BASE_URL}/chat/models", timeout=10).raise_for_status()
            print("OK")
            return
        except Exception as e:
            print(f"failed: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError("API not responding. Is the Docker stack running?")


def _upload_doc() -> int:
    payload = {
        "filename": DOCUMENT_FILENAME,
        "text": DOCUMENT_TEXT.strip(),
        "document_type": "text",
        "user_id": USER_ID,
    }
    r = httpx.post(DOCS_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    print(f"  document_id : {data['document_id']}")
    print(f"  chunks      : {data['chunks_created']}")
    print(f"  tokens      : {data['tokens_processed']}")
    print(f"  model       : {data['embedding_model']}")
    return data["document_id"]


def _chat(question: str, conversation_id: str) -> str:
    payload = {
        "messages": [{"role": "user", "content": question}],
        "model": MODEL,
        "user_id": USER_ID,
        "conversation_id": conversation_id,
        "temperature": 0,
    }
    full_text = ""
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
                full_text += delta
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
    return full_text.strip()


def _save_state(doc_id: int) -> None:
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"document_id": doc_id, "filename": DOCUMENT_FILENAME}),
        encoding="utf-8",
    )


def _load_state() -> int | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())["document_id"]
    except Exception:
        return None


# ── output writers ───────────────────────────────────────────────────────────

def _write_results_json(results: list[dict]) -> Path:
    slug = MODEL.replace("/", "_").replace(":", "_")
    out = RESPONSES_DIR / f"eval_grep_vs_bm25_results_{slug}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _write_md(results: list[dict]) -> Path:
    slug = MODEL.replace("/", "_").replace(":", "_")
    out = RESPONSES_DIR / f"eval_grep_vs_bm25_{slug}.md"

    lines: list[str] = [
        f"# eval_grep_vs_bm25 — {MODEL}",
        "",
        "Scoring guide:",
        "- **correct_chunk_retrieved**: Did the answer come from the right chunk? (yes / no / partial)",
        "- **answer_correct**: Does the answer match expected_answer? (yes / no / partial)",
        "- **hallucination**: Did the model fabricate something not in the document? (yes / no)",
        "- **notes**: Anything notable (wrong chunk, wrong value, partial answer, etc.)",
        "",
        "---",
        "",
        "## Category A — Exact / Pattern Questions",
        "",
        "| ID | Question | Expected | Retrieval Hint | Behavior | correct_chunk_retrieved | answer_correct | hallucination | notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        q = next((q for q in QUESTIONS if q["question_id"] == r["question_id"]), {})
        if q.get("category") != "EXACT":
            continue
        hint = q.get("retrieval_hint") or "—"
        lines.append(
            f"| {r['question_id']} "
            f"| {q['text']} "
            f"| `{q['expected_answer']}` "
            f"| `{hint}` "
            f"| {q['expected_behavior']} "
            f"| "
            f"| "
            f"| "
            f"| |"
        )

    lines += [
        "",
        "### Answers (A)",
        "",
    ]
    for r in results:
        q = next((q for q in QUESTIONS if q["question_id"] == r["question_id"]), {})
        if q.get("category") != "EXACT":
            continue
        lines.append(f"**{r['question_id']}** — {q['text']}")
        lines.append(f"> Expected: `{q['expected_answer']}`")
        lines.append("")
        for line in r["answer"].splitlines():
            lines.append(f"> {line}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines += [
        "## Category B — Hallucination Trap",
        "",
        "| ID | Question | Behavior | correct_chunk_retrieved | answer_correct | hallucination | notes |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        q = next((q for q in QUESTIONS if q["question_id"] == r["question_id"]), {})
        if q.get("category") != "TRAP":
            continue
        lines.append(
            f"| {r['question_id']} "
            f"| {q['text']} "
            f"| {q['expected_behavior']} "
            f"| "
            f"| "
            f"| "
            f"| |"
        )

    lines += [
        "",
        "### Answers (B)",
        "",
    ]
    for r in results:
        q = next((q for q in QUESTIONS if q["question_id"] == r["question_id"]), {})
        if q.get("category") != "TRAP":
            continue
        lines.append(f"**{r['question_id']}** — {q['text']}")
        lines.append(f"> Expected behavior: {q['expected_behavior']} (information not in document)")
        lines.append("")
        for line in r["answer"].splitlines():
            lines.append(f"> {line}")
        lines.append("")
        lines.append("---")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main(skip_upload: bool = False) -> None:
    _health_check()
    print()
    print("=" * 70)
    print(f"  eval_grep_vs_bm25  |  model={MODEL}")
    print(f"  {len(QUESTIONS)} questions — each with an independent conversation_id")
    print("=" * 70)
    print()

    if skip_upload:
        doc_id = _load_state()
        if doc_id is None:
            print("ERROR: --skip-upload given but state file not found.")
            print(f"  Expected: {STATE_FILE}")
            sys.exit(1)
        print(f"Using existing document: id={doc_id} ({DOCUMENT_FILENAME})\n")
    else:
        print(f"Uploading document: {DOCUMENT_FILENAME}")
        try:
            doc_id = _upload_doc()
            _save_state(doc_id)
        except Exception as e:
            print(f"UPLOAD ERROR: {e}")
            sys.exit(1)
        print("\n2s bekleniyor (embedding settle)...\n")
        time.sleep(2)

    results: list[dict] = []

    try:
        for i, q in enumerate(QUESTIONS, 1):
            conv_id = str(uuid.uuid4())
            print(f"[{i}/{len(QUESTIONS)}] {q['question_id']} ({q['category']}) — {q['expected_behavior']}")
            print(f"  conv_id  : {conv_id}")
            print(f"  question : {q['text'][:90]}")
            print()

            try:
                answer = _chat(q["text"], conv_id)
                print("  ── ANSWER " + "─" * 57)
                for line in answer.splitlines():
                    print(f"  {line}")
                print("  " + "─" * 66)
                results.append({
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "question": q["text"],
                    "answer": answer,
                    "conv_id": conv_id,
                    "model": MODEL,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "question": q["text"],
                    "answer": f"ERROR: {e}",
                    "conv_id": conv_id,
                    "model": MODEL,
                })

            if i < len(QUESTIONS):
                print(f"\n[waiting {PAUSE_BETWEEN_QUESTIONS}s...]\n")
                time.sleep(PAUSE_BETWEEN_QUESTIONS)
            print()

    finally:
        pass

    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip upload and reuse the existing document_id from state file.",
    )
    args = parser.parse_args()
    main(skip_upload=args.skip_upload)
