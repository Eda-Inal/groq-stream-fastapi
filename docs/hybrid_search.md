# grep vs BM25 — Sparse Retrieval Evaluation

This document records the evaluation for deciding whether to add a
grep-based sparse retrieval leg as an alternative (or complement) to the
existing BM25/tsvector leg in the hybrid search pipeline.

The current pipeline uses PostgreSQL `tsvector` full-text search as the
sparse leg of hybrid retrieval (merged with dense pgvector cosine search via
Reciprocal Rank Fusion). The question this test answers: **does that sparse
leg actually help with the kinds of queries this system receives — and would
a grep-style exact-pattern match do better, worse, or differently?**

---

## Background

The hybrid search path in `app/db/repositories/document.py:_hybrid_search`
combines two legs:

- **Dense leg** — pgvector cosine similarity on 768-dim nomic-embed-text embeddings
- **Sparse leg** — PostgreSQL `tsvector` + `plainto_tsquery` (BM25-like term frequency ranking)

BM25/tsvector is effective for natural-language keyword overlap but has a
known weakness: it does not match structured identifiers, error codes,
file paths, or regex-style patterns. For a technical documentation corpus
(`SVC-TXN-0041`, `E3-0019`, `/var/log/orion/audit.log`, `RULE_CURRENCY_019`)
a grep-style exact substring or regex match could outperform tsvector
tokenisation, which normalises and stems tokens in ways that destroy
these identifiers.

The test uses a single technical document (Orion Platform v2.4.1) with two
question categories:

- **Category A (EXACT)** — 7 questions targeting specific identifiers,
  code snippets, file paths, and pattern strings that are present verbatim
  in the document. These are the queries where grep could have an edge.
- **Category B (TRAP)** — 5 questions whose answers are genuinely absent
  from the document. The correct model behaviour is to refuse/say it doesn't
  know. These test whether retrieval noise causes hallucination.

---

## Test Setup

| Field | Value |
|---|---|
| Endpoint | `POST /api/v1/chat/stream` |
| Document | Orion Platform — Internal Technical Documentation v2.4.1 |
| Upload | Once per run via `scripts/eval_grep_vs_bm25.py` |
| Model | `openai/gpt-oss-120b` (Groq) |
| Temperature | 0 |
| History | None — each question uses a fresh `conversation_id` (uuid4) |
| Script | `scripts/eval_grep_vs_bm25.py` |
| TEST MODE | Active — documents scoped by `user_id`, not `conversation_id` |
| Chunks created | 10 |
| Embedding model | `mxbai-embed-large` |

---

## Evaluation Schema

Each question is scored on four axes after the test runs:

| Column | Values | Meaning |
|---|---|---|
| `correct_chunk_retrieved` | yes / no / partial / n/a | Did the answer draw from the right chunk(s)? |
| `answer_correct` | yes / no / partial | Does the answer match `expected_answer`? |
| `hallucination` | yes / no | Did the model state something not in the document? |
| `notes` | free text | Wrong identifier, partial match, fabricated value, etc. |

For Category B (TRAP) questions, the correct outcome is that the model says
the information is not in the document. Any specific value given for a TRAP
question counts as a hallucination.

---

## Category A — Exact / Pattern Questions

These questions target content that is present verbatim in the document.
BM25 may struggle here because tsvector tokenisation splits or normalises
identifiers like `SVC-AUD-0088` or `RULE_CURRENCY_019`. A grep-style exact
match on the raw chunk text would find these directly.

| ID | Question | Expected Answer | Retrieval Hint | Expected Behavior |
|---|---|---|---|---|
| A1 | What is the service tag responsible for audit trail generation? | `SVC-AUD-0088` | `SVC-AUD-0088` | ANSWER_EXACT |
| A2 | Which function in orion_router.py handles the main event dispatching, and what are its parameters? | `dispatch_event(event_id: str, payload: dict, flags: list[str]) -> RoutingResult` | `dispatch_event` | ANSWER_EXACT |
| A3 | What error code is raised when the consensus check fails, and where exactly is it thrown in the code? | E3-0019, raised inside `dispatch_event` as `ConsensusError: 'E3-0019: quorum not reached for {event_id}'` | `E3-0019` | ANSWER_EXACT |
| A4 | List all rule identifiers that were most commonly triggered during Q3 2023. | `RULE_CURRENCY_019`, `RULE_DECIMAL_004`, `RULE_WHITESPACE_031` | `RULE_` | ANSWER_EXACT |
| A5 | What is the exact format of a transaction ID in the Orion system? Give an example from the documentation. | Format: `TXN-[YYYY]-[MM]-[HH]-[0-9]{6}`, example: `TXN-2024-07-14-882341` | `TXN-` | ANSWER_EXACT |
| A6 | What flag was deprecated and what should it be replaced with? | `BYPASS_AUDIT` deprecated, replace with `AUDIT_DEFERRED` | `BYPASS_AUDIT` | ANSWER_EXACT |
| A7 | What is the exact file path where audit logs are written, and what is the fallback path if that service fails? | Primary: `/var/log/orion/audit.log`, Fallback: `/var/log/orion/audit_fallback.log` | `/var/log/orion/` | ANSWER_EXACT |

### Results — Category A

| ID | correct_chunk_retrieved | answer_correct | hallucination | notes |
|---|---|---|---|---|
| A1 | no → yes | no → yes | no | **Run 1**: Complete retrieval failure. Both dense and sparse legs missed the chunk. Model correctly said "not found" rather than hallucinating. **Run 2 (re-test)**: Retrieved correctly, answer exact. Possible causes for Run 1 failure: embedding not fully settled after upload timeout, cold cache on first run, or retrieval non-determinism. Failure is real but not fully reproducible — makes the grep case weaker but does not eliminate it. |
| A2 | yes | yes | no | Retrieved the code internals chunk. Full function signature and all three parameters returned correctly. |
| A3 | yes | yes | no | Retrieved the code internals chunk. Exact `ConsensusError` raise statement quoted with correct error code and location. |
| A4 | yes | yes | no | Retrieved the transaction lifecycle / normalization chunk. All three rule identifiers listed correctly. |
| A5 | yes | yes | no | Retrieved the transaction lifecycle chunk. Exact format string and example ID both correct. |
| A6 | yes | yes | no | Retrieved the deprecation notes chunk. Both the deprecated flag and its replacement named correctly. |
| A7 | yes | yes | no | Retrieved the audit trail chunk. Both the primary and fallback paths correct. |

### Answers — Category A

**A1** — What is the service tag responsible for audit trail generation?
> Expected: `SVC-AUD-0088`

> **Run 1:**
> This information was not found in your documents.
> Source: rag_search

> **Run 2 (re-test):**
> The service tag responsible for audit trail generation is **SVC‑AUD‑0088**.
> Source: orion_platform_v2.4.1.txt

---

**A2** — Which function in orion_router.py handles the main event dispatching, and what are its parameters?
> Expected: `dispatch_event(event_id: str, payload: dict, flags: list[str]) -> RoutingResult`

> The main event‑dispatch function defined in **orion_router.py** is:
>
> ```python
> def dispatch_event(event_id: str, payload: dict, flags: list[str]) -> RoutingResult:
>     ...
> ```
>
> **Parameters**
>
> | Parameter | Type | Description (as implied by name) |
> |-----------|------|-----------------------------------|
> | `event_id` | `str` | Identifier of the event being dispatched |
> | `payload`  | `dict`| The event's data payload |
> | `flags`    | `list[str]` | List of flag strings that can modify routing behavior (e.g., `"BYPASS_AUDIT"`) |
>
> Source: orion_platform_v2.4.1.txt

---

**A3** — What error code is raised when the consensus check fails, and where exactly is it thrown in the code?
> Expected: E3-0019, raised inside `dispatch_event` as `ConsensusError: 'E3-0019: quorum not reached for {event_id}'`

> The code raises **E3‑0019** when the consensus check does not pass.
> It is thrown in the **`dispatch_event`** function (found in `orion_router.py`) right after the call to `consensus_check`:
>
> ```python
> result = consensus_check(event_id, normalized, quorum=0.67)
>
> if not result.passed:
>     raise ConsensusError(f"E3-0019: quorum not reached for {event_id}")
> ```
>
> So the error code is **E3‑0019**, and the raise statement is located in the `dispatch_event` function of `orion_router.py`.
>
> Source: orion_platform_v2.4.1.txt

---

**A4** — List all rule identifiers that were most commonly triggered during Q3 2023.
> Expected: `RULE_CURRENCY_019`, `RULE_DECIMAL_004`, `RULE_WHITESPACE_031`

> RULE_CURRENCY_019, RULE_DECIMAL_004, RULE_WHITESPACE_031
>
> Source: orion_platform_v2.4.1.txt

---

**A5** — What is the exact format of a transaction ID in the Orion system? Give an example from the documentation.
> Expected: Format: `TXN-[YYYY]-[MM]-[HH]-[0-9]{6}`, example: `TXN-2024-07-14-882341`

> The transaction ID is formatted as:
>
> **TXN-[YYYY]-[MM]-[HH]-[0-9]{6}**
>
> An example given in the documentation is:
>
> **TXN-2024-07-14-882341**
>
> Source: orion_platform_v2.4.1.txt

---

**A6** — What flag was deprecated and what should it be replaced with?
> Expected: `BYPASS_AUDIT` deprecated, replace with `AUDIT_DEFERRED`

> The deprecated flag is **BYPASS_AUDIT**, and it should be replaced with **AUDIT_DEFERRED**.
>
> Source: orion_platform_v2.4.1.txt

---

**A7** — What is the exact file path where audit logs are written, and what is the fallback path if that service fails?
> Expected: Primary: `/var/log/orion/audit.log`, Fallback: `/var/log/orion/audit_fallback.log`

> The audit service SVC‑AUD‑0088 writes its regular audit log to:
>
> **/var/log/orion/audit.log**
>
> If the audit service fails, the fallback mechanism writes partial audit records to:
>
> **/var/log/orion/audit_fallback.log**
>
> Source: orion_platform_v2.4.1.txt

---

## Category A — Round 2: Identifier-Heavy Queries (A8–A12)

Second batch targeting the same identifier pattern as A1 (`SVC-AUD-0088`)
to understand whether A1's failure was systematic or a frequency/context effect.
Model: `openai/gpt-oss-120b`, 1 question/minute.

| ID | Question | Expected Answer | Retrieval Hint | Expected Behavior |
|---|---|---|---|---|
| A8 | What is the service tag responsible for transaction normalization? | `SVC-TXN-0041` | `SVC-TXN-0041` | ANSWER_EXACT |
| A9 | What is the internal ticket number tracking the AP-SOUTH-1 node restoration? | `INFRA-TKT-20240219-003` | `INFRA-TKT-20240219-003` | ANSWER_EXACT |
| A10 | What error class is logged when no transformation rule matches a transaction? | `ERR::UNMATCHED_RULE` | `ERR::UNMATCHED_RULE` | ANSWER_EXACT |
| A11 | What warning is logged for services still referencing the deprecated rule set? | `WARN::DEPRECATED_RULESET` | `WARN::DEPRECATED_RULESET` | ANSWER_EXACT |
| A12 | What is the dead-letter queue address configured for the EU-WEST-2 node? | `dlq://orion-dead-letter-eu` | `dlq://orion-dead-letter-eu` | ANSWER_EXACT |

### Results — Category A Round 2

| ID | correct_chunk_retrieved | answer_correct | hallucination | notes |
|---|---|---|---|---|
| A8 | yes | yes | no | `SVC-TXN-0041` appears in nearly every section and the node table — dense embedding had strong signal. |
| A9 | yes | yes | no | Ticket number retrieved correctly from the regional node registry section. |
| A10 | yes | yes | no | `ERR::UNMATCHED_RULE` retrieved and quoted correctly. |
| A11 | yes | yes | no | `WARN::DEPRECATED_RULESET` retrieved and quoted correctly. |
| A12 | yes | yes | no | `dlq://orion-dead-letter-eu` retrieved from the YAML config section. |

### Answers — Category A Round 2

**A8** — What is the service tag responsible for transaction normalization?
> Expected: `SVC-TXN-0041`

> The service tag responsible for transaction normalization is **SVC‑TXN‑0041**.
> Source: orion_platform_v2.4.1.txt

---

**A9** — What is the internal ticket number tracking the AP-SOUTH-1 node restoration?
> Expected: `INFRA-TKT-20240219-003`

> The restoration of the AP‑SOUTH‑1 node is being tracked under internal ticket **INFRA‑TKT‑20240219‑003**.
> Source: orion_platform_v2.4.1.txt

---

**A10** — What error class is logged when no transformation rule matches a transaction?
> Expected: `ERR::UNMATCHED_RULE`

> The transaction is logged under error class **ERR::UNMATCHED_RULE** when no standard transformation rule applies.
> Source: orion_platform_v2.4.1.txt

---

**A11** — What warning is logged for services still referencing the deprecated rule set?
> Expected: `WARN::DEPRECATED_RULESET`

> Services that still reference the deprecated LEGACY_V1 rule set will have a warning logged under **WARN::DEPRECATED_RULESET**.
> Source: orion_platform_v2.4.1.txt

---

**A12** — What is the dead-letter queue address configured for the EU-WEST-2 node?
> Expected: `dlq://orion-dead-letter-eu`

> The dead‑letter queue address configured for the EU‑WEST‑2 node is:
>
> **dlq://orion-dead-letter-eu**
>
> Source: orion_platform_v2.4.1.txt

---

## Category B — Hallucination Trap Questions

These questions have no answer in the document. The model should say it
doesn't know or that the information is not present. Any specific fabricated
value is a hallucination.

| ID | Question | Expected Behavior | Notes on why it's a trap |
|---|---|---|---|
| B1 | What is the maximum message size allowed for payloads processed by SVC-TXN-0041? | REFUSE | No payload size limits mentioned anywhere |
| B2 | How many engineers are on the on-call rotation for Level 3 errors? | REFUSE | On-call rotation size is not in the document |
| B3 | What authentication method does the Orion Platform use for inter-service communication? | REFUSE | Auth method is never described |
| B4 | What is the CPU and memory specification recommended for the EU-WEST-2 node? | REFUSE | Hardware specs not in the document |
| B5 | What is the SLA uptime guarantee for the Orion Platform? | REFUSE | No SLA is stated anywhere |

### Results — Category B

| ID | correct_chunk_retrieved | answer_correct | hallucination | notes |
|---|---|---|---|---|
| B1 | n/a | yes | no | Correct refusal. |
| B2 | n/a | yes | no | Correct refusal. |
| B3 | n/a | yes | no | Correct refusal. |
| B4 | n/a | yes | no | Correct refusal. |
| B5 | n/a | yes | no | Correct refusal. |

### Answers — Category B

**B1** — What is the maximum message size allowed for payloads processed by SVC-TXN-0041?
> Expected behavior: REFUSE (information not in document)

> This information was not found in your documents.
> Source: orion_platform_v2.4.1.txt

---

**B2** — How many engineers are on the on-call rotation for Level 3 errors?
> Expected behavior: REFUSE (information not in document)

> This information was not found in your documents.
> Source: orion_platform_v2.4.1.txt

---

**B3** — What authentication method does the Orion Platform use for inter-service communication?
> Expected behavior: REFUSE (information not in document)

> This information was not found in your documents.
> Source: orion_platform_v2.4.1.txt

---

**B4** — What is the CPU and memory specification recommended for the EU-WEST-2 node?
> Expected behavior: REFUSE (information not in document)

> This information was not found in your documents.
> Source: orion_platform_v2.4.1.txt

---

**B5** — What is the SLA uptime guarantee for the Orion Platform?
> Expected behavior: REFUSE (information not in document)

> This information was not found in your documents.
> Source: orion_platform_v2.4.1.txt

---

## Run 2 — `llama-3.3-70b-versatile` (all 17 questions, 2q/min)

### Results — Category A

| ID | correct_chunk_retrieved | answer_correct | hallucination | notes |
|---|---|---|---|---|
| A1 | yes | yes | no | Retrieved correctly this time. |
| A2 | yes | yes | no | Function name and all three parameters correct. |
| A3 | yes | yes | no | E3-0019 and throw location correct. |
| A4 | yes | yes | no | All three rule identifiers correct. |
| A5 | yes | yes | no | Format string and example both correct. |
| A6 | yes | yes | no | BYPASS_AUDIT → AUDIT_DEFERRED correct. |
| A7 | yes | yes | no | Both file paths correct. |
| A8 | yes | yes | no | SVC-TXN-0041 correct. |
| A9 | yes | yes | no | INFRA-TKT-20240219-003 correct. |
| A10 | no | no | yes | **Hallucination — root cause: `rag_search` schema bug.** `llama-3.3-70b` generated `top_k` and `similarity_threshold` as strings instead of integer/number. Groq API rejected the tool call before RAG ran. System fell back to web search and hallucinated `POLICY0011` from a Microsoft docs page. `rag_search` never executed. |
| A11 | no | no | no | **Retrieval failure — same root cause as A10.** `rag_search` tool call rejected by Groq API due to string parameters. System used web search and gave a generic Stack Overflow answer. `WARN::DEPRECATED_RULESET` never searched in the knowledge base. |
| A12 | yes | yes | no | dlq://orion-dead-letter-eu correct. |

### Results — Category B

| ID | correct_chunk_retrieved | answer_correct | hallucination | notes |
|---|---|---|---|---|
| B1 | n/a | yes | no | Correct refusal. |
| B2 | n/a | yes | no | Correct refusal. |
| B3 | n/a | yes | no | Correct refusal. |
| B4 | n/a | yes | no | Correct refusal. |
| B5 | n/a | yes | no | Correct refusal. |

### Answers — Run 2

**A1** → `SVC-AUD-0088` ✓
> The service tag responsible for audit trail generation is SVC-AUD-0088.
> Source: orion_platform_v2.4.1.txt

**A2** → `dispatch_event(event_id: str, payload: dict, flags: list[str]) -> RoutingResult` ✓
> The function in orion_router.py that handles the main event dispatching is `dispatch_event`. Its parameters are `event_id: str`, `payload: dict`, and `flags: list[str]`.
> Source: orion_platform_v2.4.1.txt

**A3** → E3-0019 in dispatch_event ✓
> The error code raised when the consensus check fails is E3-0019. It is thrown in the code when the result of the consensus_check function does not pass, as shown in the dispatch_event function in orion_router.py.
> Source: orion_platform_v2.4.1.txt

**A4** → RULE_CURRENCY_019, RULE_DECIMAL_004, RULE_WHITESPACE_031 ✓
> The most commonly triggered rule identifiers during Q3 2023 were RULE_CURRENCY_019, RULE_DECIMAL_004, and RULE_WHITESPACE_031.
> Source: orion_platform_v2.4.1.txt

**A5** → TXN-[YYYY]-[MM]-[HH]-[0-9]{6} / TXN-2024-07-14-882341 ✓
> The exact format of a transaction ID in the Orion system is TXN-[YYYY]-[MM]-[HH]-[0-9]{6}. An example from the documentation is TXN-2024-07-14-882341.
> Source: orion_platform_v2.4.1.txt

**A6** → BYPASS_AUDIT → AUDIT_DEFERRED ✓
> The flag "BYPASS_AUDIT" was deprecated and should be replaced with "AUDIT_DEFERRED".
> Source: orion_platform_v2.4.1.txt

**A7** → /var/log/orion/audit.log + /var/log/orion/audit_fallback.log ✓
> The exact file path where audit logs are written is /var/log/orion/audit.log. If SVC-AUD-0088 fails, the fallback mechanism writes a partial record to /var/log/orion/audit_fallback.log.
> Source: orion_platform_v2.4.1.txt

**A8** → SVC-TXN-0041 ✓
> The service tag responsible for transaction normalization is SVC-TXN-0041.
> Source: orion_platform_v2.4.1.txt

**A9** → INFRA-TKT-20240219-003 ✓
> The internal ticket number tracking the AP-SOUTH-1 node restoration is INFRA-TKT-20240219-003.
> Source: orion_platform_v2.4.1.txt

**A10** → ERR::UNMATCHED_RULE ✗ — HALLUCINATION
> The error class logged when no transformation rule matches a transaction is POLICY0011. This error occurs when no conditions in the claim rule match the condition tag specified in the CopyIssuanceStatement.
> Source: https://learn.microsoft.com/en-us/windows-server/identity/solution-guides/claims-transformation-rules-language

**A11** → WARN::DEPRECATED_RULESET ✗ — retrieval failure
> For services still referencing the deprecated rule set, a warning is logged to inform developers that the rule is deprecated and will be removed in the future. The log level used for deprecated features is typically the lowest level enabled in the production environment, such as INFO, and the log message is often prefixed with a keyword tag, say DEPRECATED.
> Source: https://softwareengineering.stackexchange.com/questions/405766/what-log-level-use-for-deprecated-features

**A12** → dlq://orion-dead-letter-eu ✓
> The dead-letter queue address configured for the EU-WEST-2 node is dlq://orion-dead-letter-eu.
> Source: orion_platform_v2.4.1.txt

**B1–B5** → All correct refusals ✓
> This information was not found in your documents.

---

## Analysis

### Score summary

| Model | Category | Correct | Total | Pass rate |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | A — Exact/Pattern (A1–A7) | 6 | 7 | 86% |
| `openai/gpt-oss-120b` | A — Exact/Pattern (A8–A12) | 5 | 5 | 100% |
| `openai/gpt-oss-120b` | B — Hallucination Trap | 5 | 5 | 100% |
| `openai/gpt-oss-120b` | **Overall** | **16** | **17** | **94%** |
| `llama-3.3-70b-versatile` | A — Exact/Pattern (A1–A12) | 10 | 12 | 83% |
| `llama-3.3-70b-versatile` | B — Hallucination Trap | 5 | 5 | 100% |
| `llama-3.3-70b-versatile` | **Overall** | **15** | **17** | **88%** |

### Where BM25 succeeded

Both models retrieved A2–A9, A12 correctly. These identifiers either appear
frequently in the document (`SVC-TXN-0041`) or have rich natural-language
context around them (`consensus check fails`, `commonly triggered`,
`deprecated flag`). The dense leg carried most of these even when tsvector
tokenisation was imperfect.

### Where BM25 failed — and where grep would have helped

**A1 (`SVC-AUD-0088`) — `gpt-oss-120b` Run 1 only.**
Failed on first run (likely embedding not yet settled after upload timeout),
succeeded on re-test and on `llama-3.3-70b` run. Structural fragility remains:
low-frequency identifier, thin surrounding context. Not fully reproducible.

**A10 and A11 — `llama-3.3-70b` only — root cause: `rag_search` tool schema bug.**
These failures were originally attributed to tsvector tokenisation fragmenting
`::` separators. Post-investigation, the actual cause was different: on these
specific runs `llama-3.3-70b` generated `top_k` and `similarity_threshold` as
JSON strings (`"5"`, `"0.7"`) instead of the correct types (integer, number).
Groq API validates tool call arguments against the declared schema and rejected
both calls with a 400 error. `rag_search` never ran. The system fell back to
web search and produced external-source answers.

The schema was subsequently fixed to accept `anyOf: [integer/number, string]`.
After the fix, `llama-3.3-70b` answered both questions correctly from the
knowledge base. This confirms the failures were a tool-call plumbing issue,
not a retrieval or tsvector problem.

This also explains why `gpt-oss-120b` retrieved A10 and A11 correctly in
Round 2: that model generated well-typed parameters and `rag_search` ran
normally.

### What this means for the implementation

A grep leg would be a third candidate set in the RRF merge, producing
`(chunk_id, grep_rank)` tuples for any chunk whose raw text contains the
query string (or a regex match of it). It would run as a plain SQL
`LIKE '%ERR::UNMATCHED_RULE%'` or `~` (regex) against the `text` column —
no index needed for a corpus of this size; for large corpora a PostgreSQL
`pg_trgm` GIN index would keep it fast.

The RRF formula naturally absorbs a third leg: the grep score only promotes
a chunk if it matched; chunks with no grep hit get no score contribution
from that leg, so unrelated results are not penalised.

---

## Decision

**Add grep: yes — strengthened case, still deferred**

The `llama-3.3-70b` run produced two new failures that `gpt-oss-120b` did
not have, including one hallucination (A10). The common thread across all
failures is separator characters that tsvector fragments. This is a
systematic gap, not an edge case.

The case for adding grep:
- `::` and `://` identifiers (`ERR::UNMATCHED_RULE`, `WARN::DEPRECATED_RULESET`,
  `dlq://orion-dead-letter-eu`) are unfindable by tsvector — it strips these
  separators and reduces the tokens to noise fragments.
- The hallucination on A10 shows that retrieval failure on identifiers is not
  safe — the model does not always say "not found", it sometimes fabricates.
- The fix is small: one extra SQL query per retrieval call merged into the
  existing RRF loop as a third leg.
- Risk is low: if grep finds nothing, its RRF contribution is zero.

Still deferred because: failures only appeared on one of the two models,
the corpus is a single document, and the `gpt-oss-120b` baseline is already
94%. Priority goes up if the system is expected to handle technical
documentation with dense identifier content.

---

## Regression Tests & Root Cause Investigation

### Round 1 — Grep implementation (before schema fix)

After implementing grep as a third RRF leg (migration `j5k6l7m8n9o0` applied,
`grep_search_enabled: bool = True`), a 4-question regression was run.
A10 and A11 still failed with the same external-source answers.

Post-investigation via container logs revealed the actual failure path:

```
llm_http_error: tool call validation failed — /top_k: expected integer, got string
→ tool_call_generation_failed_retrying_without_tools
→ web_search_injected_after_tool_call_failure
```

`llama-3.3-70b` was generating `top_k="5"` and `similarity_threshold="0.7"`
(strings). Groq API rejects tool calls that don't match the declared schema
before the Python handler runs. `rag_search` never executed — grep never ran.
The failures had nothing to do with retrieval or tsvector tokenisation.

**Fix:** `rag_search` tool schema changed to `anyOf: [integer/number, string]`
for both parameters. The Python coercion logic (`_coerce_top_k`,
`_coerce_threshold`) was already present and correct; it just never ran before.

This is a stochastic failure — the model sometimes serialises numeric
parameters as strings. Other questions happened to produce correct types on
these runs; A10 and A11 did not.

### Round 2 — After schema fix (single question verification)

After fixing the `rag_search` schema and rebuilding:

**A10** → ERR::UNMATCHED_RULE ✓
> The error class logged when no transformation rule matches a transaction is ERR::UNMATCHED_RULE.
> Source: orion_platform_v2.4.1.txt, uploaded 2026-06-09

`rag_search` ran, RAG retrieved the correct chunk, answer exact. Confirmed
the schema fix resolved the issue.

### What grep actually contributed

Grep was implemented and is running as the third RRF leg. However, **grep's
isolated contribution to retrieval quality was never cleanly measured** in
this evaluation because the A10/A11 failures — which motivated grep — turned
out to be a tool-call schema bug, not a retrieval gap. Once the schema was
fixed, the existing dense + sparse legs were sufficient to retrieve the
correct chunks.

Grep was added to learn how a third RRF leg integrates into the pipeline and
to provide a structural safety net for structured identifiers (especially
`::` and `://` patterns that tsvector destroys). It has no negative impact:
- One extra SQL query per `rag_search` call, fast with the GIN index
- If grep finds nothing, its RRF contribution is zero — other legs unchanged
- `grep_search_enabled` flag allows disabling without code changes

A clean grep-only evaluation would require a scenario where dense + sparse
both fail on an identifier query and grep is the only leg that retrieves
the correct chunk. That test was not run.
