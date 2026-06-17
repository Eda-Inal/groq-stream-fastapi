# Agentic RAG — Evaluation Results

## Setup

| Field | Value |
|---|---|
| Document | Sherlock Holmes.txt — first 15% |
| document_id | 103 |
| user_id | `sherlock-rag-test` |
| Chunks indexed | 104 |
| Tokens indexed | ~27,600 |
| Embedding model | mxbai-embed-large (1024-dim, Ollama local) |
| Chunk size | 400 tokens / 80 overlap |
| Hybrid search | enabled (dense pgvector + BM25) |
| Reranker | enabled (Jina) |
| Chat model | `openai/gpt-oss-120b` (Groq) |
| Agent model | `llama-3.3-70b-versatile` (Groq) — switched from gpt-oss-120b; see Analysis §4 |
| Test scripts | `scripts/test_sherlock_chat.py` · `scripts/test_sherlock_agent.py` |

---

## Endpoints Under Test

| Endpoint | Pipeline |
|---|---|
| `POST /api/v1/chat/stream` | Routing LLM → `rag_search` tool call → Finalization LLM |
| `POST /api/v1/agent/stream` | ReAct loop (Thought / Action / Observation / Final Answer) |

The key difference: the chat pipeline uses native function-calling and a fixed routing pass. The agent pipeline uses a text-based ReAct loop with explicit Thought/Action/Observation steps. Both use the same underlying `rag_search` tool and the same indexed document.

---

## Scoring Key

| Score | Meaning |
|---|---|
| ✅ Pass | Answer is factually correct and grounded in the retrieved text |
| ⚠️ Partial | Answer is partially correct or missing one of the required pieces |
| ❌ Fail | Answer is wrong, hallucinated, or refused when it should not have |
| 🚫 Not run | Question was skipped in this test run |
| 🛡️ Correct refusal | Model correctly said the information was not in the documents (hallucination traps only) |
| 💥 False answer | Model answered a hallucination trap instead of refusing |

---

## Category 1 — Baseline Results

> Single-chunk retrieval. Expected: 1 `rag_search` call, answer grounded in one passage.

### Chat Stream (`/chat/stream`)

| ID | Question (abbreviated) | Score | Notes |
|---|---|---|---|
| Q1 | How does Holmes deduce Watson returned to medical practice? | ✅ | Iodoform, nitrate of silver, stethoscope — all three correct |
| Q2 | How does Holmes identify the note's author as German? | 🚫 | |
| Q3 | What is Watson's exact role in the Briony Lodge plan? | 🚫 | |
| Q4 | Holmes's disguise at Briony Lodge + what Watson says about the stage? | 🚫 | |
| Q5 | Who founded the Red-Headed League, nationality, eligibility condition? | 🚫 | |

### Agent (`/agent/stream`)

| ID | Question (abbreviated) | Score | RAG calls | Thoughts | Notes |
|---|---|---|---|---|---|
| Q1 | How does Holmes deduce Watson returned to medical practice? | ✅ | 1 | — | Iodoform, nitrate of silver, stethoscope — all three correct; chunk 7/8 (rerank 0.237) held the key passage; top_k=8 was required |
| Q2 | How does Holmes identify the note's author as German? | 🚫 | — | — | |
| Q3 | What is Watson's exact role in the Briony Lodge plan? | 🚫 | — | — | |
| Q4 | Holmes's disguise at Briony Lodge + what Watson says about the stage? | 🚫 | — | — | |
| Q5 | Who founded the Red-Headed League, nationality, eligibility condition? | 🚫 | — | — | |

---

## Category 2 — Multi-Retrieval Results

> Requires combining information from 2+ chunks in different sections of the text.
> Key metric for agent: does `rag_search_count` reach ≥ 2?

### Chat Stream (`/chat/stream`)

| ID | Sections spanned | Score | Notes |
|---|---|---|---|
| Q6 | Watson stethoscope ↔ Wilson tattoo + coin (~31 chunks apart) | ❌ | 1 RAG call (not 2); model marked checklist [x] complete without finding either fact; hallucinated "medical bag" (Watson) and "brass Chinese seal" (Wilson) from training knowledge |
| Q7 | King's mask purpose ↔ Holmes's clergyman disguise (~11 chunks) | 🚫 | |
| Q8 | Holmes "cocaine & ambition" ↔ "three pipe problem" (~45 chunks) | ❌ | Neither passage retrieved; model admitted first was missing but hallucinated "So I am. Very much so." for the engagement phrase (correct: "three pipe problem") |
| Q9 | Observation lecture ↔ Spaulding's pierced ears (~41 chunks) | 🚫 | |
| Q10 | "Capital mistake to theorise" ↔ "bizarre = less mysterious" (~40 chunks) | 🚫 | |
| Q11 | Count Von Kramm alias ↔ William Morris alias (~37 chunks) | 🚫 | |
| Q12 | "I know where it is" ↔ "graver issues" to Wilson (~11 chunks) | 🚫 | |
| Q13 | Seven-pound correction ↔ King's unmasking (~7 chunks) | 🚫 | |
| Q14 | Watson reads the note ↔ Watson reads Wilson (~24 chunks) | 🚫 | |
| Q15 | Watson "ridiculously simple" ↔ "Omne ignotum" (~33 chunks) | 🚫 | |

### Agent (`/agent/stream`)

| ID | Sections spanned | Score | RAG calls | Both chunks retrieved? | Notes |
|---|---|---|---|---|---|
| Q6 | Watson stethoscope ↔ Wilson tattoo + coin | ❌ | 1 | No — neither chunk retrieved | Watson-focused query; Wilson "China" mention found but not the HOW (fish tattoo + coin chunk never retrieved); model self-certified [x] both items without specific objects; did not hallucinate but gave empty/vague answer |
| Q7 | King's mask ↔ Holmes clergyman disguise | 🚫 | — | — | |
| Q8 | "cocaine & ambition" ↔ "three pipe problem" | 🚫 | — | — | |
| Q9 | Observation lecture ↔ Spaulding's ears | 🚫 | — | — | |
| Q10 | "Capital mistake" ↔ "bizarre = less mysterious" | 🚫 | — | — | |
| Q11 | Count Von Kramm ↔ William Morris | 🚫 | — | — | |
| Q12 | "I know where it is" ↔ "graver issues" | 🚫 | — | — | |
| Q13 | Weight correction ↔ King's unmasking | 🚫 | — | — | |
| Q14 | Watson reads note ↔ Watson reads Wilson | 🚫 | — | — | |
| Q15 | "ridiculously simple" ↔ "Omne ignotum" | 🚫 | — | — | |

---

## Category 3 — Hallucination Trap Results

> Correct behavior: model says the information is not in the documents.
> Failure: model produces an answer using training knowledge.

### Chat Stream (`/chat/stream`)

| ID | Trap type | Score | Model response (summary) | Notes |
|---|---|---|---|---|
| H1 | Resolution not in upload (Adler keeps photo) | 🚫 | | |
| H2 | Physical description absent from text | 🚫 | | |
| H3 | Story not in upload (Speckled Band) | 🚫 | | |

### Agent (`/agent/stream`)

| ID | Trap type | Score | RAG calls | Model response (summary) | Notes |
|---|---|---|---|---|---|
| H1 | Resolution not in upload | 🚫 | — | | |
| H2 | Physical description absent | 🚫 | — | | |
| H3 | Story not in upload | 🚫 | — | | |

---

## Summary

### Chat Stream

| Category | Tested | Pass | Partial | Fail |
|---|---|---|---|---|
| Baseline | 1 | 1 | 0 | 0 |
| Multi-Retrieval | 1 | 0 | 0 | 1 |
| Hallucination | — | — | — | — |
| **Total** | 2 | 1 | 0 | 1 |

### Agent

| Category | Tested | Pass | Partial | Fail | Avg RAG calls |
|---|---|---|---|---|---|
| Baseline | 1 | 1 | 0 | 0 | 1.0 |
| Multi-Retrieval | 1 | 0 | 0 | 1 | 1.0 |
| Hallucination | — | — | — | — | — |
| **Total** | 2 | 1 | 0 | 1 | 1.0 |

---

## Observations

### What worked

- **Q1 (chat baseline):** Chat pipeline retrieved the correct chunk on the first try and produced a fully accurate three-part answer (iodoform, nitrate of silver, stethoscope).

### What failed

- **Q1 (agent baseline, llama):** ✅ Pass after switching model and fixing top_k. The iodoform/stethoscope chunk ranked 7th of 8 (rerank-score 0.237). With `gpt-oss-120b` and `top_k=5` this chunk was never seen. With `llama-3.3-70b-versatile` and `top_k=8` (system default, model sent no override) it appeared at the margin — and the model correctly extracted all three physical clues.

- **Q8 (chat multi-retrieval):** Single `rag_search` call could not bridge the ~45-chunk gap between the "cocaine and ambition" passage (Scandal in Bohemia opening) and the "three pipe problem" phrase (Red-Headed League closing). Neither target passage appeared in the 8 returned chunks. The model correctly admitted the first piece was absent but hallucinated the engagement phrase ("So I am. Very much so." instead of "It is quite a three pipe problem...").

### Chat vs Agent differences

**Key finding — Q6 (two different failure modes):**

*Chat pipeline:* The model independently recognized its first retrieval was incomplete and attempted a second `rag_search` (`"Watson arc-and-compass breastpin doctor Holmes deduces Watson profession"`). The pipeline blocked it (finalization runs with `tools=None`; Groq returned HTTP 400). The model's intent was correct — the limitation was architectural.

*Agent pipeline:* Made only 1 `rag_search` call. The retrieved chunks contained neither the Watson/stethoscope passage nor the Wilson/tattoo+coin passage. Despite this, the model marked both checklist items `[x]` as "found" and generated answers entirely from training knowledge — hallucinating "medical bag" for Watson (correct: stethoscope bulge in top hat) and "brass Chinese seal" for Wilson (correct: fish tattoo above right wrist). It even cited the wrong story ("A Study in Scarlet" instead of "A Scandal in Bohemia" / "The Red-Headed League").

**The contrast:** In the chat pipeline the model *knew* it needed more and tried to act on it. In the agent pipeline the checklist prompt caused it to *claim* completion rather than admit failure and retry — a prompt-following failure, not an architectural one. The `rag_search` returned plausible-looking chunks (Watson mentioned as "Dr. Watson", practice mentioned), and the model pattern-matched "close enough" instead of reformulating.

### Multi-retrieval behavior

> Q8 confirms a second failure mode: a single `rag_search` call with a long, multi-part question retrieves topically related chunks but misses the two specific passages ~45 chunks apart. The model correctly admitted one piece was absent but hallucinated the other ("So I am. Very much so." instead of "three pipe problem").
>
> Q6 shows the model *can* self-diagnose the gap — it just cannot act on that diagnosis within the chat pipeline. The agent's iterative loop removes this constraint.

---

## Full Analysis — What This Test Run Revealed

### 1. Retrieval is query-sensitive, not just question-sensitive

Q1 is the clearest demonstration. The chat pipeline used the raw question text and retrieved the correct chunk — the one containing:

> *"if a gentleman walks into my rooms smelling of iodoform, with a black mark of nitrate of silver upon his right forefinger, and a bulge on the right side of his top-hat to show where he has secreted his stethoscope..."*

The agent's model tried to issue a native tool call (intercepted from `failed_generation`) with a slightly rephrased query:

```
"Holmes deduce Watson has recently returned to medical practice physical clues"
```

The top-5 chunks returned for that query had scores in this range:

| Score | Content preview |
|---|---|
| 0.316 | Baker Street arrival scene — no physical clues |
| 0.297 | *"And in practice again… very wet lately… careless servant girl"* |
| 0.277 | Brougham carriage scene — irrelevant |
| 0.248 | Three pipe problem — irrelevant |
| 0.237 | *"Wedlock suits you… seven and a half pounds… And in practice again"* |

The iodoform/stethoscope chunk was not in the top 5 — its score fell just below the cutoff (~0.20–0.22 estimated).

The model read chunks 2 and 5 and saw "And in practice again", "very wet lately", "careless servant girl", "seven and a half pounds" — all of which appear in the same scene. **But these are different Holmes deductions from different observations in the same conversation.** The wet walk and the servant girl are explained separately from the medical practice deduction. The model treated them as the medical practice evidence because "practice" appeared nearby, and confidently listed them as the answer.

**Implication:** All chunk scores were within a 0.08-point band (0.237–0.316). In this range a small query rephrasing shifts which chunk edges into top-5. The reranker could not help because the iodoform chunk never made it to the reranking stage — it was already below the similarity threshold cutoff. Query phrasing is a first-class variable: the system got Q1 right with the raw question and failed it with a semantically near-identical reformulation.

### 2. The chat pipeline has a hard single-retrieval ceiling

Both Q6 and Q8 hit this ceiling. The finalization LLM runs with `tools=None`, so it cannot issue a follow-up search even when it recognizes the first result was incomplete. Q6 is especially instructive: the model produced a second `rag_search` call (correct query, correct intent) that Groq rejected with HTTP 400. The pipeline's architecture — one routing round, one tool call, one finalization — is the binding constraint for multi-part questions.

### 3. The agent's ReAct loop did not help for multi-retrieval

Despite the iterative design, the agent made only 1 `rag_search` call for Q6 — same as the chat pipeline. The checklist prompt was supposed to enforce "don't write Final Answer until all items are [x]", but the model self-certified [x] on both items after a single retrieval that contained neither answer, then hallucinated from training knowledge. The prompt enforces the *format* of self-checking, not the *accuracy* of it.

### 4. openai/gpt-oss-120b does not reliably follow text-based ReAct

This model (GPT-4o on Groq) is trained for native function-calling. Across 5 runs of Q1 it exhibited four distinct failure modes: native tool call → Groq 400 (recovered via `failed_generation`), inline `Action:` without a preceding newline (parser missed it), hallucinated observation loop, and TPM/timeout failures. Only one of the five runs completed the ReAct loop — and that run returned the wrong answer due to `top_k=5`.

**Switched to `llama-3.3-70b-versatile`.** This model is instruction-tuned for text generation tasks and follows the Thought/Action/Action Input/Final Answer format consistently. On the first attempt it produced a clean 2-thought ReAct trace, retrieved 8 chunks (top_k=8 system default, model sent no top_k override), and the iodoform/stethoscope chunk appeared at position 7/8 — delivering a correct answer. All subsequent agent results in this document use `llama-3.3-70b-versatile`.

### 5. Infrastructure issues during agent testing

Three separate blockers were hit before stable agent runs:
- `agent_rag_timeout = 10s` — too short for mxbai-embed-large (~4-8s per query). Fixed: `AGENT_RAG_TIMEOUT=60.0`.
- TPM limit (8000/min for this model on Groq on-demand) — the agent system prompt is ~1500 tokens, leaving little room per iteration.
- Native tool-call bypass requiring code changes to `react_agent_service.py` to recover from `failed_generation`.

### 6. Hallucination pattern: confident confabulation over honest refusal

In every failure case the model produced a specific, plausible-sounding answer rather than saying "I couldn't find this." Q6 (agent) cited a non-existent "brass Chinese seal" and the wrong story title. Q8 (chat) produced a real-sounding but wrong Holmes quote. Q1 (agent) reported adjacent deductions from the correct scene as if they were the target answer. The model never said "I didn't find it" — it always filled the gap from training knowledge.
