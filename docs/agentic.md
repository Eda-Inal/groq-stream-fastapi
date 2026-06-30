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
| Agent model (baseline Q1–Q5) | `llama-3.3-70b-versatile` (Groq) — switched from gpt-oss-120b; see Analysis §4 |
| Agent model (multi-retrieval Q8–Q15) | `Meta-Llama-3.3-70B-Instruct` (SambaNova) — faster token throughput |
| Test scripts | manual eval scripts (not included in repo) |

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

### Chat Stream (`/chat/stream`) — openai/gpt-oss-120b

| ID | Question (abbreviated) | Score | Notes |
|---|---|---|---|
| Q1 | How does Holmes deduce Watson returned to medical practice? | ✅ | Iodoform, nitrate of silver, stethoscope — all three correct |
| Q2 | How does Holmes identify the note's author as German? | ✅ | Sentence construction ("This account of you we have from all quarters received") correctly identified as German verb placement; Egria/Bohemia paper detail mentioned briefly but not emphasized — primary deduction correct |
| Q3 | What is Watson's exact role in the Briony Lodge plan? | ⚠️ | "Station near window, watch Holmes, do not interfere" correct — but **smoke-rocket + "Fire!" shout completely missing**. Chunk 3 had the setup instructions but cut off before the hand-signal/rocket part; chunk 0 mentioned smoke-rocket in narration but model did not connect it to the explicit instruction |
| Q4 | Holmes's disguise at Briony Lodge + what Watson says about the stage? | ❌ | **Empty answer** — `tool_use_failed` error during routing pass; see Open Problems §2 |
| Q5 | Who founded the Red-Headed League, nationality, eligibility condition? | ✅ | Ezekiah Hopkins, American, red hair + 21+ age + sound body/mind — correct. Minor omissions: "bright, blazing, fiery red" specificity and London-only restriction not mentioned (both in chunk 3, model answered from chunks 0–1) |

### Chat Stream (`/chat/stream`) — llama-3.3-70b-versatile

| ID | Question (abbreviated) | Score | Notes |
|---|---|---|---|
| Q1 | How does Holmes deduce Watson returned to medical practice? | 🚫 | |
| Q2 | How does Holmes identify the note's author as German? | ✅ | Sentence construction + **Bohemian paper** both identified — more complete than gpt-oss which omitted the paper detail |
| Q3 | What is Watson's exact role in the Briony Lodge plan? | ⚠️ | Model honestly said "document does not explicitly state the exact role" — inferred window post + **found smoke-rocket** from narration chunk. Still missing hand-signal trigger and "Fire!" shout (chunk boundary). Better than gpt-oss which missed smoke-rocket entirely |
| Q4 | Holmes's disguise at Briony Lodge + what Watson says about the stage? | ⚠️ | **"The stage lost a fine actor"** correctly extracted from chunk 5 (rerank 0.263). But "Nonconformist clergyman" disguise not found — that passage was not in any of the 8 retrieved chunks. No hallucination: model said "document does not specify the disguise." Retrieval gap, not model error |
| Q5 | Who founded the Red-Headed League, nationality, eligibility condition? | ✅ | Full answer: Ezekiah Hopkins, American, **"bright, blazing, fiery red"** hair + **London-only** restriction + 21+ age + sound body/mind. Used chunks 0–3; more complete than gpt-oss which missed the fiery-red specificity and London restriction |

### Chat Stream — model comparison

| ID | gpt-oss-120b | llama-3.3-70b | Winner |
|---|---|---|---|
| Q2 | ✅ | ✅ | llama — included Bohemian paper detail |
| Q3 | ⚠️ | ⚠️ | llama — found smoke-rocket, gpt-oss didn't |
| Q4 | ❌ (tool_use_failed) | ⚠️ | llama — answered honestly; gpt-oss crashed |
| Q5 | ✅ | ✅ | llama — "bright blazing fiery red" + London |

**Conclusion:** `llama-3.3-70b-versatile` is more reliable on chat stream than `gpt-oss-120b` for this document. No `tool_use_failed` errors, more complete answers, and honest refusals when retrieval is incomplete (no hallucination). Q3 and Q4 partial scores are both retrieval gaps (chunk boundary / chunk not in top-8), not model errors.

### Agent (`/agent/stream`) — llama-3.3-70b-versatile

| ID | Question (abbreviated) | Score | RAG calls | Thoughts | Notes |
|---|---|---|---|---|---|
| Q1 | How does Holmes deduce Watson returned to medical practice? | ✅ | 1 | 2 | Iodoform, nitrate of silver, stethoscope — all three correct; chunk 7/8 (rerank 0.237) held the key passage; top_k=8 was required |
| Q2 | How does Holmes identify the note's author as German? | ✅ | 1 | 2 | Sentence construction + Bohemian paper + **Egria watermark + Gesellschaft/Papier abbreviations** — most detailed answer across all endpoints/models. Checklist: 2 items, both `[x]` after single search. More complete than both chat stream runs |
| Q3 | What is Watson's exact role in the Briony Lodge plan? | ⚠️ | 1 | 2 | Same chunk boundary issue as chat stream: "remain neutral, station near window, watch Holmes" correct but **smoke-rocket + hand signal + "Fire!" missing**. Chunk 7 (rerank 0.308) has instructions up to "watch me" but cuts off; chunk 2 (rerank 0.508) has smoke-rocket narration but model didn't connect. Did not attempt 2nd search — checklist self-certified `[x]` |
| Q4 | Holmes's disguise at Briony Lodge + what Watson says about the stage? | ✅ | 2 | 3 | **Agent advantage demonstrated.** 1st search ("Holmes disguise Briony Lodge") found "changed his costume" + "stage lost a fine actor" but NOT the clergyman name → `[x] 1, [ ] 2`. 2nd search ("Watson comment stage loss") retrieved the **Nonconformist clergyman chunk** (rerank 0.206). Full answer: "amiable and simple-minded Nonconformist clergyman" + "The stage lost a fine actor" + John Hare comparison. Chat stream (both models) failed this — gpt-oss crashed (tool_use_failed), llama got only half |
| Q5 | Who founded the Red-Headed League, nationality, eligibility condition? | ✅ | 1 | 2 | Ezekiah Hopkins, American, "bright, blazing, fiery red" hair + 21+ age + sound body/mind. Checklist: 3 items, all `[x]` after single search. Minor omission: London-only restriction (chunk 4, rerank 0.196 — retrieved but not used in answer) |

### Agent (`/agent/stream`) — openai/gpt-oss-120b

> All 4 questions hit `tool_use_failed` (native tool-call with `tools=None`). The `failed_generation` recovery in `react_agent_service.py` successfully converted each to a ReAct-format `Action:` block, so RAG calls executed normally. The issues below are model-level, not infrastructure.

| ID | Question (abbreviated) | Score | RAG calls | Thoughts | Notes |
|---|---|---|---|---|---|
| Q2 | How does Holmes identify the note's author as German? | ✅ | 1 | 1 | Sentence construction + Gesellschaft/Papier + Egria — detailed and correct. But **no checklist**: thought was just "I need to search the documents." No `[ ]` items, no `[x]` tracking |
| Q3 | What is Watson's exact role in the Briony Lodge plan? | ❌ | 1 | 1 | **Wrong passage selected.** Retrieved chunks included the correct instructions (window, watch, smoke-rocket narration) but model picked the wrong scene — answered "Watson is asked to call Holmes the next afternoon at three o'clock." This is from a completely different part of the story. Model comprehension failure, not retrieval failure |
| Q4 | Holmes's disguise at Briony Lodge + what Watson says about the stage? | ❌ | 1 | 1 | **Hallucination.** Model described the **King of Bohemia's** costume (astrakhan coat, vizard mask, flame-coloured silk) as Holmes's disguise. Clergyman chunk was not retrieved (same gap as other runs), but model confabulated the King's appearance as Holmes's instead of saying "not found." Got "stage lost a fine actor" correct. Thought was malformed: "Searching for item 1.Action: rag_search" — Action tag leaked into thought text |
| Q5 | Who founded the Red-Headed League, nationality, eligibility condition? | ✅ | 1 | 2 | Ezekiah Hopkins, American, red-headed + 21+ + sound body/mind. Correct but missing "bright, blazing, fiery red" and London restriction. Recovered from `native_tool_call_converted_to_react` |

### Agent — model comparison

| ID | llama-3.3-70b | gpt-oss-120b | Winner |
|---|---|---|---|
| Q2 | ✅ (checklist ✓) | ✅ (no checklist) | Tie — both detailed |
| Q3 | ⚠️ (correct partial, chunk boundary) | ❌ (wrong passage selected) | llama — at least got the right scene |
| Q4 | ✅ (2 RAG, checklist triggered 2nd search) | ❌ (hallucinated King's costume as Holmes's) | **llama — decisive win** |
| Q5 | ✅ (fiery red + London) | ✅ (missing details) | llama — more complete |

**Conclusion:** `gpt-oss-120b` is unreliable on the agent endpoint. Three structural problems: (1) never produces checklist format — thoughts are single-line "I need to search", (2) never triggers a 2nd RAG call because there's no checklist to track incomplete items, (3) prone to wrong-passage selection (Q3) and hallucination (Q4 — attributed the King's costume to Holmes). The `failed_generation` recovery code works correctly for native tool-call errors, but the model's text-based reasoning quality is too low for the ReAct format. `llama-3.3-70b-versatile` remains the recommended agent model.

---

## Category 2 — Multi-Retrieval Results

> Requires combining information from 2+ chunks in different sections of the text.
> Key metric for agent: does `rag_search_count` reach ≥ 2?

### Chat Stream (`/chat/stream`)

| ID | Sections spanned | Score | Notes |
|---|---|---|---|
| Q6 | Watson stethoscope ↔ Wilson tattoo + coin (~31 chunks apart) | ❌ | 1 RAG call (not 2); model marked checklist [x] complete without finding either fact; hallucinated "medical bag" (Watson) and "brass Chinese seal" (Wilson) from training knowledge |
| Q7 | King's mask purpose ↔ Holmes's clergyman disguise (~11 chunks) | 🚫 | |
| Q8 | Holmes "cocaine & ambition" ↔ "three pipe problem" (~45 chunks) | ❌ | Neither passage retrieved; model admitted first was missing but hallucinated "So I am. Very much so." for the engagement phrase (correct: "three pipe problem"). See Agent Q8 results below for extended analysis |
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
| Q6 (pre-fix) | Watson stethoscope ↔ Wilson tattoo + coin | ⚠️ | 2 | Wilson ✅, Watson ❌ | **Before checklist-validation prompt fix.** 1st search missed Watson stethoscope chunk — model self-certified `[x]` despite saying "not explicitly found." 2nd search targeted Wilson → fish tattoo (0.595) + Chinese coin (0.622) fully correct. Watson vague ("can be inferred... military doctor") |
| Q6 (post-fix v1) | Watson stethoscope ↔ Wilson tattoo + coin | ⚠️ | 2 | Watson ✅, Wilson ⚠️ | **After checklist-validation prompt fix (citable-fact rule only).** Model retried Watson search → **stethoscope chunk found** (rerank 0.252). But both searches spent on Watson, Wilson degraded to "right hand + seal" (wrong). Shows retry works but iteration budget is zero-sum |
| Q6 (post-fix v2) | Watson stethoscope ↔ Wilson tattoo + coin | ⚠️ | 2 | Watson ❌, Wilson ✅ | **After keyword-targeting rule added.** 1st search Watson → not found. Model correctly moved to `[ ] 2` and targeted Wilson keywords: "Jabez Wilson object on body and watch chain reveals China" → **fish tattoo (0.677) + Chinese coin (0.635) fully correct.** Watson still vague — the iodoform/stethoscope chunk doesn't mention Watson by name ("if a gentleman walks into my rooms smelling of iodoform..."), so Watson-focused queries can't surface it. This is a retrieval gap, not a prompt gap. The keyword-targeting rule correctly distributed searches across both items instead of doubling up on one |
| Q6 (post-fix v3) | Watson stethoscope ↔ Wilson tattoo + coin | ⚠️ | 2 | Watson ❌, Wilson ✅ | **After exact-phrase citation rule added (3 separate rules).** Same search distribution as v2 — Wilson fully correct with **direct quote**: *"The fish that you have tattooed immediately above your right wrist could only have been done in China... When, in addition, I see a Chinese coin hanging from your watch-chain, the matter becomes even more simple."* Watson: "not explicitly stated" — no hallucination. Citation rule working: model quoted passage for Wilson, admitted gap for Watson |
| Q6 (consolidated) | Watson stethoscope ↔ Wilson tattoo + coin | — | — | — | **3 rules consolidated into 1 line** and added to prompt. Not yet tested (rate limit exhausted). Prompt: "Only mark [x] when you can quote the exact supporting phrase from the Observation; in the Final Answer, cite that phrase for each item. When reformulating, target only the unfound item's keywords." |
| Q7 | King's mask ↔ Holmes clergyman disguise | 🚫 | — | — | |
| Q8 (SambaNova, overfetch=3) | "cocaine & ambition" ↔ "three pipe problem" | ⚠️ | 3 | cocaine ✅, three-pipe ❌ | **Model: `Meta-Llama-3.3-70B-Instruct` (SambaNova).** 1st RAG call failed (embedding error). 2nd call ("Sherlock Holmes personality traits between cases") retrieved "cocaine and ambition" chunk (rerank 0.339) — model correctly identified the two states. 3rd call ("Red-Headed League engagement phrase") returned "You could not possibly have come at a better time" chunk (rerank 0.676) — model used this as the engagement phrase. Plausible but **not the expected answer** ("It is quite a three pipe problem"). Chunk 588 containing "three pipe problem" was never retrieved. No hallucination: model answered from retrieved text only. Duration: 34.65s |
| Q8 (SambaNova, overfetch=10) | "cocaine & ambition" ↔ "three pipe problem" | ❌ | 5 | cocaine ❌, three-pipe ❌ | **Same model, `reranker_overfetch_multiplier` raised to 10 (80 candidates per search).** Model made 5 RAG calls (max iterations exhausted), none retrieved the "cocaine and ambition" chunk or the "three pipe problem" chunk. Every thought remained `[ ]` — model never self-certified any item. 1st query ("Scandal in Bohemia opening Holmes states") returned Bohemia dialogue scenes, not the opening narration. Subsequent queries reformulated but kept returning the same Red-Headed League plot chunks. **Final output was raw Thought/Action text** — agent hit max iterations without producing a Final Answer. The wider overfetch pool (80 vs 24 chunks) introduced more noise and the reranker scored topically related but non-target chunks higher, pushing the critical passages further down. Duration: 29.76s |
| Q8 (SambaNova, post-fix) | "cocaine & ambition" ↔ "three pipe problem" | ✅ | 4 | Both ✅ | **After checklist evidence-quoting rule. Previously unsolved across all runs.** 4 RAG calls: first 2 searches returned no quotable evidence → model kept both items `[ ]` and reformulated (previously would either false-certify or exhaust iterations). 3rd search found "cocaine and ambition" chunk → `[x]` with verbatim quote: *"alternating from week to week between cocaine and ambition, the drowsiness of the drug, and the fierce energy of his own keen nature."* 4th search found "three pipe problem" chunk → `[x]` with verbatim quote: *"It is quite a three pipe problem, and I beg that you won't speak to me for fifty minutes."* Final Answer uses both quotes, matches expected answer exactly. The evidence-quoting rule fixed Q8 by (1) preventing false [x] on wrong passages and (2) forcing continued reformulation until the actual target passages were retrieved. Duration: 24.27s |
| Q9 (SambaNova) | Observation lecture ↔ Spaulding's ears (~41 chunks) | ✅ | 4 | Both ✅ | **Model: `Meta-Llama-3.3-70B-Instruct` (SambaNova).** Checklist split into 4 items: (1) observe-vs-see lecture, (2) Holmes questioning Wilson, (3) Wilson's physical detail admission, (4) Holmes's reaction. Each item got its own RAG call. "You see, but you do not observe" chunk retrieved at rerank 0.456. Spaulding interrogation chunk at rerank 0.752 (strongest score in all Q8–Q11 runs). Answer: Wilson noticed "white splash of acid" + "ears pierced for earrings"; Holmes "sat up in considerable excitement" + "I thought as much" + "sinking back in deep thought." All correct. Minor gap: answer doesn't explicitly connect the observe-vs-see principle to the Spaulding scene — lists both but leaves the parallel implicit. Duration: 24.07s |
| Q10 (SambaNova) | "Capital mistake to theorise" ↔ "bizarre = less mysterious" (~40 chunks) | ✅ | 2 | Both ✅ | **Model: `Meta-Llama-3.3-70B-Instruct` (SambaNova).** Both target passages retrieved and **quoted verbatim**: *"It is a capital mistake to theorise before one has data"* and *"the more bizarre a thing is the less mysterious it proves to be."* Notably, the "bizarre" quote lives in chunk 588 — the same chunk that contains "three pipe problem" (Q8's target). The chunk was retrievable here because the query targeted "bizarre cases" directly, not an abstract paraphrase like "engagement phrase." Confirms Q8 failure was query-formulation, not embedding gap. "Omne ignotum" chunk also retrieved (rerank 0.537). Duration: 8.88s |
| Q11 (SambaNova) | Count Von Kramm ↔ William Morris (~37 chunks) | ✅ | 2 | Both ✅ | **Model: `Meta-Llama-3.3-70B-Instruct` (SambaNova).** Both false identities retrieved and correctly reported: "Count Von Kramm, a Bohemian nobleman" (King of Bohemia) and "William Morris" (Duncan Ross / Red-Headed League office manager, given to landlord). Clean 2-search checklist execution — 1st search for Von Kramm, 2nd for the League office alias. Duration: 9.65s |
| Q12 (SambaNova) | "I know where it is" ↔ "graver issues" (~11 chunks) | ✅ | 2 | Both ✅ | Clean 2-item checklist. Wilson: "lost four pound a week." Holmes: "graver issues hang from it." Both retrieved, correctly contrasted. Duration: 9.0s |
| Q13 (SambaNova) | Weight correction ↔ King's unmasking (~7 chunks) | ⚠️ | 2 | Both ✅ | **Retrieval correct, comprehension/reasoning issue.** Both chunks retrieved. Model got the direction right but did not quote the text — missed "Indeed, I should have thought a little more" and "I was also aware of that." Interpreted rather than cited. Duration: 13.17s |
| Q13 (SambaNova, post-fix) | Weight correction ↔ King's unmasking (~7 chunks) | ⚠️ | 2 | Both ✅ | **After checklist evidence-quoting rule.** Item 2 (King scene) improved: model quoted *"Your Majesty had not spoken before I was aware that I was addressing Wilhelm Gottsreich Sigismond von Ormstein"* verbatim. Item 1 (weight correction) still partial: model quoted Watson's *"Seven!"* but missed Holmes's reply *"Indeed, I should have thought a little more"* which is 2 lines below in the same chunk. Model wrote "no direct quote available" and inferred — acknowledging the rule but bypassing it. This is a model-level chunk-reading issue: the target phrase is adjacent to what the model DID extract. Score stays ⚠️ but item 2 now correct with evidence. Duration: 16.53s |
| Q14 (SambaNova) | Watson reads note ↔ Watson reads Wilson (~24 chunks) | ⚠️ | 3 | Both ⚠️ | **Retrieval correct, comprehension/reasoning issue.** Relevant chunks retrieved (including Omne ignotum at rerank 0.093) but model failed to extract Watson's specific conclusions from them. Vague summary instead of textual grounding. Duration: 13.17s |
| Q14 (SambaNova, post-fix) | Watson reads note ↔ Watson reads Wilson (~24 chunks) | ⚠️ | 3 | Both ✅ | **After checklist evidence-quoting rule.** Major improvement in checklist behavior: (1) Item 1 first search found no matching quote → model kept `[ ]` and re-searched (previously marked false `[x]`). (2) All three [x] marks now carry direct quotes: `"nothing remarkable about the man save his blazing red head"` (Wilson), `"The man who wrote it was presumably well to do"` (note), `"the thing always appears to me to be so ridiculously simple"` (signal). (3) Final Answer uses all three quotes. **Remaining gap:** item 3 uses a general Watson quote about Holmes's explanations rather than contrasting the two specific signals — Holmes confirming Watson's note deduction (partial success) vs. Holmes rattling off 5 facts Watson missed for Wilson (failure). The model did not differentiate success vs. failure across the two cases. Score stays ⚠️ but quality significantly improved. Duration: 26.36s |
| Q15 (SambaNova) | "ridiculously simple" ↔ "Omne ignotum" (~33 chunks) | ✅ | 1 | Both ✅ | Single RAG call retrieved both targets. "Omne ignotum pro magnifico" quoted verbatim. Watson's "ridiculously simple" correctly connected as foreshadowing. Duration: 8.25s |

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

### Chat Stream — openai/gpt-oss-120b

| Category | Tested | Pass | Partial | Fail |
|---|---|---|---|---|
| Baseline | 5 | 3 | 1 | 1 |
| Multi-Retrieval | 2 | 0 | 0 | 2 |
| Hallucination | — | — | — | — |
| **Total** | 7 | 3 | 1 | 3 |

### Chat Stream — llama-3.3-70b-versatile

| Category | Tested | Pass | Partial | Fail |
|---|---|---|---|---|
| Baseline | 4 | 2 | 2 | 0 |
| Multi-Retrieval | — | — | — | — |
| Hallucination | — | — | — | — |
| **Total** | 4 | 2 | 2 | 0 |

### Agent — llama-3.3-70b-versatile

| Category | Tested | Pass | Partial | Fail | Avg RAG calls |
|---|---|---|---|---|---|
| Baseline | 5 | 4 | 1 | 0 | 1.2 |
| Multi-Retrieval | 1 | 0 | 0 | 1 | 1.0 |
| Hallucination | — | — | — | — | — |
| **Total** | 6 | 4 | 1 | 1 | 1.2 |

### Agent — Meta-Llama-3.3-70B-Instruct (SambaNova)

| Category | Tested | Pass | Partial | Fail | Avg RAG calls |
|---|---|---|---|---|---|
| Baseline | — | — | — | — | — |
| Multi-Retrieval | 8 | 5 | 2 | 1 | 2.4 |
| Hallucination | — | — | — | — | — |
| **Total** | 8 | 5 | 2 | 1 | 2.4 |

> 8 unique questions (Q8–Q15). Q8 overfetch=10 counted as a parameter variant, not a separate test.
> ✅ Q9, Q10, Q11, Q12, Q15. ⚠️ Q13, Q14 (retrieval correct, comprehension issue). ❌ Q8 (query generation failure — three-pipe never retrieved; overfetch=10 variant further degraded results).

### Agent — Meta-Llama-3.3-70B-Instruct (SambaNova) — after evidence-quoting rule

| Category | Tested | Pass | Partial | Fail | Avg RAG calls |
|---|---|---|---|---|---|
| Multi-Retrieval (retested) | 3 | 1 | 2 | 0 | 3.0 |

> Retested Q8, Q13, Q14 after adding evidence-quoting rule to `_ROLE_WITH_TOOLS`.
> ✅ Q8 (previously ❌ — both passages now retrieved and quoted verbatim; 4 RAG calls). ⚠️ Q13 (item 2 improved with quote, item 1 still skims past adjacent line). ⚠️ Q14 (2/3 items correct with quotes, success/failure contrast not differentiated).
> **Key result:** Q8 went from unsolvable (❌ across 3 prior runs) to fully correct (✅) — the evidence-quoting rule prevented false [x] marks and forced query reformulation until target passages were found.

### Agent — openai/gpt-oss-120b

| Category | Tested | Pass | Partial | Fail | Avg RAG calls |
|---|---|---|---|---|---|
| Baseline | 4 | 2 | 0 | 2 | 1.0 |
| Multi-Retrieval | — | — | — | — | — |
| Hallucination | — | — | — | — | — |
| **Total** | 4 | 2 | 0 | 2 | 1.0 |

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

*Agent pipeline (gpt-oss-120b, original run):* Made only 1 `rag_search` call. The retrieved chunks contained neither the Watson/stethoscope passage nor the Wilson/tattoo+coin passage. Despite this, the model marked both checklist items `[x]` as "found" and generated answers entirely from training knowledge — hallucinating "medical bag" for Watson (correct: stethoscope bulge in top hat) and "brass Chinese seal" for Wilson (correct: fish tattoo above right wrist). It even cited the wrong story ("A Study in Scarlet" instead of "A Scandal in Bohemia" / "The Red-Headed League").

*Agent pipeline (llama-3.3-70b, after prompt fixes):* The checklist-validation prompt fix addressed this failure. Across 4 iterations of Q6:

| Iteration | Prompt change | Watson | Wilson | Key behavior |
|---|---|---|---|---|
| pre-fix | — | ❌ | ✅ | 2 RAG calls, each item got its own search |
| post-fix v1 | +citable fact | ✅ | ❌ | Model retried Watson (good), but both searches consumed by Watson |
| post-fix v2 | +keyword targeting | ❌ | ✅ | Model correctly moved to unfound Wilson item after Watson miss |
| post-fix v3 | +exact phrase cite | ❌ | ✅ + quote | Wilson answer now includes direct quote from text; Watson says "not found" |

The prompt fixes progressively improved agent behavior: v1 proved retry works, v2 fixed search targeting, v3 added grounding. The three rules were then consolidated into a single line to reduce prompt clutter.

**The remaining gap:** Watson's stethoscope chunk is unreachable via Watson-focused queries because the passage doesn't mention Watson by name — it says "if a gentleman walks into my rooms smelling of iodoform..." This is a fundamental retrieval challenge that prompt engineering cannot solve; it requires either a 3rd search attempt or a different query strategy (e.g. searching for the medical-practice deduction scene rather than Watson specifically).

**The contrast:** In the chat pipeline the model *knew* it needed more and tried to act on it. In the agent pipeline, the original checklist prompt caused it to *claim* completion rather than admit failure and retry — but this was fixable via prompt engineering. The keyword-targeting rule was the most impactful single change.

### Multi-retrieval behavior

> Q8 confirms a second failure mode: a single `rag_search` call with a long, multi-part question retrieves topically related chunks but misses the two specific passages ~45 chunks apart. The model correctly admitted one piece was absent but hallucinated the other ("So I am. Very much so." instead of "three pipe problem").
>
> Q6 shows the model *can* self-diagnose the gap — it just cannot act on that diagnosis within the chat pipeline. The agent's iterative loop removes this constraint.

### Checklist evidence-quoting rule — Q13/Q14 comprehension fix

Q13 and Q14 both had the same failure mode: correct chunks retrieved, model marked `[x]` with vague summaries ("found: he handles it with ease"), then produced a generic Final Answer with zero quotes. The root cause was that the checklist rule said "mark completed items [x]" but did not define what [x] *means* — the model treated any topical overlap as "found."

**Fix applied** (`react_agent_prompts.py`, `_ROLE_WITH_TOOLS`): added an evidence-quoting requirement to the [x] marking rule and updated both examples to demonstrate the new format.

Rule added:
> To mark an item [x], you MUST quote the exact phrase from the Observation that answers it. Format: `[x] 1. <item> — "<exact quote>"`. If you cannot find an exact quote, keep the item [ ] and search again with different keywords. Never mark [x] with a summary or paraphrase — only a direct quote.

Additional rule:
> In your Final Answer, use the quoted evidence from your checklist. Do not discard the quotes and write a generic summary.

**Q14 retest results (SambaNova, `Meta-Llama-3.3-70B-Instruct`):**

| Behavior | Before fix | After fix |
|---|---|---|
| Item 1 first search: no match | ❌ Marked false [x] | ✅ Kept [ ], re-searched |
| [x] marks have quotes | ❌ Zero quotes | ✅ All 3 items quoted |
| Final Answer uses evidence | ❌ Generic summary | ✅ 3 direct quotes |
| Overall score | ⚠️ (vague, partly wrong) | ⚠️ (correct facts, weak on contrast) |

Score stays ⚠️ because the model did not differentiate Watson's partial success (note) from his failure (Wilson) — it used a general quote about Watson's bafflement instead of contrasting Holmes's confirming response (note) vs. Holmes's 5-deduction rebuke (Wilson). But the factual extraction improved substantially: the model now quotes the correct passages instead of inventing summaries.

**Regression testing (SambaNova, `Meta-Llama-3.3-70B-Instruct`, post-fix):**

| Question | Previous | Post-fix | RAG calls | Regression? |
|---|---|---|---|---|
| Q10 | ✅ 2 RAG | ✅ 2 RAG, same verbatim quotes | 2 | None |
| Q12 | ✅ 2 RAG | ✅ 2 RAG, same verbatim quotes | 2 | None |
| Q15 | ✅ 1 RAG | ✅ 1 RAG, same verbatim quotes | 1 | None |
| Q14 | ⚠️ 3 RAG | ⚠️ 3 RAG, consistent with first post-fix run | 3 | None |

All previously-passing questions remain at the same score and RAG call count. The evidence-quoting rule adds no overhead to questions that already had correct [x] marks.

**Q8 retrieval variance:** Q8 was retested twice on SambaNova post-fix. First run: ✅ (both "cocaine and ambition" + "three pipe problem" verbatim). Second run: ⚠️ (item 1 correct, item 2 retrieved `"It is a little off the beaten track"` instead of `"three pipe problem"`). Same model, same prompt, same provider — the difference is retrieval variance. The "three pipe problem" chunk is at the margin of retrievability; whether it appears in the top-8 depends on query formulation, which varies slightly between runs even at temperature=0.

**Groq `llama-3.3-70b-versatile` cross-validation:** Q10 and Q14 were also tested on Groq's `versatile` variant. Q10: ✅ (identical behavior). Q14: ❌ (model marked `[x] — not found` — acknowledged the rule but bypassed it; only 2 RAG calls instead of 3). The `versatile` variant follows the evidence-quoting rule less reliably than `Instruct` on SambaNova.

**Context trim experiment (reverted):** A `_trim_old_observations` method was tested that replaced all but the most recent Observation with a short placeholder. While it solved Groq's 12K TPM limit (413 errors), Q14 on Groq with trim produced worse results (2 RAG calls, false [x]) than without. The trim was reverted pending isolated testing on SambaNova where TPM is not a constraint.

**Agent rate limit handling added** (`react_agent_service.py`): The agent loop previously crashed on any 429 error. Three changes:
1. **TPM (≤60s retry_after):** Wait and retry without consuming an iteration (while-loop instead of for-loop).
2. **TPD (>60s retry_after):** Return error immediately — daily limit cannot be waited out.
3. **Max 3 retries:** Prevents infinite retry loops. After 3 consecutive 429s, returns error.

---

### Q8 deep-dive — two-layer failure: query quality + chunk comprehension

**Setup:** Chunk 588 (document_id 103) contains the target passage: *"It is quite a three pipe problem, and I beg that you won't speak to me for fifty minutes."* Verified present in pgvector via `SELECT ... WHERE text ILIKE '%three pipe problem%'`. The chunk is 1,256 characters and also contains *"the more bizarre a thing is the less mysterious it proves to be"* — both key phrases for Q8 and Q10.

#### Layer 1 — Agent query quality: chunk never retrieved

| Run | Model | Overfetch | Total chunks examined | Chunk 588 retrieved? | "cocaine & ambition" found? |
|---|---|---|---|---|---|
| 142638 | Meta-Llama-3.3-70B (SambaNova) | 3 (24 candidates) | 3 RAG calls × 8 = 24 chunks | ❌ | ✅ (rerank 0.339) |
| 143632 | Meta-Llama-3.3-70B (SambaNova) | 10 (80 candidates) | 5 RAG calls × 8 = 40 chunks | ❌ | ❌ |
| 142303 | llama-4-scout-17b (Groq) | 3 (24 candidates) | 2 RAG calls × 8 = 16 chunks | ❌ | ✅ (rerank 0.339) |

Across 10 RAG calls with 5 different query formulations, the agent never generated a query that retrieved chunk 588. The queries used abstract paraphrases ("Red-Headed League engagement phrase", "Holmes fully engaged phrase", "Holmes energy levels between cases") that have no semantic overlap with the literary idiom "three pipe problem."

**Overfetch=10 made things worse:** The wider candidate pool (80 chunks) introduced more noise. The reranker scored topically related but non-target chunks higher, and even the "cocaine and ambition" chunk (found at rerank 0.339 with overfetch=3) was pushed out of the top 8.

#### Layer 2 — Chunk comprehension: model ignores retrieved passage

**Q8B probe test** (run 144827, chat stream, `Meta-Llama-3.3-70B-Instruct` via SambaNova): a simplified, single-part question was used:

> *"What specific phrase does Holmes use after Jabez Wilson leaves to signal he is now fully engaged in thinking about the Red-Headed League problem?"*

**Result:** Chunk 588 was retrieved at **index 5, rerank score 0.456**. The chunk contains the exact target phrase in full:

> *"To smoke," he answered. "It is quite a three pipe problem, and I beg that you won't speak to me for fifty minutes."*

Despite having this chunk in the context, the model responded: *"It seems that the specific phrase used by Holmes after Jabez Wilson leaves to signal he is now fully engaged in thinking about the Red-Headed League problem is not explicitly stated in the provided passages."*

This is the same "chunk in context but not read" failure observed previously — the model does not parse all 8 chunks carefully during finalization; it skims and declares "not found" even when the answer is present.

#### Revised root cause

The failure is **not** an embedding/BM25 problem. With the right query, hybrid search retrieves chunk 588 reliably (rerank 0.456, well within top 8). The two actual problems are:

1. **Agent query generation** — the agent generates abstract paraphrases instead of content-specific keywords. It never formulates a query close enough to "Holmes smoke three pipe problem Wilson leaves" for the retrieval system to surface the chunk.
2. **Finalization comprehension** — even when the chunk is retrieved (Q8B chat stream), the model fails to extract the answer from it. This is a known pattern: the model skims retrieved passages rather than reading each one completely.

#### Resolved questions

1. **Is this a SambaNova-specific comprehension issue?** No — Q13 and Q14 (also SambaNova) showed the same pattern: retrieval correct, but model failed to extract or quote details from retrieved chunks. This is a model-level comprehension weakness (Llama-3.3-70B skimming rather than reading), not a provider issue.
2. **Does the agent perform better with Q8B?** Not tested separately, but Q10 confirmed that chunk 588 is retrievable with the right query ("bizarre cases") and the model correctly quoted from it. The failure is query-formulation (Q8 agent never generates a query that surfaces chunk 588) combined with comprehension (Q8B chat stream retrieved the chunk but model said "not found").
3. **Can the finalization prompt be improved?** The FINALIZATION_SYSTEM_MESSAGE already says "Read every tool message carefully before answering" — the model ignores this instruction. Further prompt changes are unlikely to fix a comprehension gap that spans both pipeline types.

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

**Switched to `llama-3.3-70b-versatile`.** This model is instruction-tuned for text generation tasks and follows the Thought/Action/Action Input/Final Answer format consistently. On the first attempt it produced a clean 2-thought ReAct trace, retrieved 8 chunks (top_k=8 system default, model sent no top_k override), and the iodoform/stethoscope chunk appeared at position 7/8 — delivering a correct answer. Baseline results (Q1–Q5) in this document use `llama-3.3-70b-versatile` on Groq. Multi-retrieval results (Q8–Q15) use `Meta-Llama-3.3-70B-Instruct` on SambaNova — same model weights, different provider for faster token throughput.

### 5. Infrastructure issues during agent testing

Three separate blockers were hit before stable agent runs:
- `agent_rag_timeout = 10s` — too short for mxbai-embed-large (~4-8s per query). Fixed: `AGENT_RAG_TIMEOUT=60.0`.
- TPM limit (8000/min for this model on Groq on-demand) — the agent system prompt is ~1500 tokens, leaving little room per iteration.
- Native tool-call bypass requiring code changes to `react_agent_service.py` to recover from `failed_generation`.

### 6. Hallucination pattern: confident confabulation over honest refusal

In every failure case the model produced a specific, plausible-sounding answer rather than saying "I couldn't find this." Q6 (agent, gpt-oss-120b) cited a non-existent "brass Chinese seal" and the wrong story title. Q8 (chat) produced a real-sounding but wrong Holmes quote. Q1 (agent, gpt-oss-120b) reported adjacent deductions from the correct scene as if they were the target answer. The model never said "I didn't find it" — it always filled the gap from training knowledge.

---

## Roadmap — What to Do Next

The goal is to understand where each pipeline actually breaks, not to fix things before seeing the full picture. Run in order; don't skip ahead.

### Step 1 — Complete baseline (Q2–Q5), both endpoints

Run `test_sherlock_chat.py` and `test_sherlock_agent.py` with `--questions Q2,Q3,Q4,Q5`.

Purpose: confirm that single-chunk retrieval is reliable across both pipelines before drawing conclusions about multi-retrieval. If baseline is shaky, multi-retrieval results are uninterpretable.

### Step 2 — Run all multi-retrieval questions as-is (Q6–Q15), both endpoints

No prompt changes, no fixes. Run as-is and record:
- How many RAG calls does the agent make per question? Does it ever reach 2?
- Does the chat pipeline always fail at the architectural ceiling (1 call)?
- What does the agent do when the first retrieval is insufficient — does it retry or self-certify?

This step answers whether the checklist prompt is the problem, or whether the loop mechanics themselves are broken.

### Step 3 — Diagnose the agent loop based on Step 2 results

Two possible findings:

**Finding A — agent never makes a 2nd RAG call:** The checklist self-certification is the problem. The model marks [x] without a direct quote. Try prompt alternatives one at a time, each tested against Q6:
- Add "do not mark [x] unless you can quote the exact phrase from the retrieved text"
- Replace checklist with explicit sub-question decomposition ("search separately for each item")
- Remove the checklist entirely and rely on the model's own judgment

**Finding B — agent sometimes makes a 2nd RAG call but still fails:** The loop works, but query reformulation is poor. The model issues the same or similar query twice instead of targeting the missing piece.

### Step 4 — Hallucination traps (H1–H3), both endpoints

Independent of multi-retrieval. Can be done any time between steps.
Run `--questions H1,H2,H3` on both scripts and record whether the model refuses or confabulates.

---

## ⭐ Open Problems

### 1. Chat pipeline single-retrieval ceiling

Q6 and Q8 confirm that the finalization LLM runs with `tools=None`. Even when the model recognises its first retrieval was incomplete (Q6: it produced a second `rag_search` call that Groq rejected with HTTP 400), the architecture blocks it. This is the fundamental constraint for multi-part questions on `/chat/stream`.

### 2. `tool_use_failed` not detected by chat_service — Q4 empty answer (gpt-oss-120b)

**Symptom:** Q4 returned an empty answer. Groq returned HTTP 400 during the routing pass:

```
Failed to parse tool call arguments as JSON
code: tool_use_failed
failed_generation: {"name": "rag_search", "arguments": {"query": "Hol"}}
```

The model tried to generate a native tool call but the JSON was truncated (`"query": "Hol"` — only 3 characters). Groq rejected it.

**Root cause:** `chat_service.py:621` detects `tool_call_generation_failed` by checking the error **message text** for `"call a function"` or `"failed_generation"`. But Groq's error message here is `"Failed to parse tool call arguments as JSON"` — neither pattern matches.

```python
# Current detection (chat_service.py:621)
msg = str(event.get("message", "")).lower()
if "call a function" in msg or "failed_generation" in msg:
    tool_call_generation_failed = True
```

Because the error is not recognised, the code falls through to the direct-answer path. But the routing system prompt (`"You are a routing agent"`) is still in `effective_messages` alongside the appended `DIRECT_ANSWER_SYSTEM_MESSAGE` — two conflicting instructions. The model produces empty output.

**Fix:** Add `"tool_use_failed"` or `"failed to parse tool"` to the detection pattern. The `failed_generation` field IS present on the error event (Groq returns it, and `groq_client.py` extracts it at line 304) — the only gap is the message-text pattern match.

```python
# Proposed fix
if "call a function" in msg or "failed_generation" in msg or "tool_use_failed" in msg:
    tool_call_generation_failed = True
```

Alternatively, check `event.get("failed_generation")` directly instead of relying on the message string — this covers any future Groq error format changes.
