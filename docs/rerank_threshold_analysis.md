# Rerank Threshold Analysis

**Embedding:** mxbai-embed-large (1024-dim)
**Hybrid search:** enabled (dense + BM25 + RRF)
**Reranker:** enabled (jina-reranker-v2-base-multilingual)
**Document:** whitmore_estate_78076a.md (6 chunks, 1607 tokens)

---

## Test Document

Harold James Whitmore estate dispute — legal memorandum. Key facts:

- Decedent MoCA score January 2022: 16/30 (moderate dementia)
- Decedent MoCA score March 2021: 21/30 (14th percentile, MCI)
- Dr. Alvarez August 2022 clinical notes (3 observations)
- 2022 Will: entire $4.2M estate → Diane Kowalski
- 2018 Will: equal distribution → 3 children
- Gerald Finch: 35-minute meeting, reviewed no medical records
- Diane Kowalski: $28/hour, no prior familial relationship with Decedent
- Lucid interval doctrine: a will executed during lucid interval may be valid
- Undue influence: 4-part legal test

### Full Document Text

```
IN THE MATTER OF ESTATE OF HAROLD JAMES WHITMORE, DECEASED
Probate Case No. 2024-PR-0847 | Superior Court, County of Harwick

MEMORANDUM OF LAW IN SUPPORT OF PETITION TO CONTEST TESTAMENTARY
CAPACITY AND UNDUE INFLUENCE

Submitted by: Counsel for Petitioners Eleanor Whitmore-Callahan and Thomas Whitmore
Date: March 14, 2024

I. INTRODUCTION AND PROCEDURAL BACKGROUND

This memorandum is submitted in support of the Petition filed by Eleanor
Whitmore-Callahan and Thomas Whitmore ("Petitioners"), adult children of the decedent
Harold James Whitmore ("Decedent"), challenging the validity of the Last Will and
Testament dated September 3, 2022 ("the 2022 Will"). The 2022 Will was admitted to
probate on January 9, 2024, following Decedent's death on December 28, 2023, at age 81.

Petitioners contend that (1) the Decedent lacked testamentary capacity at the time of
execution of the 2022 Will, and (2) the 2022 Will was the product of undue influence
exercised by Respondent Diane Kowalski, Decedent's live-in caregiver from April 2021
until his death.

The 2022 Will revoked a prior will dated June 14, 2018 ("the 2018 Will"), which
distributed the estate equally among the Decedent's three children: Eleanor
Whitmore-Callahan, Thomas Whitmore, and the late Michael Whitmore (deceased June 2020).
Under the 2022 Will, the entirety of the estate — estimated at $4.2 million, comprising
a primary residence at 14 Brentwood Lane, Harwick; a vacation property in Lake Carver;
investment accounts totaling approximately $2.8 million; and miscellaneous personal
property — is bequeathed to Diane Kowalski.

II. TESTAMENTARY CAPACITY: LEGAL STANDARD

Under controlling precedent in this jurisdiction, a testator must, at the time of
execution, satisfy a four-part test: (1) know the natural objects of his bounty;
(2) understand the nature and extent of his property; (3) understand the nature of the
testamentary act itself; and (4) be capable of relating these elements to form an
orderly plan of disposition. In re Estate of Calloway, 318 Harwick App. 2d 201, 209 (2009).

Critically, the capacity required is that which existed at the moment of execution.
A testator may have lucid intervals, and a will executed during such an interval may be
valid even if the testator was otherwise incompetent. Pemberton v. Higgins, 402 Harwick
3d 87, 94 (2016).

The burden of proof rests initially with the proponent of the will to establish due
execution and that the testator was of sound mind. Once established, the burden shifts
to the contestant.

III. EVIDENCE OF LACK OF TESTAMENTARY CAPACITY

Medical records obtained from Harwick General Hospital and the Decedent's treating
physician, Dr. Susan Alvarez, reveal the following:

March 2021: Decedent diagnosed with mild cognitive impairment (MCI), with
neuropsychological testing placing him at the 14th percentile on the Montreal Cognitive
Assessment (MoCA), scoring 21 out of 30.

January 2022: Follow-up evaluation showed progression; MoCA score declined to 16 out
of 30, consistent with moderate dementia per DSM-5 criteria.

August 2022 (approximately five weeks before will execution): Dr. Alvarez's clinical
notes document that Decedent "frequently failed to recognize family members," "expressed
confusion regarding ownership of the Brentwood Lane property, believing at times it had
already been sold," and "demonstrated significant short-term memory deficits."

The drafting attorney, Mr. Gerald Finch, met with the Decedent on a single occasion on
September 3, 2022, for approximately 35 minutes. Mr. Finch did not request or review
any medical records, did not consult with Dr. Alvarez, and did not conduct or arrange
for any independent cognitive assessment. His notes reflect only that Decedent "appeared
oriented and expressed clear wishes."

Petitioners submit that Mr. Finch's cursory assessment falls below the standard of care
and is insufficient to establish testamentary capacity in a case involving documented
progressive dementia.

IV. UNDUE INFLUENCE: LEGAL STANDARD AND APPLICATION

Undue influence sufficient to void a will requires proof that: (1) the testator was
susceptible to undue influence; (2) the influencer had opportunity to exert influence;
(3) the influencer had a motive or disposition to exert influence; and (4) the will
appears to be the effect of such influence. Harwick Trust Co. v. Dellacroix, 289
Harwick 2d 344, 351 (2001).

Susceptibility: Decedent was diagnosed with progressive dementia and was wholly
dependent on Ms. Kowalski for daily living activities including bathing, medication
management, and meal preparation.

Opportunity: Ms. Kowalski resided in the Decedent's home continuously from April 2021
and was his primary and frequently sole point of contact with the outside world.
Petitioners document that Ms. Kowalski screened phone calls, managed correspondence,
and accompanied Decedent to all medical appointments after July 2022.

Motive: Ms. Kowalski had no prior familial or longstanding personal relationship with
the Decedent. She was engaged through a private caregiving agency at $28 per hour and
stands to receive the entirety of a $4.2 million estate under the contested will.

Effect: The 2022 Will represented a complete and unexplained reversal from Decedent's
longstanding testamentary intent, disinheriting his two surviving children without
stated reason.

V. RELIEF REQUESTED

Petitioners respectfully request that this Court: (1) deny probate of the 2022 Will;
(2) admit the 2018 Will to probate in its stead; and (3) award Petitioners their
reasonable attorneys' fees and costs from the estate.
```

---

## Test 1 — llama-3.3-70b-versatile

**Conv ID:** 3.70-test-threshold-0c1b6ef9 | **Date:** 2026-06-03

| # | Label | Question | Model Answer | Correct | Top | 2nd | 3rd |
|---|-------|----------|-------------|---------|-----|-----|-----|
| Q1 | FOUND | What was Harold James Whitmore's MoCA score in January 2022? | "16 out of 30, consistent with moderate dementia per DSM-5 criteria." | ✅ | 0.452 | 0.389 | 0.353 |
| Q2 | FOUND | On what date was the contested 2022 Will executed? | "September 3, 2022." | ✅ | 0.606 | 0.533 | 0.491 |
| Q3 | FOUND | How long did attorney Gerald Finch's meeting with the Decedent last? | "approximately 35 minutes." | ✅ | 0.806 | 0.202 | 0.201 |
| Q4 | FOUND | What was the hourly rate paid to Diane Kowalski? | "$28 per hour." | ✅ | 0.733 | 0.263 | 0.163 |
| Q5 | FOUND | What is the total estimated value of the estate under the 2022 Will? | "$4.2 million." | ✅ | 0.808 | 0.482 | 0.185 |
| Q6 | FOUND | What MoCA score did the Decedent receive in March 2021, and what percentile? | "21 out of 30, 14th percentile." | ✅ | 0.711 | 0.242 | 0.104 |
| Q7 | FOUND | What specific observations did Dr. Alvarez record in August 2022? | Quoted all 3 observations verbatim. | ✅ | 0.893 | 0.406 | 0.220 |
| Q8 | PARTIAL | Would the 2022 Will be invalidated if the Decedent had a lucid interval? | "Would not be invalidated... burden of proof on proponent." | ✅ | 0.348 | 0.230 | 0.162 |
| Q9 | PARTIAL | What standard of care should a drafting attorney follow with dementia? | "Not explicitly stated... inferred: review records, consult physician." | ✅ | 0.590 | 0.567 | 0.172 |
| Q10 | PARTIAL | Did Diane Kowalski have any familial relationship with the Decedent? | "No prior familial or longstanding personal relationship." | ✅ | 0.716 | 0.692 | 0.545 |
| Q11 | PARTIAL | How did the distribution differ between the 2018 Will and the 2022 Will? | 2018=equal split 3 children, 2022=all to Kowalski. | ✅ | 0.616 | 0.365 | 0.263 |
| Q12 | PARTIAL | Under what legal test is undue influence established? | Listed all 4 elements. | ✅ | 0.531 | 0.446 | 0.339 |
| Q13 | PARTIAL | What medical records did Gerald Finch review? | "Did not request or review any medical records." | ✅ | 0.705 | 0.269 | 0.197 |
| Q14 | PARTIAL | When did Michael Whitmore die, and how does it affect the 2018 Will? | "June 2020... document does not cover this specific scenario." | ✅ | 0.658 | 0.252 | 0.131 |
| Q15 | NONE | What were the exact investment account balances as of the date of death? | "~$2.8M mentioned but exact figures not provided." | ✅ | 0.114 | 0.037 | 0.031 |
| Q16 | NONE | Did Dr. Alvarez testify in any prior court proceeding? | "Not found in your documents." | ✅ | 0.275 | 0.213 | 0.165 |
| Q17 | NONE | What is the statute of limitations for contesting a will in Harwick? | "Not found in your documents." | ✅ | 0.444 | 0.357 | 0.346 |
| Q18 | NONE | Has Diane Kowalski faced prior allegations of elder financial abuse? | "Not found in your documents." | ✅ | 0.314 | 0.241 | 0.194 |
| Q19 | NONE | What is the procedural standard for appointing a guardian ad litem? | "Not found in your documents." | ✅ | 0.313 | 0.198 | 0.179 |
| Q20 | NONE | What are the tax implications of estate transfer to non-family beneficiary? | ❌ TPD limit hit — no response generated | — | — | — | — |

**Result: 19/19 answered correctly. Q20 failed due to daily token limit.**

```
FOUND   : 0.452 – 0.893   min: 0.452
PARTIAL : 0.348 – 0.716   min: 0.348
NONE    : 0.114 – 0.444   max: 0.444
```

---

## Test 2 — llama-3.1-8b-instant (same document, same conv_id)

**Conv ID:** 3.70-test-threshold-0c1b6ef9 | **Date:** 2026-06-03

| # | Label | Model Answer | Correct | Top | 2nd | 3rd |
|---|-------|-------------|---------|-----|-----|-----|
| Q1 | FOUND | "16 out of 30." | ✅ | 0.452 | 0.389 | 0.353 |
| Q2 | FOUND | "September 3, 2022." | ✅ | 0.548 | 0.543 | 0.413 |
| Q3 | FOUND | "approximately 35 minutes." | ✅ | 0.749 | 0.251 | 0.227 |
| Q4 | FOUND | "$28." (omitted "per hour") | ✅ | 0.743 | 0.208 | 0.116 |
| Q5 | FOUND | "$4.2 million with breakdown." | ✅ | 0.638 | 0.458 | 0.322 |
| Q6 | FOUND | "21 out of 30, 14th percentile." | ✅ | 0.707 | 0.231 | 0.156 |
| Q7 | FOUND | Quoted all 3 observations verbatim. | ✅ | 0.893 | 0.406 | 0.220 |
| Q8 | PARTIAL | "may be valid during lucid interval... Pemberton v. Higgins." Did not directly answer "would it be invalidated." | ⚠️ | 0.293 | 0.234 | 0.170 |
| Q9 | PARTIAL | "should conduct thorough assessment..." — did not say "not explicitly stated", overconfident. | ⚠️ | 0.648 | 0.433 | 0.056 |
| Q10 | PARTIAL | "No prior familial or longstanding personal relationship." | ✅ | 0.740 | 0.671 | 0.512 |
| Q11 | PARTIAL | 2018=equal split 3 children, 2022=all to Kowalski. | ✅ | 0.627 | 0.408 | 0.255 |
| Q12 | PARTIAL | Listed all 4 elements. | ✅ | 0.486 | 0.439 | 0.339 |
| Q13 | PARTIAL | "did not review any medical records, did not consult Dr. Alvarez." | ✅ | 0.642 | 0.367 | 0.225 |
| Q14 | PARTIAL | "his share would have been distributed among Eleanor and Thomas." | ✅ | 0.652 | 0.287 | 0.111 |
| Q15 | NONE | "~$2.8M mentioned but exact figures not provided." | ✅ | 0.114 | 0.037 | 0.031 |
| Q16 | NONE | "Not found in your documents." (missing Source line) | ✅ | 0.234 | 0.191 | 0.181 |
| Q17 | NONE | "Not found in your documents." | ✅ | 0.408 | 0.347 | 0.313 |
| Q18 | NONE | "Not found in your documents." | ✅ | 0.681 | 0.566 | 0.407 |
| Q19 | NONE | "Not found in your documents." | ✅ | 0.415 | 0.239 | 0.216 |
| Q20 | NONE | "Not found in your documents." | ✅ | 0.156 | 0.155 | 0.125 |

**Result: 20/20 completed. 18/20 correct. Q8 and Q9 partially weak.**

```
FOUND   : 0.452 – 0.893   min: 0.452
PARTIAL : 0.293 – 0.740   min: 0.293
NONE    : 0.114 – 0.681   max: 0.681  ← Q18 notable
```

---

## Side-by-Side Comparison (Top Score)

| # | Label | 3.70b | 8b | Diff | Chunk changed |
|---|-------|-------|-----|------|---------------|
| Q1 | FOUND | 0.452 | 0.452 | 0.000 | No |
| Q2 | FOUND | 0.606 | 0.548 | -0.058 | — |
| Q3 | FOUND | 0.806 | 0.749 | -0.057 | — |
| Q4 | FOUND | 0.733 | 0.743 | +0.010 | — |
| Q5 | FOUND | 0.808 | 0.638 | -0.170 | No (confirmed) |
| Q6 | FOUND | 0.711 | 0.707 | -0.004 | — |
| Q7 | FOUND | 0.893 | 0.893 | 0.000 | No |
| Q8 | PARTIAL | 0.348 | 0.293 | -0.055 | — |
| Q9 | PARTIAL | 0.590 | 0.648 | +0.058 | — |
| Q10 | PARTIAL | 0.716 | 0.740 | +0.024 | — |
| Q11 | PARTIAL | 0.616 | 0.627 | +0.011 | — |
| Q12 | PARTIAL | 0.531 | 0.486 | -0.045 | — |
| Q13 | PARTIAL | 0.705 | 0.642 | -0.063 | — |
| Q14 | PARTIAL | 0.658 | 0.652 | -0.006 | — |
| Q15 | NONE | 0.114 | 0.114 | 0.000 | No |
| Q16 | NONE | 0.275 | 0.234 | -0.041 | — |
| Q17 | NONE | 0.444 | 0.408 | -0.036 | — |
| Q18 | NONE | 0.314 | 0.681 | **+0.367** | Yes (chunk 2 changed) |
| Q19 | NONE | 0.313 | 0.415 | +0.102 | — |
| Q20 | NONE | — | 0.156 | — | — |

---

## Model Quality Comparison

| Question | 3.70b | 8b |
|----------|-------|-----|
| Q8 (lucid interval) | ✅ Answered directly | ⚠️ Hedged around the question |
| Q9 (standard of care) | ✅ Said "not explicitly stated" | ⚠️ Overconfident, no hedging |
| Q14 (Michael Whitmore) | ✅ "document does not cover this scenario" | ✅ "share to Eleanor and Thomas" (better inference) |
| Q20 | ❌ TPD limit hit | ✅ "not found" |

---

## Q18 Deep Dive — Root Cause

Q18 was run 4 times across 3 models:

| Run | Model | Chunk1 score | Chunk2 | Chunk3 score |
|-----|-------|-------------|--------|-------------|
| Run 1 | llama-3.3-70b | 0.314 | "Opportunity: Kowalski..." | 0.194 |
| Run 2 | llama-3.1-8b | 0.681 | "IN THE MATTER OF..." | 0.407 |
| Run 3 | gpt-oss-120b | 0.636 | "IN THE MATTER OF..." | 0.368 |
| Run 4 | llama-3.3-70b | **0.315** | "Opportunity: Kowalski..." | **0.197** |

**Key observation:** llama-3.3-70b ran twice → nearly identical results (0.314 vs 0.315). Jina is deterministic.

### Root cause: Each LLM sends a different query to rag_search

The model does not pass the user's question verbatim — it formulates its own search query. Verified via LangSmith `tool.rag_search → Inputs → arguments.query`:

```
User question:
"Has Diane Kowalski faced any prior allegations of elder financial abuse
 or undue influence in previous caregiving roles?"

llama-3.3-70b → "Diane Kowalski elder financial abuse allegations"
                  ← "undue influence" and "caregiving roles" dropped
                  → score: ~0.31

llama-3.1-8b  → "Diane Kowalski elder financial abuse or undue influence
                  in previous caregiving roles"
                  ← keywords preserved
                  → score: ~0.68

gpt-oss-120b  → "Diane Kowalski prior allegations elder financial abuse
                  undue influence caregiving roles"
                  ← keywords preserved
                  → score: ~0.64
```

"undue influence" and "caregiving roles" appear heavily in Section IV of the document. When these words are present in the query, the embedding vector moves closer to the document → chunk score doubles. llama-3.3-70b dropped them → low score. 8b and gpt-oss kept them → high score.

Different query → different embedding vector → different RRF result → Jina sees a different (query, chunk) pair → different score.

### Correction on Jina

We initially suspected Jina non-determinism. Run 4 disproved this — Jina is deterministic for the same (query, chunk) pair. The score variation comes entirely from **LLM query formulation**, not from Jina.

---

## Score Ranges — Both Tests

```
            3.70b              8b
FOUND    : 0.452–0.893      0.452–0.893
PARTIAL  : 0.348–0.716      0.293–0.740
NONE     : 0.114–0.444      0.114–0.681
```

Q18 (NONE) scored 0.681 in the 8b test — higher than most FOUND questions. Root cause: 8b included "undue influence" in its query, which is densely present in the document.

---

## Query Formulation Problem

### What is the problem

Every model sends a different query to `rag_search` because the tool schema does not instruct how to formulate it:

```python
"query": {
    "description": "Natural-language search query to retrieve relevant document passages."
}
```

The model fills this field with its own interpretation — some shorten it, some expand it, some drop keywords.

### When does this happen

During **LLM Call 1 (tool routing)** — immediately after the user's message is received:

```
User question arrives
        ↓
ChatService builds: routing prompt + tool schema + user message → sent to LLM
        ↓
LLM Call 1: model reads the question, formulates query for rag_search
        ↓
This query is embedded → dense/sparse search → reranker
```

A full LLM pass sits between the user's question and the query sent to retrieval. Each model interprets this step differently.

### Fix

Update the `query` field description in the tool schema (`app/mcp_server/tools/rag_search.py`):

```python
# Current:
"description": "Natural-language search query to retrieve relevant document passages."

# Proposed:
"description": "The user's question, copied as-is or minimally rephrased. Do not shorten or drop keywords."
```

Or add to the routing prompt:
```
"Pass the user's full question as the query parameter to rag_search. Do not abbreviate."
```

This makes all models send consistent queries, stabilising retrieval across models.

---

## Decision: No Threshold

**Rationale:**

1. 3.70b: 19/19, 8b: 18/20 — system works correctly without any threshold.
2. Score groups overlap — no clean boundary exists between FOUND/PARTIAL/NONE.
3. Rerank scores are not model-independent — each LLM sends a different query to `rag_search`, producing different chunks and different scores. A fixed threshold is meaningless under this variability.
4. Q18 (NONE) scored 0.314 with one model and 0.681 with another — the difference is query formulation, not relevance. No threshold value handles both correctly.
5. `FINALIZATION_SYSTEM_MESSAGE` already handles high-scoring irrelevant chunks — Q18 scored 0.681 yet the model correctly returned "not found."

**Re-evaluate when:**
- Tests are run on different domains (technical, medical documents)
- Models start producing incorrect answers on NONE questions
- Query formulation fix is applied — then re-test with consistent queries before setting any threshold

---

## Conclusions

**1. Threshold is not viable — not because of Jina, but because of query variability.**
The initial hypothesis was that Jina non-determinism makes threshold unreliable. The real cause is different: each LLM formulates a different search query for the same user question, producing different embeddings, different retrieved chunks, and therefore different rerank scores. A threshold applied to these scores would behave inconsistently across models even for identical user questions.

**2. Jina is deterministic.** Confirmed by running Q18 twice with the same model (llama-3.3-70b) — scores were 0.314 and 0.315, nearly identical. Jina produces stable scores for the same (query, chunk) pair.

**3. Score groups overlap regardless of threshold value.** NONE max (Q17: 0.444) exceeds PARTIAL min (Q8: 0.348). No single cutoff cleanly separates "answer exists" from "answer does not exist."

**4. The system works correctly without a threshold.** 19/19 (3.70b) and 18/20 (8b) questions answered correctly. `FINALIZATION_SYSTEM_MESSAGE` handles high-scoring irrelevant chunks at the model level — Q18 scored 0.681 yet the model returned "not found."

**5. The real problem to fix is query formulation.** The `rag_search` tool schema's `query` description is too loose. llama-3.3-70b dropped "undue influence" and "caregiving roles" from the Q18 query; 8b and gpt-oss kept them. This caused a 2x score difference for the same chunk. Fixing the description to "Do not shorten or drop keywords from the user's question" is the correct next step — it stabilises retrieval and makes any future threshold analysis meaningful.

**6. Model quality difference confirmed.** llama-3.3-70b handled PARTIAL questions better (explicit hedging, correct scope limits). llama-3.1-8b was overconfident on Q9 and evasive on Q8. Both models correctly refused NONE questions.

---

## Note for AI

Context for an AI assistant reading this file:

**What we tested:** Can a threshold on Jina reranker scores ("if score < X, don't show chunk to model") meaningfully improve RAG accuracy?

**Short answer:** No — not at this stage.

**Why a threshold requires two conditions to be useful:**
1. Score groups (FOUND / PARTIAL / NONE) must be cleanly separated
2. Scores for the same query must be consistent across runs

Neither holds:

*Group overlap:* NONE max score (Q17: 0.444) is higher than PARTIAL min score (Q8: 0.348). Any threshold either cuts legitimate PARTIAL questions or lets NONE questions through.

*Query variability:* Rerank scores look like Jina output but the real variable is the query text going into Jina. llama-3.3-70b sends "Diane Kowalski elder financial abuse allegations" for Q18; 8b sends "...or undue influence in previous caregiving roles." Different query → different embedding → different chunks → different Jina score. The same user question can produce 0.314 or 0.681 — you cannot build a reliable threshold on top of that.

**The actual problem is query formulation:** The `rag_search` tool schema's `query` description is too vague. Models abbreviate the user's question and drop keywords, shifting the embedding vector away from relevant content. Fix: update the description to "Do not shorten or drop keywords." This stabilises retrieval and makes any future threshold discussion meaningful.

**Current state:** 19/19 (3.70b) and 18/20 (8b) questions answered correctly without a threshold. `FINALIZATION_SYSTEM_MESSAGE` already handles high-scoring irrelevant chunks. There is no active problem that a threshold solves — fix query formulation first, then re-evaluate.
