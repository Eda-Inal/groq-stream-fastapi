# Agentic RAG — Multi-step Retrieval Test

## Context

Document under test: **Whitmore Estate** (`whitmore_estate_78076a.md`)  
A fictional probate case memo: Harold Whitmore's contested will, cognitive decline records,
undue influence claim against caregiver Diane Kowalski.

The goal of this test is to observe whether the current single-pass `rag_search` is
sufficient to answer complex questions, or whether multi-step retrieval (query
reformulation, multi-hop) is needed — establishing the baseline **before** the
agentic RAG loop is implemented.

---

## Setup

| Field | Value |
|---|---|
| Script | manual eval script (not included in repo) |
| Model | `openai/gpt-oss-120b` |
| User ID | `test_user` |
| Document | already in DB — no upload needed |
| Rate | 3 questions/minute (20s delay) |
| Each question | independent `conversation_id` |

---

## Questions & Expected Answers

**Q01.** How old was Harold Whitmore and when did he die?

> Harold James Whitmore died on December 28, 2023, at the age of 81.

**Q02.** On what date was the 2022 Will executed?

> The 2022 Will was executed on September 3, 2022.

**Q03.** Why is attorney Gerald Finch's capacity assessment considered inadequate and what medical evidence supports that conclusion?

> Gerald Finch met with the Decedent only once, for approximately 35 minutes, on the same day the will was signed. He did not review any medical records, did not consult with Dr. Alvarez, and did not arrange for any independent cognitive assessment. His notes only reflect that the Decedent "appeared oriented and expressed clear wishes." This is inadequate because medical records from just five weeks earlier documented that Harold frequently failed to recognize family members, was confused about ownership of his own property, and had significant short-term memory deficits. A single brief meeting without any medical due diligence falls below the standard of care required in a case involving documented progressive dementia.

**Q04.** What was Harold's cognitive condition in the weeks immediately before the will was executed, and which elements of the testamentary capacity test does that condition fail to satisfy?

> Approximately five weeks before the will was executed, Dr. Alvarez's August 2022 clinical notes recorded that Harold frequently failed to recognize family members, was confused about whether he still owned the Brentwood Lane property, and had significant short-term memory deficits. Under the four-part testamentary capacity test from In re Estate of Calloway, a testator must (1) know the natural objects of his bounty, (2) understand the nature and extent of his property, (3) understand the nature of the testamentary act, and (4) be capable of forming an orderly plan of disposition. Harold's failure to recognize family members undermines element (1); his confusion about property ownership undermines element (2); and his memory deficits call into question elements (3) and (4).

**Q05.** Is there any legitimate prior relationship between Diane Kowalski and the Decedent that could explain her inclusion in the will, and what does that mean under the undue influence test?

> No. Diane Kowalski had no prior familial or longstanding personal relationship with the Decedent. She was a professional caregiver hired through a private agency at $28 per hour starting April 2021. Under the undue influence test from Harwick Trust Co. v. Dellacroix, one required element is that the influencer had a motive or disposition to exert influence. The absence of any pre-existing relationship, combined with the fact that she stands to receive a $4.2 million estate, directly satisfies this motive element and eliminates any innocent explanation for her inclusion in the will.

**Q06.** What changed between the 2018 Will and the 2022 Will, and how is that change being used as legal argument?

> The 2018 Will distributed the estate equally among Harold's three children. The 2022 Will revoked it entirely and bequeathed the whole estate — approximately $4.2 million — to Diane Kowalski, disinheriting the two surviving children without any stated reason. Under the undue influence framework, this complete and unexplained reversal of longstanding testamentary intent serves as the "effect" element — the fourth prong of the test — suggesting the will reflects external pressure rather than the Decedent's genuine wishes.

**Q07.** How did Harold's cognitive decline progress from 2021 to 2022, and how does that progression support the susceptibility argument?

> In March 2021, Harold was diagnosed with mild cognitive impairment (MCI), scoring 21/30 on the MoCA, placing him at the 14th percentile. By January 2022, his MoCA score had declined to 16/30, consistent with moderate dementia under DSM-5 criteria. This documented progression directly supports the "susceptibility" element of the undue influence test, which is heightened where the testator suffers from cognitive decline, social isolation, or dependency on the alleged influencer — all three of which apply to Harold.

**Q08.** How does Diane Kowalski's control over Harold's daily life and external communications establish the opportunity element of undue influence?

> Diane Kowalski lived in Harold's home continuously from April 2021 until his death. After July 2022 — about two months before the will was signed — she screened his phone calls, managed his correspondence, and accompanied him to all medical appointments, making her his primary and frequently sole point of contact with the outside world. This establishes the "opportunity" element, which requires that the influencer had the means and access to exert pressure on the testator.

**Q09.** What two core claims must Petitioners prove together to have the 2022 Will invalidated and the 2018 Will admitted, and what is the key evidence for each?

> Petitioners must prove two independent claims: (1) Lack of Testamentary Capacity — that Harold did not meet the four-part capacity test at the moment of execution. Key evidence: the January 2022 MoCA score of 16/30, Dr. Alvarez's August 2022 clinical notes, and Finch's failure to conduct any meaningful evaluation. (2) Undue Influence — that Kowalski exploited Harold's vulnerability to substitute her wishes for his. Key evidence: Harold's total dependency on Kowalski, her control over his communications, her financial motive, and the complete reversal from the 2018 Will with no stated rationale. If either claim succeeds independently, the court can deny probate of the 2022 Will and admit the 2018 Will in its place.

**Q10.** If the court applies the lucid interval doctrine, accepting that Harold may have had a moment of clarity when signing, what evidence can Petitioners use to counter that argument?

> The lucid interval doctrine from Pemberton v. Higgins holds that a will executed during a moment of clarity may be valid even if the testator was otherwise incompetent. Petitioners can counter this with: (1) Dr. Alvarez's August 2022 notes — just five weeks before execution — documenting persistent confusion and memory failure, making a genuine lucid interval on September 3 unlikely; (2) Finch's 35-minute meeting, which was too brief and clinically uninformed to reliably detect a true lucid interval, as he reviewed no medical records and consulted no specialists; and (3) the complete absence of any independent cognitive assessment on or around the date of execution, meaning there is no objective contemporaneous evidence of lucidity beyond Finch's bare notation that Harold "appeared oriented."

---

## Results — BEFORE (single-pass rag_search)

> Run date: 2026-06-16
> Run ID: 0cd3c49a
> Model: openai/gpt-oss-120b

**Q01.** How old was Harold Whitmore and when did he die?

> Harold James Whitmore was 81 years old when he died, and his death occurred on December 28, 2023.
> Source: whitmore_estate_993b0a.md

**Result: ✅ Correct**

---

**Q02.** On what date was the 2022 Will executed?

> The 2022 Will was executed on September 3, 2022.
> Source: whitmore_estate_a82f0f.md

**Result: ✅ Correct**

---

**Q03.** Why is attorney Gerald Finch's capacity assessment considered inadequate and what medical evidence supports that conclusion?

> Attorney Gerald Finch's assessment is deemed inadequate because it was a brief, single-meeting observation that relied on no medical records, no consultation with the decedent's physician, and no independent cognitive testing. Finch spent only about 35 minutes with the decedent, noted only that the decedent "appeared oriented and expressed clear wishes," and failed to request or review any of the decedent's medical documentation or to arrange a formal capacity evaluation. Petitioners argue that this cursory approach falls below the standard of care for establishing testamentary capacity, especially given the decedent's documented cognitive decline.
>
> The medical evidence underscoring the inadequacy includes:
> - January 2022 neurocognitive testing: MoCA score dropped to 16/30, consistent with moderate dementia under DSM-5 criteria.
> - August 2022 clinical notes from Dr. Alvarez (≈ five weeks before the will was executed): Harold "frequently failed to recognize family members," was "confused regarding ownership of the Brentwood Lane property," and showed "significant short-term memory deficits."
>
> Source: whitmore_estate_a82f0f.md

**Result: ✅ Correct**

---

**Q04.** What was Harold's cognitive condition in the weeks immediately before the will was executed, and which elements of the testamentary capacity test does that condition fail to satisfy?

> In the weeks immediately before the will was executed (approximately five weeks prior, in August 2022), Dr. Alvarez's clinical notes describe: MoCA score of 16/30 (moderate dementia under DSM-5), frequently failed to recognize family members, confused about ownership of the Brentwood Lane property, and significant short-term memory deficits.
>
> The documents provide this description of Harold's cognitive impairment but do **not** list the specific elements of the testamentary-capacity test nor explain which of those elements his condition fails to satisfy. Consequently, the required information about which capacity-test elements are not met is not covered in the supplied documents.
>
> Source: whitmore_estate_a82f0f.md

**Result: ✅ Correct** — When retested with a single document in DB, the model correctly retrieved the four-part capacity test from Section II and mapped Harold's symptoms to elements (1), (2), and (4). The initial failure was caused by retrieval noise from three duplicate Whitmore documents; two copies (id=43, id=77) were deleted and the question answered correctly on retest.

---

**Q05.** Is there any legitimate prior relationship between Diane Kowalski and the Decedent that could explain her inclusion in the will, and what does that mean under the undue influence test?

> Diane Kowalski had no prior familial or longstanding personal relationship with the Decedent. She was a professional caregiver hired through a private agency at $28 per hour starting April 2021. Under the undue influence analysis: Opportunity — lived continuously with the decedent and was his primary point of contact; Motive — no family tie, stands to receive the whole estate; Effect — 2022 will completely reversed the 2018 plan without stated reason.
>
> Source: whitmore_estate_78076a.md

**Result: ✅ Correct**

---

**Q06.** What changed between the 2018 Will and the 2022 Will, and how is that change being used as legal argument?

> The 2018 Will distributed the estate equally among Harold's three children. The 2022 Will revoked it entirely and bequeathed the whole estate (~$4.2 million) to Diane Kowalski, disinheriting the children. Petitioners argue this "complete and unexplained reversal" of longstanding testamentary intent is the product of undue influence.
>
> Source: whitmore_estate_993b0a.md, whitmore_estate_a82f0f.md, whitmore_estate_78076a.md

**Result: ✅ Correct**

---

**Q07.** How did Harold's cognitive decline progress from 2021 to 2022, and how does that progression support the susceptibility argument?

> March 2021: diagnosed with MCI, MoCA 21/30 (14th percentile). January 2022: MoCA fell to 16/30, meeting DSM-5 criteria for moderate dementia. August 2022: frequently failed to recognize family members, confused about property, significant short-term memory deficits. This progression demonstrates heightened susceptibility to undue influence, especially combined with total dependency on Kowalski for daily living activities.
>
> Source: whitmore_estate_a82f0f.md

**Result: ✅ Correct**

---

**Q08.** How does Diane Kowalski's control over Harold's daily life and external communications establish the opportunity element of undue influence?

> Kowalski lived in Harold's home continuously from April 2021 and became his primary — often sole — point of contact. She screened phone calls, managed correspondence, and accompanied him to every medical appointment after July 2022. This level of control and proximity satisfies the "opportunity" element.
>
> Source: whitmore_estate_78076a.md

**Result: ✅ Correct**

---

**Q09.** What two core claims must Petitioners prove together to have the 2022 Will invalidated and the 2018 Will admitted, and what is the key evidence for each?

> Petitioners must prove **both**: (1) Lack of Testamentary Capacity — four-part test not satisfied, key evidence: January 2022 MoCA 16/30, Dr. Alvarez's August 2022 notes, Finch's inadequate evaluation. (2) Undue Influence — Kowalski exploited Harold's vulnerability, key evidence: continuous residence, control over communications, no prior relationship, financial motive, complete reversal from 2018 Will.
>
> Source: whitmore_estate_78076a.md, whitmore_estate_a82f0f.md

**Result: ❌ Fail** — Model said Petitioners must prove **both** claims together. The document says **either** claim succeeding independently is sufficient to deny probate of the 2022 Will. Critical logical error; model conflated "two claims raised" with "both must be proven."

---

**Q10.** If the court applies the lucid interval doctrine, accepting that Harold may have had a moment of clarity when signing, what evidence can Petitioners use to counter that argument?

> Petitioners can counter with: (1) Progressive decline documented — MoCA 21/30 (March 2021) falling to 16/30 (January 2022), making a genuine lucid interval on September 3 unlikely; (2) Finch's 35-minute meeting was too brief and clinically uninformed to reliably detect a true lucid interval; (3) No independent cognitive assessment on or around the date of execution — only Finch's bare notation that Harold "appeared oriented"; (4) Undue influence factors (dependency, Kowalski's control) further erode any claim of clarity.
>
> Source: whitmore_estate_993b0a.md, whitmore_estate_78076a.md, whitmore_estate_a82f0f.md

**Result: ✅ Correct**

---

**Score: 9/10**

Failed: Q09 (logical error: model said "must prove both" — document states "if either claim succeeds independently")

---

## Results — AFTER (agentic RAG loop)

---

### Q04 — llama-3.3-70b-versatile

> Run date: 2026-06-16
> Script: manual eval script (not included in repo)
> Endpoint: `/api/v1/agent/stream`
> Iterations: 3 thoughts, 2 actions, 2 observations

**Model answer:**

> Harold's cognitive condition in the weeks immediately before the will was executed was one of moderate dementia, with a MoCA score of 16 out of 30, and significant short-term memory deficits. This condition likely fails to satisfy the first, second, and possibly fourth elements of the testamentary capacity test, which require that the testator know the natural objects of his bounty, understand the nature and extent of his property, and be capable of relating these elements to form an orderly plan of disposition.
> Source: whitmore_estate_78076a.md

**Agent trace:**
- Iteration 1: searched "Harold cognitive condition weeks before will executed testamentary capacity test" → got MoCA 16/30, Dr. Alvarez notes
- Iteration 2: searched "Harold cognitive condition impact on understanding property and relationships and will" → confirmed dependency/confusion details
- Final answer on iteration 2

**Result: ⚠️ Partial** — Elements 1 and 2 correctly identified. Element 4 mentioned as "possibly." Element 3 (understanding the nature of the testamentary act itself) missed entirely. Expected answer flags elements 3 and 4 as both undermined by memory deficits. Second rag_search call (iteration 2) retrieved lower-relevance chunks (rerank-score 0.187 vs 0.540 first call) — did not add new information, wasted an iteration.

---

### Q01 — llama-3.3-70b-versatile

> Run date: 2026-06-16
> Script: manual eval script (not included in repo)
> Endpoint: `/api/v1/agent/stream`
> Iterations: 2 thoughts, 1 action, 1 observation

**Model answer:**

> Harold Whitmore was 81 years old when he died on December 28, 2023.
> Source: whitmore_estate_78076a.md

**Agent trace:**
- Iteration 1: searched "Harold Whitmore age death" → got probate case header with age and death date directly
- Final answer on iteration 1

**Result: ✅ Correct** — Single retrieval was sufficient; fact was present in the first chunk returned (rerank-score 0.595).

---

### Q — llama-3.3-70b-versatile (cross-test)

> Run date: 2026-06-16
> Script: manual eval script (not included in repo)
> Endpoint: `/api/v1/agent/stream`
> Iterations: 3 thoughts, 2 actions, 2 observations

**Question:** Based on Harold's cognitive condition in the weeks before the will was executed, and considering both the testamentary capacity test and the undue influence test, was Harold more vulnerable to capacity failure or to undue influence — and why?

**Model answer:**

> Based on the information provided and the legal standards outlined, Harold was more vulnerable to undue influence due to his dependency on Diane Kowalski, her control over his daily life, and the significant benefit she stands to gain from the 2022 Will. While his dementia undoubtedly compromised his testamentary capacity, the evidence of undue influence, including opportunity, motive, and the drastic change in the will's disposition, is substantial.
> Source: whitmore_estate_78076a.md

**Agent trace:**
- Iteration 1: searched "Harold cognitive condition weeks before will execution testamentary capacity test undue influence test" → MoCA 16/30, Dr. Alvarez notes, undue influence elements
- Iteration 2: searched "testamentary capacity vs undue influence in cases of dementia and caregiver involvement" → Finch's inadequate assessment, Kowalski control details (rerank-score 0.641)
- Final answer on iteration 2

---

### Q (cross-test, AFTER checkbox prompt) — llama-3.3-70b-versatile

> Run date: 2026-06-16
> Script: manual eval script (not included in repo)
> Endpoint: `/api/v1/agent/stream`
> Iterations: 3 thoughts, 2 actions, 2 observations
> Prompt version: checkbox mechanism added to `_ROLE_WITH_TOOLS`

**Question:** Based on Harold's cognitive condition in the weeks before the will was executed, and considering both the testamentary capacity test and the undue influence test, was Harold more vulnerable to capacity failure or to undue influence — and why?

**Agent trace:**
- Iteration 1 — Thought: listed 4-item checklist `[ ] 1. cognitive condition [ ] 2. capacity test [ ] 3. undue influence test [ ] 4. comparative analysis`. Searched "Harold cognitive condition before will execution".
  Observation: MoCA 16/30, Dr. Alvarez notes, all four capacity elements, all four undue influence elements returned in one chunk.
  → marked items 1, 2, 3 as `[x]`, item 4 still `[ ]`
- Iteration 2 — Thought: updated checklist, item 4 `[ ]` still open. Searched "comparing vulnerability to capacity failure and undue influence in cases of dementia".
  Observation: Finch's inadequate assessment, Kowalski control details (rerank-score 0.459).
  → marked item 4 `[x]`, "All items complete. Writing Final Answer."

**Model answer:**

> Based on the information provided, Harold was more vulnerable to undue influence due to his dependency on Ms. Kowalski and her significant control over his interactions with the outside world, as well as her potential motive to exert influence given that she was the sole beneficiary of the 2022 Will. While Harold's moderate dementia also suggests a vulnerability to capacity failure, the circumstances surrounding the execution of the will and Ms. Kowalski's role in his life indicate a higher risk of undue influence.
> Source: whitmore_estate_78076a.md

**Result vs. previous run:**

| | Before (no checklist) | After (checklist) |
|---|---|---|
| Plan tracking | None — iterated freely | 4-item `[ ]`→`[x]` checklist, visible in every Thought |
| Final Answer gate | Stopped after 2 searches regardless | Only wrote Final Answer after all `[x]` |
| Depth | Surface-level summary | Listed both tests' elements before concluding |

**Result: ✅ Checklist mechanism working** — model tracked its plan, updated checkboxes after each Observation, and only proceeded to Final Answer when all items were marked `[x]`. Answer depth improved over the pre-checklist run.

---

### Q (plan tracking test 1, AFTER checkbox prompt) — llama-3.3-70b-versatile

> Run date: 2026-06-16
> Script: manual eval script (not included in repo)
> Endpoint: `/api/v1/agent/stream`
> Iterations: 3 thoughts, 2 actions, 2 observations

**Question:** If Petitioners succeed on the undue influence claim but fail on the testamentary capacity claim, what is the legal outcome for the 2022 Will — and what evidence would have been sufficient to prove capacity at the time of signing?

**Agent trace:**
- Thought 1: listed 2-item checklist `[ ] 1. legal outcome if undue influence succeeds alone [ ] 2. evidence sufficient to prove capacity`. Searched item 1.
- Observation 1: Petitioners contend both claims; if either succeeds independently the 2022 Will is voided. → marked `[x] 1`, item 2 still `[ ]`, ran second search.
- Observation 2: four-part Calloway test, MoCA, testimony. → marked `[x] 2`, "All items complete. Writing Final Answer."

**Model answer:**

> If the Petitioners succeed on the undue influence claim but fail on the testamentary capacity claim, the 2022 Will would be voided due to being the product of undue influence, regardless of the outcome on the testamentary capacity claim. To prove testamentary capacity, evidence could include satisfaction of the four-part test outlined in In re Estate of Calloway — knowing the natural objects of one's bounty, understanding the nature and extent of one's property, understanding the nature of the testamentary act, and being capable of forming an orderly plan of disposition — as well as medical records, cognitive assessments like the MoCA, and testimony from individuals who interacted with the testator around the time of will execution.
> Source: whitmore_estate_78076a.md

**Result: ✅ Correct** — Plan tracking worked: `[ ] 1` → search → `[x] 1, [ ] 2` → search → `[x] 2` → Final Answer. Notably, this question directly tests the Q09 logic error from the ChatService baseline ("must prove both" vs "either is sufficient") — the agent answered correctly: undue influence alone is sufficient to void the 2022 Will.

---

### Q (3-part plan tracking test) — llama-3.3-70b-versatile vs meta-llama/llama-4-scout-17b-16e-instruct

> Run date: 2026-06-16
> Script: manual eval script (not included in repo)
> Endpoint: `/api/v1/agent/stream`

**Question:** What was Harold's MoCA score progression from 2021 to 2022, what specific control did Kowalski establish over Harold's external communications after July 2022, and what was the exact deficiency in attorney Finch's capacity evaluation?

#### llama-3.3-70b-versatile

> Iterations: 3 thoughts, 3 actions, 3 observations

**Agent trace:**
- Thought 1: listed `[ ] 1. MoCA [ ] 2. Kowalski communications [ ] 3. Finch deficiency`. Searched item 1.
- Observation 1: MoCA scores found. → `[x] 1`, `[ ] 2`, `[ ] 3`. Searched item 2.
- Observation 2: Kowalski control found, Finch info partially visible. → `[x] 2`, kept `[ ] 3`, ran 3rd search "to ensure all aspects are covered".
- Observation 3: rate limit hit (TPM exceeded on 3rd search). Final answer: empty.

**Result: ❌ Rate limit** — Plan tracking correct (3 separate searches for 3 items), but 3rd search hit TPM limit and produced an empty final answer.

---

#### meta-llama/llama-4-scout-17b-16e-instruct

> Iterations: 3 thoughts, 2 actions, 2 observations

**Agent trace:**
- Thought 1: listed `[ ] 1. MoCA [ ] 2. Kowalski communications [ ] 3. Finch deficiency`. Searched item 1.
- Observation 1: MoCA + partial Finch info returned. → `[x] 1`, `[ ] 2` (partial), `[ ] 3` (partial). Searched item 2.
- Observation 2: Kowalski control found. → `[x] 2`, `[ ] 3` still open. Wrote "Searching for item 3" but recognised item 3 was already in prior observations — skipped the 3rd search and wrote Final Answer directly.

**Model answer:**

> Harold's MoCA score progression: March 2021 — 21/30 (mild cognitive impairment); January 2022 — 16/30 (moderate dementia). Kowalski screened phone calls, managed correspondence, and accompanied Decedent to all medical appointments after July 2022. Finch did not request or review any medical records, did not consult with Dr. Alvarez, and did not conduct or arrange for any independent cognitive assessment — his notes only reflected that Decedent "appeared oriented and expressed clear wishes."
> Source: whitmore_estate_78076a.md

**Result: ✅ Correct** — All 3 items answered accurately. Scout recognised item 3 was already covered by prior observations and skipped the redundant 3rd search, avoiding the rate limit that blocked llama-3.3-70b.

| | llama-3.3-70b | Scout |
|---|---|---|
| Searches | 3 | 2 |
| Checkbox rule | Strict — searched for every `[ ]` | Pragmatic — skipped redundant search |
| Outcome | Empty (rate limit) | Complete and correct |

---

### Aynı soru — Chat Stream vs Agent karşılaştırması

**Question:** If Petitioners succeed on the undue influence claim but fail on the testamentary capacity claim, what is the legal outcome for the 2022 Will — and what evidence would have been sufficient to prove capacity at the time of signing?

| | Chat Stream (gpt-oss-120b) | Agent (Scout) | Agent (llama-3.3-70b) |
|---|---|---|---|
| Searches | 1 (single pass) | 3 | 2 |
| Correctness | ✅ | ✅ | ✅ |
| Depth | ⭐⭐⭐ Most detailed | ⭐⭐ | ⭐⭐ |
| Plan tracking | N/A | ❌ `[ ]` never updated | ✅ Correct |

**Chat Stream answer (gpt-oss-120b):** Voided the 2022 Will, admitted 2018 Will. Listed the four-part Calloway test in full, cited lucid interval doctrine, identified Finch's note as insufficient, described what robust capacity evidence would look like. Most structured and detailed of the three.

**Conclusion:** This question is answered entirely from a single document section — chat stream retrieved the right passage in one pass and produced the richest answer. Agent added no value here; it spent extra iterations on a question that didn't require multi-hop retrieval. Agent's advantage appears on questions that need information from genuinely separate document sections.

---

### Q (3-section retrieval test) — Chat Stream vs Agent karşılaştırması

**Question:** Harold signed the 2022 Will on September 3, 2022 — at that moment, what was his documented medical condition according to his physician, who had control over his access to the outside world, and what would a court need to find in the drafting attorney's notes to conclude the signing was legally valid?

#### Agent — Scout (meta-llama/llama-4-scout-17b-16e-instruct)

> Iterations: 2 thoughts, 1 action, 1 observation

- Checklist: only 2 items listed — missed "who had control over access" entirely
- 1 search, answered items 1 and 3 correctly, Kowalski section absent from answer

**Result: ❌ Incomplete** — Scout mis-parsed the question, dropped one of the three parts from the checklist and therefore never searched for it.

---

#### Agent — gpt-oss-120b (openai/gpt-oss-120b)

> First attempt: empty output — caused by a transient error elsewhere in the pipeline, not a model compatibility issue.
> Second attempt: 2 thoughts, 1 action, 1 observation

- Checklist: correctly listed all 3 items
- 1 search returned chunks covering all three sections
- All 3 parts answered accurately and in detail

**Result: ✅ Correct** — gpt-oss-120b works on the agent endpoint with the text-based ReAct format. It also produces strong results on chat stream (native function-calling). Best model tested across both endpoints.

---

#### Chat Stream — gpt-oss-120b

> Single pass, no iterations

- Retrieved all 3 sections in one query (top_k=5 returned chunks from medical, Kowalski, and Finch sections)
- Most detailed answer of the three runs: cited Dr. Alvarez August 2022 notes, Kowalski's specific controls post-July 2022, full 4-part Calloway test, lucid interval doctrine

**Result: ✅ Correct and most detailed** — Single retrieval sufficient because all three answer sections had high cosine similarity to the combined query.

---

#### Cross-run summary

| | Scout (agent) | gpt-oss-120b (agent) | gpt-oss-120b (chat stream) |
|---|---|---|---|
| Checklist | ❌ 2/3 items | ✅ 3/3 items | N/A |
| Searches | 1 | 1 | 1 |
| Answer completeness | ❌ Missing Kowalski | ✅ All 3 parts | ✅ All 3 parts, most detail |

**Conclusion:** For this document, pgvector retrieves chunks from multiple sections in a single query — chat stream and agent converge to the same number of searches. Agent's multi-hop advantage appears when the first retrieval is insufficient and query reformulation is needed, not when all sections surface in one pass.

---

## Observations

**Iteration count scales with question complexity.**

- **Q01** (factual lookup — age and death date): 1 action, resolved in a single retrieval. The answer was a discrete fact present in the document header; no reformulation needed.
- **Q04** (multi-part legal analysis — cognitive condition + testamentary capacity elements): 2 actions, 3 thoughts. The model correctly recognised after the first retrieval that it needed more evidence and issued a second search with a reformulated query. The second query returned lower-relevance chunks (rerank-score 0.187 vs 0.540) and did not contribute new information, but the agent's instinct to verify before answering is the right behaviour for complex questions.

Simple factual questions converge in 1 iteration. Questions that require synthesising multiple pieces of evidence — or mapping retrieved facts to a legal framework — naturally trigger additional retrieval rounds. The agent loop budget (5 iterations) leaves headroom for harder multi-hop questions without over-searching on easy ones.

---

## ⭐ Open Problems

### ⭐ gpt-oss-120b agent endpoint'te soru tipine bağlı native tool-call hatası

**Hata:**
```
Tool choice is none, but model called a tool
status=400, code=tool_use_failed
```

**Ne oluyor:** Agent `tools=None` ile çalışıyor — LLM'den native function-call değil, düz metin `Action: rag_search` formatı bekleniyor. gpt-oss-120b spesifik, factual sorularda text formatını takip ediyor. Soyut veya dolaylı sorularda (`"What independent verification steps were skipped..."`) native tool-call üretiyor, API reddediyor.

**Tutarsız davranış:** Aynı model aynı endpoint'te bazı sorularda çalışıyor, bazılarında kırılıyor — önceden tahmin edilemiyor.

**Çözüm seçenekleri:**
1. Prompt'a explicit yasak ekle: `"Do NOT use built-in function-calling, write Action: as plain text"` — kısmen işe yarayabilir, garanti değil
2. Agent'a da `tools=` ver, native function-calling'e geç — güvenilir ama `Thought:` stream kaybedilir
3. Agent endpoint'ini sadece llama modelleriyle kısıtla
