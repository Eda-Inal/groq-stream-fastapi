# Multidoc Source Test — Run 1

**Date:** 2026-06-04  
**Model:** `openai/gpt-oss-120b` (Groq)  
**Script:** `eval/run_multidoc_source_test.py`  
**Documents:** `whitmore_estate.txt` (legal) + `brt447_trial.txt` (academic)  
**Mode:** TEST MODE active — conversation_id filter off, user_id scoping only

---

## Overall: 4/7 PASS

| Q | Type | Source correct? | Content correct? | Result |
|---|------|----------------|-----------------|--------|
| Q1 | single_source | whitmore_estate.txt ✅ | ✅ | **PASS** |
| Q2 | single_source | brt447_trial.txt ✅ | ✅ | **PASS** |
| Q3 | cross_document | both ✅ | EORTC QLQ-C30 missing ❌ | **PARTIAL** |
| Q4 | cross_document | both ✅ | Bridgeton consultancy fees missing ❌ | **FAIL** |
| Q5 | cross_document | both ✅ | ECOG/vulnerability missing from brt447 ❌ | **FAIL** |
| Q6 | web_fallback | URL ✅ | ✅ | **PASS** |
| Q7 | web_fallback | URLs ✅ | ✅ | **PASS** |

---

## Q1 — Single source (Document 1 — legal)

**Question:** What was Harold Whitmore's MoCA score in January 2022, and what did it indicate about his cognitive condition?

**Answer:**
> Harold Whitmore's MoCA score in the January 2022 follow-up evaluation was **16 out of 30**. The record notes that this score was "consistent with moderate dementia per DSM-5 criteria."
>
> Source: whitmore_estate.txt

**Verdict:** PASS — correct source, correct content, no bleed from brt447.

---

## Q2 — Single source (Document 2 — academic)

**Question:** What percentage of BRT-447 patients experienced Grade 3–4 adverse events, and what were the most common ones?

**Answer:**
> 18.3% of patients receiving BRT-447 experienced Grade 3–4 adverse events. The most common Grade 3–4 events in the BRT-447 arm were fatigue (6.3%), elevated liver enzymes (4.9%), and infusion-related reactions (3.5%).
>
> Source: brt447_trial.txt

**Verdict:** PASS — correct source, all statistics accurate, no bleed from whitmore.

---

## Q3-A — Cross-document (cognitive measurement tools) — original question

**Question:** One document discusses a person's mental deterioration, and another discusses cognitive assessment tools used in clinical trials. What do both documents reveal about how cognitive decline is measured?

**Answer:**
> The estate file shows cognitive decline tracked with the Montreal Cognitive Assessment (MoCA), reporting scores of 21/30 (mild impairment) in March 2021 and a drop to 16/30 (moderate dementia) by January 2022. The clinical-trial file does not mention any cognitive-assessment instruments; it reports outcomes such as overall survival, progression-free survival, quality-of-life scores, and safety data, but provides no details on how cognitive decline was measured in the study.
>
> Source: whitmore_estate.txt, brt447_trial.txt

**Verdict:** PARTIAL — both sources cited. Model says brt447 has "no cognitive-assessment instruments."

**Finding (confirmed on retest):** This is a **question design issue, not a retrieval failure.** EORTC QLQ-C30 is a *quality-of-life* instrument, not a *cognitive assessment* tool. The model's answer is technically correct — brt447 does not measure cognitive decline specifically. The question vocabulary ("cognitive decline", "cognitive assessment") does not semantically map to EORTC ("quality of life", "health status scores").

**Conclusion:** Question needs rephrasing to surface EORTC. See Q3-B below.

---

## Q3-B — Cross-document (patient condition measurement) — corrected question

**Question:** Both documents involve measuring or tracking patient condition over time. What standardized tools or scoring systems does each document use to assess how a person is doing, and what do they measure?

**Answer:**
> **Whitmore Estate document**
> - **Montreal Cognitive Assessment (MoCA)** — decedent scored 21/30 (mild cognitive impairment) and later 16/30 (moderate dementia).
>
> **BRT-447 trial document**
> - **EORTC QLQ-C30** — global health status scores; BRT-447 arm improved 14.2 points vs −3.1 points in control.
> - **Overall Survival (OS)** — 41.3% vs 22.7% alive at 18 months.
> - **Progression-Free Survival (PFS)** — 7.4 months vs 3.9 months.
> - **Grade 3-4 adverse-event rates** — safety endpoint.
>
> Source: whitmore_estate.txt; brt447_trial.txt

**Verdict:** PASS — both sources cited, MoCA and EORTC QLQ-C30 both surfaced correctly.

**Lesson:** Rephrasing from "cognitive decline measurement" → "standardized tools or scoring systems to assess how a person is doing" was enough to retrieve the EORTC chunk. Abstract/general vocabulary surfaces domain-specific content better than narrow clinical vocabulary.

---

## Q4-A — Cross-document (financial motivation) — original question

**Question:** Compare the role of financial motivation in both documents — how does money influence the behavior of key individuals in each case?

**Answer:**
> Whitmore: $4.2M estate, $28/hour — financial motive central to undue influence claim.
> BRT-447: "no reference to money, financial incentives, or any financial motivation."
>
> Source: whitmore_estate.txt, brt447_trial.txt

**Verdict:** FAIL — Conflicts of Interest section (consultancy fees) not retrieved. Query vocabulary "financial motivation" did not match document vocabulary "consultancy fees."

---

## Q4-B — Cross-document (financial relationships) — corrected question

**Question:** What financial relationships or conflicts of interest exist between key parties in each document, and how might these affect the outcomes described?

**Answer:**
> **Whitmore estate:** Diane Kowalski paid $28/hour, stands to receive $4.2M estate — creates motive for undue influence, supports petition to invalidate 2022 Will.
>
> **BRT-447 trial:** Funded by Bridgeton Therapeutics + NCI. Dr. Nandakumar and Dr. Holweg disclosed consultancy fees from Bridgeton Therapeutics; Dr. Forsythe reported no conflicts. Could introduce bias favoring BRT-447.
>
> Source: whitmore_estate.txt; brt447_trial.txt

**Verdict:** PASS — both sources cited, consultancy fees surfaced correctly from brt447.

**Lesson:** "Financial motivation" → "financial relationships or conflicts of interest" was enough to retrieve the Conflicts of Interest chunk.

---

## Q5-A — Cross-document (vulnerability) — original question

**Question:** Both documents involve vulnerable individuals. How is vulnerability defined or described differently in each document?

**Answer:**
> Whitmore: progressive dementia, total dependence on caregiver.
> BRT-447: only "4. DISCUSSION" heading retrieved — no vulnerability content found.
>
> Source: whitmore_estate.txt, brt447_trial.txt

**Verdict:** FAIL — ECOG / vulnerability chunks from brt447 not retrieved. Word "vulnerable" not in document; concept expressed as "ECOG performance status" and "frailer populations."

---

## Q5-B — Cross-document (physical/medical condition) — corrected question

**Question:** How does each document describe the physical or medical condition of the individuals involved, and what limitations or dependencies does this create for them?

**Answer:**
> **Whitmore estate:** Progressive dementia (MoCA 21/30 → 16/30), wholly dependent on caregiver for daily activities — creates susceptibility to undue influence.
>
> **BRT-447 trial:** Stage III-IV pancreatic adenocarcinoma, ECOG performance status 0-2. Excludes frailer patients (poorer ECOG), limiting generalizability. Outcomes depend on participants' ability to tolerate treatment per ECOG scores.
>
> Source: whitmore_estate.txt; brt447_trial.txt

**Verdict:** PASS — both sources cited, MoCA + ECOG both surfaced correctly.

**Lesson:** "Vulnerability" → "physical or medical condition and limitations/dependencies" retrieved ECOG and dementia context from both documents.

---

## Q6 — Web fallback (FDA approval status)

**Question:** internetten ara: What is the current FDA approval status of mRNA-based therapeutics for pancreatic cancer as of 2024?

**Answer:**
> As of 2024, the FDA has not yet approved any mRNA-based therapeutics for pancreatic cancer. First commercial approvals of RNA-based cancer vaccines anticipated around 2029.
>
> Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12153701

**Verdict:** PASS — web_search triggered correctly despite documents being present ("internetten ara" worked). Accurate, sourced answer.

---

## Q7 — Web fallback (US legal standards for contesting a will)

**Question:** internetten ara: What are the general legal standards for contesting a will on grounds of undue influence in the United States today?

**Answer:**
> Comprehensive 5-element standard: burden of proof, confidential relationship, testator susceptibility, opportunity and motive, resulting disposition. Notes state variation.
>
> Source: nelsonmullins.com, rutkinwolf.com, weightmans.com

**Verdict:** PASS — web_search triggered correctly. Multiple authoritative sources cited. Did not overgeneralize from Harwick fictional jurisdiction in whitmore_estate.txt.

---

## Failure Analysis

All 3 failures (Q3 partial, Q4, Q5) share the same root cause: **RAG retrieval not surfacing the right chunks from brt447_trial.txt.**

| Failed content | Location in document | Likely reason not retrieved |
|---------------|---------------------|---------------------------|
| EORTC QLQ-C30 | Section 3.3 Quality of Life | Query about "cognitive measurement" → low semantic overlap with "quality of life" |
| Conflicts of Interest (consultancy fees) | End of document, after Conclusion | Query about "financial motivation" → document talks about "consultancy fees" not "financial motivation" |
| ECOG / vulnerability | Methods (2.2) + Discussion | Query about "vulnerability" → document uses "ECOG performance status" and "frailer populations", not "vulnerable" |

**Pattern:** The vocabulary in the query does not match the vocabulary in the relevant chunks. The model searches for "financial motivation" but the document says "consultancy fees." It searches for "vulnerability" but the document says "ECOG performance status."

---

## Rerank Data (Q3, Q4, Q5)

### Q3 — 5 chunks retrieved, top rerank score: 0.190

| Source | Rerank score | Content preview |
|--------|-------------|-----------------|
| whitmore_estate.txt | 0.190 | EVIDENCE OF LACK OF TESTAMENTARY CAPACITY — MoCA scores |
| whitmore_estate.txt | 0.077 | Mr. Finch section |
| brt447_trial.txt | 0.061 | **"4. DISCUSSION"** — heading only, no content |

EORTC QLQ-C30 chunk (Section 3.3) never appeared — not retrieved in the initial stage.

### Q4 — 8 chunks retrieved, top rerank score: 0.104

| Source | Rerank score | Content preview |
|--------|-------------|-----------------|
| brt447_trial.txt | 0.104 | **"4. DISCUSSION"** — heading only, no content |
| whitmore_estate.txt | 0.049 | Testamentary capacity / undue influence intro |
| whitmore_estate.txt | 0.043 | Mr. Finch section |

Conflicts of Interest section (consultancy fees) never appeared — not retrieved in the initial stage.

### Q5 — 5 chunks retrieved, top rerank score: 0.067

| Source | Rerank score | Content preview |
|--------|-------------|-----------------|
| whitmore_estate.txt | 0.067 | Mr. Finch section |
| whitmore_estate.txt | 0.061 | Kowalski opportunity section |
| whitmore_estate.txt | 0.057 | Burden of proof section |

No brt447 chunks at all. ECOG / vulnerability section was not retrieved even in the initial dense retrieval stage.

---

## Root Cause (confirmed from rerank data)

**Problem 1 — Noise chunk:** `"4. DISCUSSION"` is stored as a standalone chunk with no content — just a section heading. It appears in Q3 and Q4 as the only brt447 representative but carries no useful information. Score: 0.061–0.104.

**Problem 2 — Dense retrieval miss:** The relevant chunks never reach the reranker. Reranker can only reorder what the initial retrieval returns.

| Missing content | Why not retrieved |
|----------------|-------------------|
| EORTC QLQ-C30 (Q3) | Query: "cognitive measurement" → chunk: "quality of life scores" — semantic gap |
| Consultancy fees (Q4) | Query: "financial motivation" → chunk: "consultancy fees" — vocabulary mismatch |
| ECOG / vulnerability (Q5) | Query: "vulnerability" → chunk: "ECOG performance status", "frailer populations" — vocabulary mismatch |

**Pattern:** All failures share the same structure — the query uses abstract/general vocabulary while the document uses domain-specific terminology. Dense embeddings alone do not bridge this gap, and BM25 sparse search also fails because the exact words don't overlap.

---

## Corrected Questions — Retest Results

All 3 failing questions were retested with rephrased queries. Documents uploaded once, shared across Q4-B and Q5-B via TEST MODE (user_id scoping).

| Original | Corrected | Key change | Result |
|----------|-----------|-----------|--------|
| "cognitive decline...cognitive assessment tools" | "standardized tools or scoring systems to assess how a person is doing" | Removed clinical specificity, used neutral framing | FAIL → **PASS** |
| "financial motivation" | "financial relationships or conflicts of interest" | Used exact document vocabulary | FAIL → **PASS** |
| "vulnerability...defined or described" | "physical or medical condition...limitations or dependencies" | Replaced abstract concept with concrete descriptors | FAIL → **PASS** |

---

## Key Finding — Query Vocabulary Determines Retrieval

**The single most impactful factor in RAG retrieval is whether the query vocabulary matches the document vocabulary.**

| Query word | Document word | Retrieved? |
|-----------|--------------|-----------|
| "financial motivation" | "consultancy fees" | ❌ No |
| "conflicts of interest" | "consultancy fees" | ✅ Yes |
| "vulnerability" | "ECOG performance status" | ❌ No |
| "physical or medical condition" | "ECOG performance status" | ✅ Yes |
| "cognitive assessment tools" | "EORTC QLQ-C30" | ❌ No |
| "standardized tools or scoring systems" | "EORTC QLQ-C30" | ✅ Yes |

**Rule:** Abstract/interpretive query vocabulary fails. Concrete/structural vocabulary that is closer to how the document actually phrases things succeeds.

This has implications for how users phrase questions in production — and for whether query rewriting (having the model rephrase the user's question before embedding) would improve retrieval.

---

## Next Steps

- Fix the noise chunk issue: `"4. DISCUSSION"` heading-only chunks should be merged with the section body or dropped by `_drop_tiny_chunks`
- Investigate why ECOG / EORTC / consultancy chunks are not surfacing in dense retrieval
- Test with hybrid search (BM25 may help with exact term matching for "ECOG", "consultancy")
- Consider query rewriting before embedding: model rephrases user question into document-vocabulary-aligned terms before RAG call
