# Sherlock RAG Retrieval Test — Report

## 1. What was tested

- **Document:** Sherlock Holmes PDF containing two stories — *A Scandal in Bohemia* and *The Red-Headed League* — split into 104 chunks by the ingestion pipeline.
- **Questions:** 16 hand-written questions across four difficulty categories:
  - **Easy (4):** single-chunk factual lookups (e.g. "What is Irene Adler's address?")
  - **Medium (4):** multi-chunk questions within one story (e.g. the smoke-rocket trick)
  - **Hard (4):** cross-story synthesis requiring chunks from both stories
  - **Trap (4):** questions about content that is absent, truncated, or belongs to a different story
- **Pipeline config:** mxbai-embed-large embeddings, hybrid search (dense + BM25 + grep), Jina reranker, RRF fusion (k=60), fetch_k=15, top_k=5.
- **Runs:** 4 runs total (run_001 through run_004). run_003 is the full 16-question baseline. run_004 tested `websearch_to_tsquery` as a BM25 improvement — no change observed.

## 2. What was found

### Overall metrics (run_003 / run_004 — identical)

| Metric | Value |
|---|---|
| recall@5 | 0.667 (10/15 questions) |
| MRR (pre-rerank) | 0.339 |
| MRR (post-rerank) | 0.444 |
| avg correct cosine | 0.6687 |
| avg chunk recall | 0.528 |

### Per-leg retrieval recall

| Leg | Recall | Notes |
|---|---|---|
| Dense (cosine) | 11/15 | Main workhorse — found the correct chunk in 73% of questions |
| Sparse (BM25) | 1/15 | Nearly dead — only matched Q04 (proper nouns "New Jersey", "1858") |
| Grep (ILIKE) | 6/15 | Outperformed BM25 by 6x — substring matching more robust for this dataset |

### Per-category breakdown

| Category | Found/Total | Notes |
|---|---|---|
| Easy | 2/4 | Q02 ("seventeen steps") and Q03 ("Briony Lodge") missed — specific details not retrieved |
| Medium | 4/4 | All found, but chunk recall ranged from 0.33 to 1.0 |
| Hard | 1/4 | Only Q09 found. Q10, Q11, Q12 scored 0.0 — cross-story retrieval completely failed |
| Trap | 3/4 | Q13, Q14, Q16 found relevant chunks. Q15 (Moriarty — absent) correctly returned nothing |

### Reranker impact

- Improved rank in 4 questions (Q01, Q07, Q08, Q16)
- No improvement or worsened rank in 4 questions
- Net effect: marginal — MRR improved from 0.339 to 0.444

### Key observations

- **All chunks have `section_heading: null`** — PDF heading detection did not produce any section metadata. Chunks carry no information about which story they belong to, making cross-story disambiguation impossible at the embedding level.
- **BM25 is structurally weak for this dataset** — 19th-century literary English does not lexically overlap with modern natural-language questions. The PostgreSQL stemmer cannot bridge "residence" ↔ "address" or "steps" ↔ "seventeen". Switching from `plainto_tsquery` to `websearch_to_tsquery` (run_004) made zero difference.
- **Hard/cross-story questions fail systematically** — when a question references both stories, the embedding pulls toward one story's chunks and misses the other entirely. This is not a retrieval parameter problem; it requires query decomposition or multi-query retrieval.
- **Grep is the most reliable non-dense leg** — ILIKE substring matching found correct chunks in 6/15 questions, compensating for BM25's weakness on this text.

## 3. Why this test series was concluded

- **Sparse search cannot be fixed for this dataset** — the lexical mismatch between 19th-century prose and modern queries is structural, not a query-parsing issue.
- **Cross-story questions need query decomposition** — splitting "In both stories, how does Holmes use deception?" into two sub-queries per story. This is an architectural change beyond retrieval tuning.
- **Section heading enrichment requires re-ingestion** — adding story names to chunks means modifying the chunking pipeline and re-embedding all 104 chunks. More efficient to validate this approach on a new, better-structured document.
- **The baseline is established** — recall@5=0.667, MRR=0.339 gives a clear reference point for measuring future improvements.

## 4. What was learned

- **Dataset choice directly affects RAG performance** — literary 19th-century text with no structural headings is a worst case for hybrid search. Technical documents with clear sections, keywords, and modern language would give a fairer evaluation of the pipeline.
- **Evaluation metrics in practice** — recall@5, MRR, and per-leg recall breakdowns provide actionable diagnostics. MRR in particular reveals whether the correct chunk is ranked first vs. buried at position 5.
- **Dense vs. sparse in practice** — dense search dominated (11/15 vs 1/15). BM25 is only useful when queries and documents share exact vocabulary. For semantic paraphrase queries, dense is the only reliable leg.
- **Reranker value depends on candidate quality** — if the correct chunk is not in the candidate pool (fetch_k=15), reranking cannot help. The reranker improved ranking in 4/10 found questions, which is meaningful but not transformative.
- **Grep as a pragmatic fallback** — simple substring matching (ILIKE) outperformed the full BM25 pipeline on this dataset. For documents with distinctive proper nouns or identifiers, grep provides high-precision retrieval at low cost.

## 5. Post-baseline: OR-based sparse fix (run_005 through run_008)

After the baseline was established, `plainto_tsquery` (AND logic) was replaced with OR-based `_build_or_tsquery` + `to_tsquery` in `app/db/repositories/document.py`. Selected questions were re-run to verify the fix on this dataset.

- **run_005 (Q01):** sparse now finds chunk 2. Previously sparse missed it (AND required all terms). No change to overall result — dense already had it.
- **run_006 (Q02):** **Previously found=False, now found=True.** Dense could not find chunk 8 ("seventeen steps") in top-15. Sparse OR matched it and brought it into the RRF candidate pool at rank 9. Reranker promoted it to rank 1. This is the strongest evidence of the sparse OR + reranker combination rescuing a chunk that was completely invisible to dense search.
- **run_007 (Q03):** Still found=False. "Briony Lodge, Serpentine Avenue" is too specific — no leg can match it. This is a dataset limitation, not a sparse issue.
- **run_008 (Q05-Q08):** Sparse recall improved from 0/4 to 4/4. All four were already found=True via dense, so overall recall unchanged. Confirms OR fix works across medium-difficulty questions on literary text.

The original conclusion that "sparse search cannot be fixed for this dataset" (Section 3) was too strong. The OR fix substantially improved sparse recall. The lexical mismatch between 19th-century prose and modern queries is real, but OR-based matching bridges it better than AND ever could.
