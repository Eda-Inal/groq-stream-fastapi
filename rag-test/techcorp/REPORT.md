# TechCorp RAG Retrieval Test — Report

## 1. What was tested

- **Document:** `techcorp_company_policies.md` — a structured company handbook with 9 top-level sections covering code of conduct, work hours, time off, compensation, security, expenses, and offboarding.
- **Ingestion:** Uploaded as plain text via `scripts/upload_document.py` with `user_id: techcorp-test`. Produced **24 chunks** (3,316 tokens) with `mxbai-embed-large` embeddings (768-dim).
- **Chunking:** Markdown headings (`#`, `##`, `###`) were detected and used for splitting. Each section landed in its own chunk with `section_heading` metadata populated (e.g. "3.2 Remote Work Policy", "5.2 Equity"). Context prefix: `[Belge: techcorp_company_policies.md]` (no page numbers — not a PDF).
- **Questions:** 10 hand-written questions across four categories:
  - **exact_match (3):** answer contains a specific value verbatim in the chunk (VPN, 30 days, two weeks)
  - **named_entity (2):** answer is a named entity — email address, dollar amount
  - **direct_fact (3):** answer is a factual statement with numbers (four-year vesting, three days bereavement, $2,000 budget)
  - **dense_primary (2):** answer requires semantic understanding — question phrasing differs from chunk wording
- **Difficulty level:** Easy. All questions are single-chunk, single-hop lookups against a well-structured document. No multi-chunk reasoning, no cross-section synthesis, no trap questions. This test establishes a retrieval baseline, not a stress test.
- **Pipeline config:** mxbai-embed-large embeddings, hybrid search (dense + BM25 + grep), Jina reranker, RRF fusion (k=60), fetch_k=15 (overfetch 3x for reranker), top_k=5.
- **Runs:** run_003 is the baseline (AND-based `plainto_tsquery`). run_004 tests OR-based sparse search (`_build_or_tsquery` with `to_tsquery`).

## 2. Results (run_003)

### Overall metrics

| Metric | Value |
|---|---|
| recall@5 | **1.000** (10/10 questions) |
| MRR (pre-rerank) | **1.000** |
| MRR (post-rerank) | **1.000** |
| avg correct cosine | 0.7480 |
| avg chunk recall | 1.000 |

All 10 questions retrieved the correct chunk at rank 1, both before and after reranking.

### Per-leg retrieval recall

| Leg | Recall | Notes |
|---|---|---|
| Dense (cosine) | **10/10** | Found every question — sole workhorse |
| Sparse (BM25) | **1/10** | Only matched Q01 ("remote work ... internal systems") |
| Grep (ILIKE) | **8/10** | Missed Q03 (resignation notice) and Q04 (security email) |

### Per-question breakdown

| Q | Category | Chunk | Cosine | Rank (pre) | Rank (post) | Dense | Sparse | Grep |
|---|---|---|---|---|---|---|---|---|
| Q01 | exact_match | 6 | 0.706 | 1 | 1 | Y | Y | Y |
| Q02 | exact_match | 21 | 0.792 | 1 | 1 | Y | - | Y |
| Q03 | exact_match | 22 | 0.704 | 1 | 1 | Y | - | - |
| Q04 | named_entity | 18 | 0.659 | 1 | 1 | Y | - | - |
| Q05 | named_entity | 15 | 0.690 | 1 | 1 | Y | - | Y |
| Q06 | direct_fact | 13 | 0.851 | 1 | 1 | Y | - | Y |
| Q07 | direct_fact | 11 | 0.909 | 1 | 1 | Y | - | Y |
| Q08 | direct_fact | 15 | 0.595 | 1 | 1 | Y | - | Y |
| Q09 | dense_primary | 10 | 0.768 | 1 | 1 | Y | - | Y |
| Q10 | dense_primary | 13 | 0.805 | 1 | 1 | Y | - | Y |

### Reranker impact

Reranker improved 0 out of 10 questions. Since dense search already placed the correct chunk at rank 1 for every question, the reranker had nothing to improve. Reranker value cannot be assessed from this test — a harder question set is needed.

## 3. Results (run_004 — OR-based sparse fix)

### What changed

Replaced `plainto_tsquery` (AND logic) with `_build_or_tsquery` + `to_tsquery` (OR logic) in `app/db/repositories/document.py`. The function strips stop words (using the existing `_STOP_WORDS` set) and words shorter than 3 characters, then joins remaining terms with `|`. PostgreSQL's `to_tsquery('english', ...)` handles stemming. `ts_rank` ranks chunks by how many terms match.

### Overall metrics

| Metric | run_003 (AND) | run_004 (OR) |
|---|---|---|
| recall@5 | 1.000 | 1.000 |
| MRR (pre-rerank) | 1.000 | 1.000 |
| MRR (post-rerank) | 1.000 | 1.000 |
| avg correct cosine | 0.7480 | 0.7480 |
| **dense_recall** | 10/10 | 10/10 |
| **sparse_recall** | **1/10** | **10/10** |
| **grep_recall** | 8/10 | 8/10 |

### Per-question sparse comparison

| Q | Category | run_003 sparse | run_004 sparse |
|---|---|---|---|
| Q01 | exact_match | Y | Y |
| Q02 | exact_match | - | **Y** |
| Q03 | exact_match | - | **Y** |
| Q04 | named_entity | - | **Y** |
| Q05 | named_entity | - | **Y** |
| Q06 | direct_fact | - | **Y** |
| Q07 | direct_fact | - | **Y** |
| Q08 | direct_fact | - | **Y** |
| Q09 | dense_primary | - | **Y** |
| Q10 | dense_primary | - | **Y** |

### Trade-off analysis

OR-based matching is broader — a typical query matches ~20 of 24 chunks (vs 0-1 with AND). This is acceptable because:

1. `ts_rank` correctly ranks the chunk with the most term matches at position 1 (10/10 correct).
2. RRF is rank-based: a noisy sparse leg at rank 1 contributes the same `1/(k+1)` as a precise one.
3. Corpus-wide high-frequency words (`techcorp` in 22/24 chunks, `employe` in 18/24) add some noise but do not displace the correct chunk from the top rank.

## 4. Analysis

### What worked

- **Dense search carried everything.** Cosine similarity ranged from 0.595 (Q08) to 0.909 (Q07), and even the weakest score was enough for rank 1. The structured chunking (one section per chunk, heading prepended to text) gives each chunk a clear semantic identity.
- **Sparse is now alive.** The OR fix brought sparse from 1/10 to 10/10. All three legs (dense + sparse + grep) now contribute to RRF fusion, which will matter on harder question sets where dense alone may not rank the correct chunk first.
- **Grep pulled its weight.** 8/10 recall, catching keyword-heavy questions that dense also found. It provides useful redundancy through RRF fusion.
- **Chunking quality is high.** Markdown heading detection produced clean, self-contained chunks. Every chunk has a meaningful `section_heading` and the heading text is part of the chunk body, contributing to both embedding quality and BM25/grep matching.

### What failed in run_003 (fixed in run_004)

- **Sparse (BM25) was nearly dead at 1/10.** Root cause: `plainto_tsquery` uses AND logic — every stemmed term in the query must appear in the chunk. For Q02 ("Within how many days must business expenses be submitted through the expense management system?"), the tsquery required 10 terms to ALL match. Chunk 21 contained 9 of 10 terms — only `many` was missing — but `@@` returned false. A single missing word killed the entire BM25 leg.
  - `websearch_to_tsquery` was also evaluated but produces the same AND query for plain sentences — it only uses OR when the literal word "or" appears in the input.
  - The OR-based `_build_or_tsquery` fix resolved this completely.

### Limitations of this test

- **Easy difficulty only.** All questions target a single chunk with clear factual answers. The document is well-structured with distinct sections — there is minimal topic overlap between chunks.
- **No multi-hop or cross-section questions.** A harder test would ask questions requiring information from two or more sections (e.g. "Can an employee on parental leave submit expenses after the 30-day window?").
- **No trap questions.** All questions have answers in the document. A real eval should include questions about topics NOT in the handbook to test retrieval precision.
- **Small corpus.** 24 chunks from a single document. Retrieval difficulty scales with corpus size — these results may not hold when the knowledge base contains hundreds of documents.

## 5. Next steps

1. **Re-run Sherlock test** with the OR-based sparse fix to compare improvement on a harder dataset (run_003 baseline sparse: 1/15).
2. **Add harder questions** — multi-hop, cross-section, and trap categories.
3. **Evaluate on larger corpus** — multiple documents with overlapping topics to stress-test RRF fusion precision.
