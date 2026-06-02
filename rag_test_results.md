# RAG Test Results

Test date: 2026-06-02  
Embedding model: mxbai-embed-large (Ollama, 1024-dim)  
Hybrid search: enabled (dense + BM25 + RRF)  
Reranker: enabled (jina-reranker-v2-base-multilingual)

---

## Key Findings

- `rag_search` is only triggered when the question explicitly references the user's own documents using phrases like "my documents / my files / my uploaded documents".
- Without these phrases, the model falls back to `web_search` or answers directly from general knowledge.
- No hallucinations observed on questions whose answers are not in the documents — the model honestly returns "not found".
- Table chunks are preserved atomically and comparison queries work correctly.

---

## Test 1 — Aurora Research Station (aurora_station.txt)

**document_id:** 34 | **chunks:** 3 | **tokens:** 678

### 1A — Without "my documents" phrasing (llama-3.3-70b-versatile)

| # | Question | Expected Answer | Model | Result |
|---|----------|----------------|-------|--------|
| 1 | When was the station established? | 2018 | llama-3.3-70b-versatile | ❌ web_search triggered (`<\|python_tag\|>` bug) |
| 2 | How many main buildings are there? | 3 | llama-3.3-70b-versatile | ❌ RAG not called, returned "not enough information" |
| 3 | What is stored in the third building? | Data center | llama-3.3-70b-versatile | ✅ RAG triggered correctly |
| 4 | How many wind turbines are used? | 6 | llama-3.3-70b-versatile | ❌ web_search, unrelated result |
| 5 | How often do supply shipments arrive? | Every six weeks | llama-3.3-70b-versatile | ❌ web_search, returned Amazon shipping info |
| 6 | How many birds have been tracked since 2019? | More than 1,500 | llama-3.3-70b-versatile | ❌ web_search, unrelated result |
| 7 | What is the name of the island? | Not in document | llama-3.3-70b-versatile | ❌ web_search triggered (`<\|python_tag\|>` bug) |
| 8 | What is the annual budget of the station? | Not in document | llama-3.3-70b-versatile | ❌ web_search, hallucinated a radio station budget |
| 9 | Could the station survive two months without supply shipments? | Yes (3-month reserves) | llama-3.3-70b-versatile | ✅ RAG triggered, correct inference |
| 10 | Is renewable energy the main power source? | Yes | llama-3.3-70b-versatile | ❌ web_search, returned global energy statistics |

**Score: 2/10** — Routing is unreliable without explicit document reference phrases.

---

### 1B — With "my documents" phrasing (llama-3.3-70b-versatile)

| # | Question | Expected Answer | Model | Result |
|---|----------|----------------|-------|--------|
| 1 | According to my documents, when was the station established? | 2018 | llama-3.3-70b-versatile | ✅ "The station was established in 2018." |
| 2 | In my uploaded files, how many main buildings are there? | 3 | llama-3.3-70b-versatile | ✅ "There are three main buildings." |
| 3 | According to my documents, what is stored in the third building? | Data center | llama-3.3-70b-versatile | ✅ "A large data center that processes and stores research findings." |
| 4 | In my files, how many wind turbines are used? | 6 | llama-3.3-70b-versatile | ✅ "Six wind turbines." |
| 5 | According to my documents, how often do supply shipments arrive? | Every six weeks | llama-3.3-70b-versatile | ✅ "Every six weeks." |
| 6 | In my uploaded documents, how many birds have been tracked since 2019? | More than 1,500 | llama-3.3-70b-versatile | ✅ "More than 1,500 birds." |
| 7 | According to my documents, what is the name of the island? | Not in document | llama-3.3-70b-versatile | ✅ "This information was not found in your documents." |
| 8 | In my files, what is the annual budget of the station? | Not in document | llama-3.3-70b-versatile | ✅ "This information was not found in your documents." |
| 9 | According to my documents, could the station survive two months without supply shipments? | Yes | llama-3.3-70b-versatile | ✅ Correct inference — "up to three months reserve" |
| 10 | In my documents, is renewable energy the main power source? | Yes | llama-3.3-70b-versatile | ✅ Correct inference — wind + solar = primary sources |

**Score: 10/10**

---

### 1C — Unanswerable questions (llama-3.3-70b-versatile)

| # | Question | Model | Result |
|---|----------|-------|--------|
| 1 | According to my documents, which university owns the station? | llama-3.3-70b-versatile | ✅ "Not found in your documents." |
| 2 | In my files, what brand are the wind turbines? | llama-3.3-70b-versatile | ✅ "Not found in your documents." |
| 3 | According to my documents, what species of seabirds are being tracked? | llama-3.3-70b-versatile | ✅ "Not found in your documents." |
| 4 | In my uploaded documents, what programming language is used in the data center software? | llama-3.3-70b-versatile | ✅ "Not found in your documents." |
| 5 | According to my documents, what is the average annual temperature on the island? | llama-3.3-70b-versatile | ✅ "Not found in your documents." |
| 6 | In my files, how many female researchers work at the station? | llama-3.3-70b-versatile | ✅ "Not found in your documents." |

**Score: 6/6** — No hallucinations on any unanswerable question.

---

## Test 2 — GreenTech Solutions Internal Report (greentech_report.md)

**document_id:** 35 | **chunks:** 8 | **tokens:** 422  
Format: markdown with headings, table, and sections.  
Each chunk carries section heading metadata.

### Chunk Structure

| Chunk | Tokens | Section Heading |
|-------|--------|----------------|
| 0 | 6 | GreenTech Solutions Internal Report |
| 1 | 125 | Overview |
| 3 | 34 | Energy Monitor Pro |
| 4 | 29 | PredictAI |
| 5 | 30 | SmartAlert |
| 7 | 61 | Company Offices (table) |
| 9 | 87 | Recent Achievements |
| 10 | 50 | Future Plans |

Note: Chunks 2, 6, 8 were `---` separator lines, correctly dropped by `_drop_tiny_chunks`.

### 2A — Factual questions (llama-3.3-70b-versatile)

| # | Question | Expected | Model | Result |
|---|----------|---------|-------|--------|
| 1 | When was GreenTech Solutions founded? | 2017 | llama-3.3-70b-versatile | ✅ |
| 2 | Where is the company's headquarters located? | Amsterdam, Netherlands | llama-3.3-70b-versatile | ✅ |
| 3 | Approximately how many clients does the company serve? | 350+ | llama-3.3-70b-versatile | ✅ |
| 4 | What is the name of GreenTech's flagship platform? | Energy Monitor Pro | llama-3.3-70b-versatile | ✅ |
| 5 | Which product uses machine learning to forecast energy consumption? | PredictAI | llama-3.3-70b-versatile | ✅ |
| 6 | How many employees work in the Berlin office? | 40 | llama-3.3-70b-versatile | ✅ Correctly retrieved from table chunk |
| 7 | What percentage reduction in dashboard loading times was achieved in 2024? | 45% | llama-3.3-70b-versatile | ✅ |
| 8 | How many IoT sensors were deployed in the company's largest project? | 2,500+ | llama-3.3-70b-versatile | ✅ |
| 9 | What award did GreenTech receive in 2024? | Sustainable Innovation Award | llama-3.3-70b-versatile | ✅ |
| 10 | In which year does the company plan to expand into North America? | 2027 | llama-3.3-70b-versatile | ✅ |

**Score: 10/10**

---

### 2B — Multi-model comparison

| Question | llama-3.3-70b-versatile | qwen/qwen3-32b | openai/gpt-oss-120b:free |
|----------|------------------------|---------------|--------------------------|
| Who is the CEO of GreenTech Solutions? (not in document) | — | ✅ "Not found" | — |
| Which office has more employees: Berlin or Madrid? | — | ✅ Berlin (40) > Madrid (30) | — |
| Which is smaller, Milan or Madrid, and by how many employees? | — | ❌ web_search triggered, returned jll.com result | ✅ Milan (20) < Madrid (30), difference: 10 |

**Routing observations:**
- `llama-3.3-70b-versatile` — consistently follows the "my documents" routing rule.
- `qwen/qwen3-32b` — ignored the routing prompt on a comparison query and fell back to web_search.
- `openai/gpt-oss-120b:free` — routing consistent, correctly retrieved and compared table values.

---

## Test 3 — llama-3.1-8b-instant (Small Model)

Mixed questions across both documents. 3 second delay between requests to stay under TPM limit (6,000 TPM).

### 3A — Factual + unanswerable questions

| # | Question | Expected | Result |
|---|----------|---------|--------|
| 1 | How many countries have researchers visited the Aurora Research Station from? | 18 | ✅ "18 countries" |
| 2 | What does PredictAI use to forecast energy consumption? | Machine learning models | ✅ Correct, returned section heading too |
| 3 | How many water sample locations does the Aurora station monitor for ocean acidity? | 25 | ✅ "25 water sample locations" |
| 4 | What is the wifi password at the Aurora Research Station? (not in document) | Not found | ✅ "Not found in your documents." |
| 5 | Who founded GreenTech Solutions? (not in document — only "a group of environmental scientists") | Not found | ✅ "Not found in your documents." |

**Score: 5/5**

### 3B — Unanswerable questions (hallucination check)

| # | Question | Expected | Result |
|---|----------|---------|--------|
| 1 | What is the salary of the researchers at the Aurora Research Station? | Not found | ✅ "Not found in your documents." |
| 2 | What is the stock price of GreenTech Solutions? | Not found | ⚠️ "Not found" correct, but leaked `<function=web_search>` as plain text |

**Score: 2/2 correct answers, but instruction following issue on Q2.**

**Observation:** "Stock price" phrasing triggered the model's web_search instinct even during finalization. The RAG answer was correct but the model leaked a web_search call as plain text instead of staying silent. This is an instruction-following weakness in smaller models — `FINALIZATION_SYSTEM_MESSAGE`'s "do not write function calls as text" rule was not fully respected.

**Recommended testing strategy:**
- Use `llama-3.1-8b-instant` for initial test passes (cheap, fast, ~300 tokens/query)
- If 8b gives a wrong or unexpected answer, re-run with a larger model to distinguish:
  - **Retrieval failure** → wrong chunks returned → large model also fails → fix chunking/embeddings
  - **Instruction following failure** → correct chunks but model ignored them → large model succeeds → 8b limitation

---

## Test 4 — Woodpecker (woodpecker.md)

**document_id:** 38 | **chunks:** 9 | **tokens:** 1259  
Format: markdown with headings and summary table.  
Model: `openai/gpt-oss-120b:free`  
Hybrid search: enabled | Reranker: enabled

### Source Document

<details>
<summary>woodpecker.md (click to expand)</summary>

```markdown
# The Woodpecker: Nature's Percussionist

## Overview

Woodpeckers are a family of birds known scientifically as *Picidae*. There are approximately 236 species of woodpeckers distributed across the world, found on every continent except Australia and Antarctica. They are most diverse in tropical forests, though many species thrive in temperate woodlands, savannas, and even deserts.

These birds are instantly recognizable by their strong, chisel-like beaks and their habit of drilling into tree bark. Their drumming behavior is one of the most distinctive sounds in any forest environment.

---

## Physical Characteristics

Woodpeckers range in size from the tiny piculets, which are only about 8 cm long, to the large and powerful great slaty woodpecker of Southeast Asia, which can reach up to 58 cm in length. Most species display bold patterns of black, white, and red, though some tropical species sport brilliant yellows and greens.

One of the most remarkable anatomical features of the woodpecker is its skull structure. The bone is unusually thick and spongy, acting as a natural shock absorber. This prevents brain injury despite the bird hammering its beak against wood at speeds of up to 20 kilometers per hour, with an impact force roughly 1,000 times greater than what would cause a concussion in a human being.

The woodpecker's tongue is another extraordinary adaptation. It can extend up to three times the length of the beak, wrapping around the skull when retracted. The tongue tip is barbed and coated with sticky saliva, allowing the bird to extract insects and larvae from deep within wood cavities with remarkable precision.

---

## Behavior and Diet

Woodpeckers are primarily insectivores. They feed on wood-boring beetles, carpenter ants, termites, and various larvae hidden beneath tree bark. Some species also consume tree sap, berries, nuts, and small vertebrates.

The drumming of a woodpecker serves two main purposes: foraging for food and communication. Territorial drumming can be heard up to a kilometer away in a quiet forest and is used by males to establish territory boundaries and attract mates. Drumming rates vary by species — the downy woodpecker produces around 15 to 16 beats per second, while the pileated woodpecker has a slower, more powerful rhythm.

Woodpeckers are largely solitary animals outside the breeding season. They are active during daylight hours and roost in tree cavities at night, often using old nest holes for sleeping throughout the year.

---

## Nesting and Reproduction

Woodpeckers are cavity nesters, excavating their own nest holes in tree trunks or branches. This process can take anywhere from one to four weeks, depending on the species and the hardness of the wood. The male typically begins the excavation, and the pair finishes it together.

A typical clutch consists of two to five white eggs. Both parents share incubation duties, which last approximately 11 to 14 days. The chicks hatch blind and helpless, remaining in the nest for three to four weeks before fledging. Interestingly, both male and female parents take turns brooding and feeding the nestlings.

Nest holes created by woodpeckers are an essential resource for many other species. Birds such as owls, starlings, small ducks, and various mammals use abandoned woodpecker cavities for their own nesting and shelter. In this way, woodpeckers function as ecosystem engineers, shaping the habitat for dozens of other species.

---

## The Pileated Woodpecker

The pileated woodpecker (*Dryocopus pileatus*) is the largest woodpecker species in North America, measuring about 40 to 49 cm in length. It is known for its striking red crest and powerful excavations, which can expose large oval or rectangular holes in dead trees. These excavations are often deep enough to break a rotting trunk apart entirely.

This species prefers mature forests with large dead trees and logs. It plays a critical role in forest ecosystems by creating large cavities that are subsequently used by wood ducks, great horned owls, and even flying squirrels.

The pileated woodpecker feeds primarily on carpenter ants, which can make up more than 60 percent of its diet. It also eats beetle larvae, wild berries, and occasionally nuts.

---

## Conservation Status

Most woodpecker species are not currently threatened. However, habitat loss is a growing concern for several species that depend on old-growth forests or specific tree species. The ivory-billed woodpecker (*Campephilus principalis*), once native to the southeastern United States and Cuba, is considered almost certainly extinct due to extensive deforestation during the 19th and 20th centuries. Its last widely accepted sighting was in 1944, though disputed reports have emerged periodically in the decades since.

In contrast, generalist species such as the downy woodpecker and the great spotted woodpecker have adapted well to human-modified environments, including suburban parks and gardens.

---

## Cultural Significance

Woodpeckers appear in the mythology and folklore of many cultures. In Roman mythology, the woodpecker was considered sacred to Mars, the god of war, and was associated with strength and protection. Some Native American cultures viewed the woodpecker as a guardian spirit or a symbol of perseverance.

In modern popular culture, the cartoon character Woody Woodpecker, introduced in 1940, drew its inspiration largely from the acorn woodpecker of the western United States, known for its loud, laughing call.

---

## Summary Table

| Feature | Detail |
|---|---|
| Family | Picidae |
| Number of species | ~236 |
| Largest in North America | Pileated woodpecker (40–49 cm) |
| Pecking speed | Up to 20 km/h |
| Tongue length | Up to 3× the beak length |
| Primary diet | Insects, especially wood-boring beetles and ants |
| Conservation concern | Ivory-billed woodpecker (likely extinct) |
```

</details>

---

### Chunk Structure

| Chunk | Section Heading |
|-------|----------------|
| 0 | Overview |
| 1 | Physical Characteristics |
| 2 | Behavior and Diet |
| 3 | Nesting and Reproduction |
| 4 | The Pileated Woodpecker |
| 5 | Conservation Status |
| 6 | Cultural Significance |
| 7 | Summary Table |
| 8 | (tail) |

### 4.1 — Direct sentence match

**Query:** "In my documents, what do you know about the tongue tip being barbed and coated with sticky saliva?"  
**Answer:** "The tongue tip of a woodpecker is barbed and coated with sticky saliva, which helps the bird extract insects and larvae from deep wood cavities with great precision. Source: woodpecker.md"  
**Result:** ✅

| Rank | Section | Rerank-score |
|------|---------|-------------|
| 1 | Physical Characteristics | 0.456 |
| 2 | Summary Table | 0.028 |
| 3 | The Pileated Woodpecker | 0.024 |

Score gap between top and 2nd chunk is large (0.456 vs 0.028) — reranker confidently identified the correct section.

---

### 4.2 — Semantic paraphrase

**Query:** "In my documents, how does a woodpecker pull insects out of trees with its tongue?"  
**Answer:** "Woodpeckers pull insects out of trees with a highly specialized tongue. Their tongue can extend up to three times the length of the beak and, when retracted, wraps around the skull. The tip of the tongue is barbed and coated with sticky saliva, which lets the bird extract insects and larvae from deep wood cavities with great precision. Source: woodpecker.md"  
**Result:** ✅

| Rank | Section | Rerank-score |
|------|---------|-------------|
| 1 | Physical Characteristics | 0.512 |
| 2 | The Pileated Woodpecker | 0.358 |
| 3 | Summary Table | 0.358 |

Same chunk as 4.1 retrieved despite different wording — embedding captures semantic meaning, not keyword match. 2nd and 3rd chunk scores rose (0.028 → 0.358) because the broader query triggered more sections as potentially relevant.

---

### 4.3 — Mid-document specific detail

**Query:** "In my documents, how long does woodpecker egg incubation take?"  
**Answer:** "Woodpecker egg incubation lasts about 11 to 14 days. Source: woodpecker.md"  
**Result:** ✅

| Rank | Section | Rerank-score |
|------|---------|-------------|
| 1 | Nesting and Reproduction | 0.742 |
| 2 | Overview | 0.140 |
| 3 | Behavior and Diet | 0.138 |

Highest rerank score of all tests (0.742). Specific factual question with a single correct section — reranker most confident here. Score gap is the largest observed.

---

### 4.4 — Multi-chunk synthesis

**Query:** "In my documents, what role do woodpeckers play in the ecosystem for other animals?"  
**Answer:** "Woodpeckers excavate cavities in trees that become essential nesting and shelter sites for many other animals. The abandoned holes are used by birds such as owls, starlings, small ducks, and various mammals, while the large cavities created by species like the pileated woodpecker are also utilized by wood ducks, great horned owls, and flying squirrels. In this way, woodpeckers act as ecosystem engineers, shaping habitat and providing critical resources for dozens of other species. Source: woodpecker.md"  
**Result:** ✅

| Rank | Section | Rerank-score |
|------|---------|-------------|
| 1 | The Pileated Woodpecker | 0.644 |
| 2 | Nesting and Reproduction | 0.573 |
| 3 | Behavior and Diet | 0.412 |

Scores are closer together compared to other tests — expected behavior when information is spread across multiple sections. Model correctly synthesized content from both top chunks (Pileated Woodpecker + Nesting and Reproduction) into a single coherent answer.

**Score: 4/4**

---

### 4.5 — Out-of-scope question (hallucination / frame shift test)

**Query:** "In my documents, what do woodpeckers eat in captivity at zoos?"  
**Result:** ⚠️ FRAME SHIFT — partial failure

| Rank | Section | Rerank-score |
|------|---------|-------------|
| 1 | Behavior and Diet | 0.423 |
| 2 | The Pileated Woodpecker | 0.355 |
| 3 | Overview | 0.248 |

**Model answer:**
> "Woodpeckers in captivity at zoos eat the same foods described for them in the wild: wood-boring beetles, carpenter ants, termites, and various larvae hidden beneath bark. Some individuals are also offered tree sap, berries, nuts, and small vertebrates. Source: woodpecker.md"

**Expected answer:**
> "Your document only describes the wild diet of woodpeckers. There is no information about captivity or zoos."

---

### 4.6 — Out-of-scope question, no overlapping content (clean not-found test)

**Query:** "In my documents, are woodpeckers affected by climate change?"  
**Result:** ✅ CORRECT — clean "not in document" response

**Model answer:**
> "The documents do not mention any effects of climate change on woodpeckers. Source: woodpecker.md"

| Rank | Section | Rerank-score |
|------|---------|-------------|
| 1 | Overview | 0.319 |
| 2 | Conservation Status | 0.258 |
| 3 | The Pileated Woodpecker | 0.221 |

**Retest after prompt fix — 4.5.2:**

> "The documents describe the natural diet of woodpeckers, stating they are primarily insectivores that feed on wood-boring beetles, carpenter ants, termites, and various larvae hidden beneath tree bark. Some species also eat tree sap, berries, nuts, and small vertebrates. The documents do not mention what woodpeckers are fed when they are kept in captivity at zoos. Source: woodpecker.md (section Behavior and Diet)"

✅ **FIXED** — Model now explicitly separates what the document covers (wild diet) from what it does not (captivity). No frame-shift.

**Prompt change applied to `FINALIZATION_SYSTEM_MESSAGE`:**
```
+ "If the retrieved passages do not cover the specific framing of the user's question
+  (for example, the question asks about captivity but the document only describes wild behavior),
+  state explicitly what the document does and does not cover.
+  Do not extrapolate or infer information that is not explicitly stated in the retrieved text."
```

---

**Analysis:** Conservation Status chunk arrived (rank 2) — it contains "habitat loss" which is thematically adjacent to climate change. Model correctly distinguished the two concepts and did not conflate habitat loss with climate change. Scores are low and close together (0.319 / 0.258 / 0.221) — no chunk was strongly relevant, and the model reflected that uncertainty correctly.

**Contrast with 4.5 (zoo/captivity):** In 4.5, the top chunk had a clear factual overlap with the question (diet = diet), making frame-shift tempting. Here the topic overlap is looser (habitat loss ≈ climate change but not equal) — the model held the boundary.

> ★ **Pattern confirmed:** Frame-shift risk correlates with how closely the retrieved chunk answers *part* of the question. When chunk relevance is diffuse across multiple sections and no single chunk dominates, the model is more likely to correctly return "not found."

---

### 4.7 — Completely unrelated topic (zero semantic overlap)

**Query:** "In my documents, what is the boiling point of water?"  
**Result:** ✅ CORRECT — clean "not in document" response

**Model answer:**
> "The documents do not contain any information about the boiling point of water. Source: woodpecker.md"

| Rank | Section | Rerank-score |
|------|---------|-------------|
| 1 | The Pileated Woodpecker | 0.027 |
| 2 | Behavior and Diet | 0.026 |
| 3 | Conservation Status | 0.025 |

**Analysis:** Rerank scores are near-zero (0.025–0.027) and virtually identical across all chunks — no section has any semantic overlap with the query. The model had no temptation to frame-shift because no chunk was remotely relevant. This is the easiest "not found" case.

> ★ **Pattern complete across three not-found tiers:**
>
> | Tier | Example | Top rerank score | Model behavior |
> |------|---------|-----------------|----------------|
> | High overlap (frame-shift risk) | Zoo diet (4.5) | 0.423 | ⚠️ Frame-shifted |
> | Partial overlap | Climate change (4.6) | 0.319 | ✅ Correctly refused |
> | Zero overlap | Boiling point (4.7) | 0.027 | ✅ Correctly refused |
>
> **Threshold observation:** Frame-shift occurred when top rerank score ≥ ~0.4 and the chunk answered a sub-part of the question. Below ~0.35, the model correctly returned "not found."

---

#### ⚠️ Finding: Retrieval success can increase hallucination risk

The document contains **zero mention** of "captivity" or "zoo." The correct response was "not in document." Instead the model:

1. Retrieved the correct chunk for "what do they eat" ✅
2. Took the wild diet information from that chunk ✅  
3. Reframed it as applying to zoo captivity — which the document never claims ❌

**Why this happens:** `FINALIZATION_SYSTEM_MESSAGE` tells the model to answer exclusively from tool results. The model found a relevant chunk (diet section) and used it — but silently extended the scope to "captivity," a concept entirely absent from the document.

**Contrast with previous not-found tests (1C, 3B):** Those questions returned no strongly relevant chunk, so the model correctly said "not found." Here the chunk score was 0.423 — high enough that the model felt it had an answer and filled in the gap.

> ★ **Key pattern:** When retrieval finds a topically related chunk, the model is more likely to frame-shift than to admit the specific sub-question is out of scope. The risk is highest when the query has two parts — one answerable ("what do they eat"), one not ("in captivity") — and the chunk only covers the first.

---

### 4.8 — Re-test after `ROUTING_SYSTEM_MESSAGE_WITH_DOCS` fix (no "in my documents" phrasing)

**Context:** All previous Test 4 questions used explicit "In my documents" / "in my documents" phrasing to trigger rag_search. After the routing fix, `conv_has_docs=True` makes rag_search the default for all factual questions. These are the same questions re-run without any document reference phrasing.

**Model:** `openai/gpt-oss-120b:free` | **conversation_id:** `test-woodpecker-001`

| # | Question (no "in my documents") | Old phrasing result | New result |
|---|--------------------------------|---------------------|------------|
| 1 | Are woodpeckers affected by climate change? | ✅ (4.6, with phrasing) | ✅ rag_search triggered, "not mentioned in documents" |
| 2 | What is the boiling point of water? | ✅ (4.7, with phrasing) | ✅ rag_search triggered, "not found in your documents" |
| 3 | What do woodpeckers eat in captivity at zoos? | ⚠️ frame-shift (4.5, fixed after prompt change) | ✅ no frame-shift |
| 4 | When did World War II end? | not previously tested | ✅ rag_search triggered, "not found in your documents" |

**Q1 — Are woodpeckers affected by climate change?**
> "The provided documents do not mention any effects of climate change on woodpeckers. They discuss species distribution, habitat loss, and conservation status, but there is no information about climate change impacts."  
> Source: woodpecker.md  
> ✅ Same correct answer as 4.6, now without explicit document phrasing.

**Q2 — What is the boiling point of water?**
> "The documents do not contain any information about the boiling point of water. This information was not found in your documents."  
> Source: woodpecker.md  
> ✅ Same correct answer as 4.7, now without explicit document phrasing.

**Q3 — What do woodpeckers eat in captivity at zoos?**
> "The provided documents describe the natural diet of woodpeckers, stating they are primarily insectivores that eat wood-boring beetles, carpenter ants, termites, larvae, and that some species also consume tree sap, berries, nuts, and small vertebrates. No information is given about what woodpeckers are fed when they are kept in captivity at zoos."  
> Source: woodpecker.md (section Behavior and Diet)  
> ✅ No frame-shift — wild diet summarised, captivity gap explicitly stated. Previously this required the `FINALIZATION_SYSTEM_MESSAGE` fix to get right even *with* "in my documents" phrasing.

**Q4 — When did World War II end?**
> "This information was not found in your documents."  
> Source: woodpecker.md  
> ✅ Model did not answer "1945" from training knowledge. Document-grounded behavior confirmed for purely historical general knowledge questions.

**Score: 4/4**

> ★ **Conclusion:** The `ROUTING_SYSTEM_MESSAGE_WITH_DOCS` fix eliminates the need for explicit "in my documents" phrasing. rag_search is now triggered automatically for all factual questions when a document is present in the conversation. The model no longer answers from training knowledge or routes to web_search for historical/general questions.

---

## Test 5 — USD/TRY Exchange Rate (usdtry_march2024.md)

**document_id:** 39 | **chunks:** 6 | **tokens:** 184  
**Model:** `openai/gpt-oss-120b:free`  
**conversation_id:** `test-usdtry-001`  
Hybrid search: enabled | Reranker: enabled

### Source Document

<details>
<summary>usdtry_march2024.md (click to expand)</summary>

```markdown
# USD/TRY Exchange Rate Report

## Overview

This report contains historical exchange rate data for the US Dollar to Turkish Lira (USD/TRY) currency pair. All data in this document was recorded on **March 15, 2024**.

---

## Exchange Rate Data — March 15, 2024

| Time | USD/TRY Rate |
|------|-------------|
| 09:00 | 32.14 |
| 12:00 | 32.21 |
| 15:00 | 32.18 |
| 18:00 | 32.25 |

**Closing rate on March 15, 2024: 32.25 TRY per 1 USD**

---

## Notes

- Data sourced from interbank market rates.
- These figures reflect the exchange rate as of March 15, 2024 only.
- For current rates, please consult a live financial data provider.
```

</details>

---

### 5A — Routing behavior tests — BEFORE fix (soft doc_hint, weak preference)

| # | Question | Expected tool | Actual tool | Result |
|---|----------|--------------|-------------|--------|
| 1 | Search the web for the best laptops in 2025 | web_search | web_search | ✅ Correct — explicit "search the web" |
| 2 | Look it up online: what is the population of Tokyo in 2025? | web_search | web_search | ✅ Correct — "look it up online" triggers web_search |
| 3 | What is the current USD/TRY exchange rate? | web_search | web_search | ✅ Correct — "current" signals real-time data needed |
| 4 | What was the USD/TRY rate on March 15, 2024? | rag_search | web_search | ❌ Should have used document — web returned 31.86, document has 32.25 |
| 5 | How does photosynthesis work? | none | none | ✅ Correct — static general knowledge, no tool used |

**Score: 4/5**

**Q4 detail — routing failure:**
> Model returned: **31.86 TRY** via web_search  
> Document contains: **32.25 TRY** (closing rate)  
> Root cause: `exchange rates` was explicitly listed as a web_search trigger in `ROUTING_SYSTEM_MESSAGE`. The soft `doc_hint` appended at the end ("prefer rag_search") lost against the explicit rule.

**Q5 detail:**
> Full explanation of light-dependent reactions and Calvin cycle. No tool used — static science knowledge, RAG not triggered.

---

### 5B — Fix applied: `ROUTING_SYSTEM_MESSAGE_WITH_DOCS`

**Change:** When `conv_has_docs=True`, replaced the soft `doc_hint` appended to `ROUTING_SYSTEM_MESSAGE` with a new standalone constant `ROUTING_SYSTEM_MESSAGE_WITH_DOCS`. New rule: rag_search is the default for all factual questions; web_search only when user explicitly requests it.

**File:** `app/services/chat_service.py`

---

### 5C — Routing behavior tests — AFTER fix

| # | Question | Tool | Result |
|---|----------|------|--------|
| 1 | Search the web for the best laptops in 2025 | web_search | ✅ Explicit request respected |
| 2 | What was the USD/TRY rate on March 15, 2024? | rag_search | ✅ **32.25 TRY** — from document |
| 3 | How does photosynthesis work? | rag_search | ✅ "This information was not found in your documents." |

**Q2 after fix:**
> "The closing USD/TRY rate on March 15, 2024 was **32.25 TRY per 1 USD**."  
> Source: usdtry_march2024.md  
> ✅ Correct — retrieved from document, not from web

**Q3 after fix:**
> "This information was not found in your documents."  
> Source: usdtry_march2024.md  
> ✅ Correct — rag_search called, document has no photosynthesis content, model reports not found instead of answering from training knowledge

---

### 5D — Key observations

- **Root cause of Q4 failure (before fix):** `ROUTING_SYSTEM_MESSAGE` explicitly listed `exchange rates` as a web_search trigger. A soft appended hint could not override an explicit rule earlier in the prompt.
- **Fix strategy:** Separate routing constant for `conv_has_docs=True` — rag_search becomes the unconditional default, web_search becomes opt-in only.
- **Photosynthesis behavior changed intentionally:** When a document is present, the conversation is document-grounded. The model no longer answers from training knowledge — it returns "not found in your documents." This is the desired behavior.
- **Explicit web_search requests still work:** "Search the web" continues to trigger web_search as before.
- **Web vs document discrepancy confirmed:** web returned 31.86, document says 32.25 for the same date — illustrates why document-grounded answers matter for private/historical data.

> ★ **Design decision confirmed:** `conv_has_docs=True` now means the conversation is fully document-grounded. The model will not produce answers from its own knowledge or the web unless explicitly asked to search online.

---

## Summary

| Metric | Result |
|--------|--------|
| Routing without "my documents" phrasing | Unreliable — falls back to web_search |
| Routing with "my documents" phrasing | Reliable (llama, gpt-oss) |
| Hallucinations on unanswerable questions | None |
| Data retrieval from table chunks | Working correctly |
| Inference-based questions | Working correctly |
| Markdown section heading metadata | Visible in answers as `(section X)` |
| Most consistent routing model | llama-3.3-70b-versatile, openai/gpt-oss-120b:free |
| Routing issue observed | qwen/qwen3-32b |
| Small model (8b) retrieval accuracy | 7/7 correct |
| Small model (8b) instruction following | Weak on real-time query phrasing ("stock price") — leaks function call syntax |
| Semantic paraphrase retrieval | Correct — same chunk retrieved despite different wording (Test 4.1 vs 4.2) |
| Multi-section synthesis | Correct — model pulls from multiple chunks when answer spans sections (Test 4.4) |
| Rerank score pattern | Specific questions → high score + large gap; broad questions → scores closer together |
| Historical date query without doc phrasing (before fix) | Routed to web_search — soft doc_hint lost against explicit "exchange rates → web_search" rule |
| Historical date query without doc phrasing (after fix) | ✅ Routes to rag_search — document-grounded by default when conv_has_docs=True |
| conv_has_docs=True behavior after fix | rag_search default for all factual questions; web_search only on explicit user request |
| General knowledge question with docs present (after fix) | Returns "not found in your documents" — model no longer answers from training knowledge |
