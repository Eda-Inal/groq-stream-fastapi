# TechCorp — RAGAS Evaluation

LLM-as-judge metrics. Faith = Faithfulness, Correct = Answer Correctness, Recall = Context Recall, Precision = Context Precision.

## run_009 — 2026-06-23 13:27 | chat: meta-llama/llama-4-scout-17b-16e-instruct | judge: meta-llama/llama-4-scout-17b-16e-instruct
**Avg: Faith=1.0 | Correct=None | Recall=1.0 | Precision=1.0**

| Q | Diff | Category | Faith | Correct | Recall | Precision | Status |
|---|---|---|---|---|---|---|---|
| Q01 | easy | exact_match | 1.00 | — | 1.00 | 1.00 | ❌ |
| Q02 | easy | exact_match | 1.00 | — | 1.00 | 1.00 | ❌ |
| Q03 | easy | exact_match | 1.00 | — | 1.00 | 1.00 | ❌ |

## run_010 — 2026-06-23 13:28 | chat: meta-llama/llama-4-scout-17b-16e-instruct | judge: meta-llama/llama-4-scout-17b-16e-instruct
**Avg: Faith=1.0 | Correct=0.8547 | Recall=1.0 | Precision=1.0**

| Q | Diff | Category | Faith | Correct | Recall | Precision | Status |
|---|---|---|---|---|---|---|---|
| Q01 | easy | exact_match | 1.00 | 0.98 | 1.00 | 1.00 | ✅ |
| Q02 | easy | exact_match | 1.00 | 0.60 | 1.00 | — | ❌ |
| Q03 | easy | exact_match | — | 0.98 | 1.00 | — | ❌ |

## run_011 — 2026-06-23 13:41 | chat: meta-llama/llama-4-scout-17b-16e-instruct | judge: meta-llama/llama-4-scout-17b-16e-instruct
**Avg: Faith=1.0 | Correct=0.8974 | Recall=1.0 | Precision=1.0**

| Q | Diff | Category | Faith | Correct | Recall | Precision | Status |
|---|---|---|---|---|---|---|---|
| Q01 | easy | exact_match | 1.00 | 0.98 | 1.00 | 1.00 | ✅ |
| Q02 | easy | exact_match | 1.00 | 0.97 | 1.00 | 1.00 | ✅ |
| Q03 | easy | exact_match | 1.00 | 0.98 | 1.00 | 1.00 | ✅ |
| Q04 | easy | named_entity | 1.00 | 0.82 | 1.00 | 1.00 | ✅ |
| Q05 | easy | named_entity | 1.00 | 0.73 | 1.00 | 1.00 | ✅ |

## run_013 — 2026-06-23 14:05 | chat: llama-3.3-70b-versatile | judge: openai/gpt-oss-120b
**Avg: Faith=0.8438 | Correct=0.6962 | Recall=0.9375 | Precision=1.0**

| Q | Diff | Category | Faith | Correct | Recall | Precision | Status |
|---|---|---|---|---|---|---|---|
| Q02 | easy | exact_match | 1.00 | 0.97 | 1.00 | 1.00 | ✅ |
| Q06 | easy | direct_fact | 1.00 | 0.97 | 1.00 | 1.00 | ✅ |
| Q11 | medium | paraphrase | 1.00 | 0.18 | 1.00 | 1.00 | ❌ |
| Q16 | medium | distractor | 0.50 | 0.57 | 1.00 | 1.00 | ❌ |
| Q21 | medium | multi_chunk | 1.00 | 0.67 | 1.00 | 1.00 | ✅ |
| Q25 | medium | inference | 0.75 | 0.88 | 1.00 | 1.00 | ❌ |
| Q28 | medium | negation | 1.00 | 0.60 | 1.00 | 1.00 | ❌ |
| Q31 | hard | adversarial_paraphrase | 0.50 | 0.72 | 0.50 | 1.00 | ❌ |
| Q33 | hard | absent_plausible | — | — | — | — | ⏭️ — |
