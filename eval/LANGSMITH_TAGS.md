# LangSmith Tag Reference

All traces in the `groq-stream-fastapi` project can be filtered by tag in LangSmith
(**Runs → Filter → Tag**). This file documents every tag in use, what it means, and
which test script produces it.

---

## How tags are structured

Every trace always carries the **model name** as a tag (added automatically by
`chat_service.py`). Test scripts add additional tags on top of that.

```
[model_name]  +  [test-suite tag]  +  [question-type tag]
```

Example for a cross-document question run with llama:
```
llama-3.3-70b-versatile | rag | source_attribution | cross_document
```

---

## Tag definitions

### Test-suite tags
These identify which test suite a trace belongs to.

| Tag | Meaning | Script |
|-----|---------|--------|
| `rag` | Trace was produced by a RAG test — documents were loaded and retrieval was expected | `eval/run_multidoc_source_test.py` |

### Capability tags
These identify what capability is being evaluated within a test suite.

| Tag | Meaning | Script |
|-----|---------|--------|
| `source_attribution` | The answer is expected to end with a correct `Source: <filename>` or `Source: <url>` line | `eval/run_multidoc_source_test.py` |

### Question-type tags
These identify the specific retrieval scenario being tested.

| Tag | Meaning | Expected behavior | Script |
|-----|---------|-------------------|--------|
| `single_source` | Question can be answered from exactly one of the loaded documents. The model should cite only that document and must not bleed information from the other. | `rag_search` returns chunks from one document; `Source:` line names that document only. | `eval/run_multidoc_source_test.py` — Q1, Q2 |
| `cross_document` | Question requires synthesizing information from both loaded documents. The model must cite both sources explicitly. | `rag_search` returns chunks from both documents; response references both and ends with both `Source:` lines. | `eval/run_multidoc_source_test.py` — Q3, Q4, Q5 |
| `web_fallback` | Question asks for real-time or externally verifiable information. The question includes an explicit "internetten ara" instruction so that `web_search` is triggered even though documents are present. | `web_search` tool is called; answer ends with `Source: <url>`. | `eval/run_multidoc_source_test.py` — Q6, Q7 |

---

## Filter cheat-sheet

| What you want to see | LangSmith filter |
|----------------------|-----------------|
| All RAG test traces | `rag` |
| All source attribution tests | `source_attribution` |
| Only single-source questions | `single_source` |
| Only cross-document questions | `cross_document` |
| Only web fallback questions | `web_fallback` |
| Combine: RAG + cross-document only | `rag` + `cross_document` |

---

## Test scripts index

| Script | Dataset / purpose | Tags produced |
|--------|-------------------|---------------|
| `eval/run_eval.py` | `tool-routing-eval-v2` — 30 routing accuracy questions, no document upload | model name only |
| `eval/run_multidoc_source_test.py` | 7 questions over 2 fixed documents (legal + academic). Tests single-source attribution, cross-document synthesis, and web fallback routing. | `rag`, `source_attribution`, + one of `single_source` / `cross_document` / `web_fallback` |

---

## Adding new tags

When a new test script is added:
1. Define its tags as a constant at the top of the script (see `BASE_TAGS` in
   `eval/run_multidoc_source_test.py`).
2. Add a row to the **Test scripts index** table above.
3. Add a row to the **Tag definitions** section for any new tags introduced.
