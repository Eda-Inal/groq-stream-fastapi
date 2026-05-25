# PROJECT_AUDIT.md

> **Purpose of this file:** Onboard an AI assistant to this codebase without needing to re-read every file.
> Read this before touching any code. Update it when architecture changes.

---

## Project Overview

A production-ready FastAPI backend that exposes an **OpenAI-compatible streaming chat API** with:
- Multi-provider LLM routing (Groq, OpenRouter, Google Gemini)
- Tool-augmented generation via an MCP (Model Context Protocol) tool server
- Private RAG knowledge base (pgvector + hybrid search + optional reranking)
- Full LangSmith tracing (every LLM call, tool call, and RAG result as child spans)
- LangSmith eval infrastructure for automated tool-routing accuracy testing

---

## Architecture

```
Client
  └─ POST /api/v1/chat/stream (SSE)
       └─ ChatService.stream_chat()
            ├─ LangSmith root span (chain)
            ├─ LLMClient.stream_chat_completion()   ← LLM call 1: tool routing
            │    └─ LangSmith child span (llm)
            ├─ RemoteMCPClient.call_tool()          ← per tool call
            │    └─ LangSmith child span (tool)
            └─ LLMClient.stream_chat_completion()   ← LLM call 3: finalization
                 └─ LangSmith child span (llm)
```

### Two-Service Docker Deployment

| Service         | Port | Entry point                    |
|-----------------|------|-------------------------------|
| `api`           | 8000 | `uvicorn app.main:app`        |
| `mcp`           | 8001 | `uvicorn app.mcp_server.main:app` |
| `db`            | 5432 | `pgvector/pgvector:pg15`      |

`api` calls `mcp` over HTTP (`MCP_SERVER_URL=http://mcp:8001`). They share the same Docker image.

---

## Key Flows

### Chat Request (happy path)

1. `POST /api/v1/chat/stream` → `chat.py:stream_chat()` resolves model, conversation_id, calls `ChatService.stream_chat()`
2. `ChatService.stream_chat()`:
   - Loads conversation history from DB (last 20 turns, scoped to user_id + conversation_id)
   - Fetches tool schemas from MCP server via `RemoteMCPClient.list_tools()`
   - Prepends `RAG_TOOL_CALL_SYSTEM_MESSAGE` (tool routing prompt)
   - **Round loop** (max 8 rounds, max 10 total tool calls):
     - **LLM call 1** (`_traced_llm_call`, name=`llm.{model}.tool_routing`): asks model to route
     - If model returns tool calls → execute each via `RemoteMCPClient.call_tool()`
       - `rag_search` gets user_id injected into `metadata_filter` (model never controls this)
       - Re-injects last successful RAG result as a system message
       - Appends `FINALIZATION_SYSTEM_MESSAGE`
       - Applies `_apply_context_budget()` to stay under token limit
       - **LLM call 3** (`_traced_llm_call`, name=`llm.{model}.finalization`): answer from tool results
     - If no tool call → yield buffered text chunks directly
     - If Groq `failed_generation` error → fallback path: manually call RAG, inject result, retry without tools (**LLM call 2**, name=`llm.{model}.fallback_no_tools`)
3. Persists `ChatLog` to DB
4. Root LangSmith span closed in `finally` block

### Document Ingestion

1. `POST /api/v1/documents/upload` (PDF) or `POST /api/v1/documents` (text/JSON)
2. `IngestionService.ingest_document()`:
   - Deduplication check by (filename, user_id)
   - `chunk_document()` → recursive split (paragraphs → lines → sentences → token windows), configurable size/overlap
   - `EmbeddingService.embed_batch()` → all-or-nothing batch embedding (retries with backoff)
   - `Document` + N `DocumentChunk` rows written atomically
3. `POST /api/v1/documents/{id}/reprocess` → `IngestionService.reprocess_document()`: delete existing chunks, re-chunk with new text/heading, re-embed, persist. Useful for correcting OCR errors without re-uploading.

### RAG Search (inside tool call)

1. `RagSearchTool.run()` embeds query → `search_document_chunks()` (pgvector)
2. Hybrid mode: dense (cosine) + sparse (BM25/tsvector) legs merged via RRF
3. Similarity threshold filter (disabled in hybrid mode — RRF scores differ)
4. Optional Jina reranker: over-fetches `top_k × 3` candidates, reranks, keeps top_k
5. Returns blocks: `Source: filename (page X)\nSimilarity: 0.85\nContent: "..."`  separated by `\n---\n`
6. `ChatService` re-injects full RAG result as system message so model cannot ignore it

---

## File Map

### Entry Points

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, CORS middleware, mounts `/api/v1` router |
| `app/mcp_server/main.py` | Separate FastAPI app for the MCP tool server, port 8001 |
| `docker-compose.yml` | Defines `api`, `mcp`, `db` services |

### Core Config

| File | Purpose |
|------|---------|
| `app/core/config.py` | `Settings` (pydantic-settings). All env vars with defaults. `AVAILABLE_MODELS` dict mapping model id → `{provider, tier, context_window}` |
| `.env` | Live config (not committed). See `.env.example` |
| `.env.example` | Template with all required keys |

**Important config fields:**
- `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` — provider auth
- `TAVILY_API_KEY` — web search
- `DATABASE_URL` — PostgreSQL connection string (`postgresql+asyncpg://...`)
- `MCP_SERVER_URL` — URL the API uses to reach the MCP server (e.g. `http://mcp:8001`)
- `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL_NAME` — embedding service
- `HYBRID_SEARCH_ENABLED`, `RERANKER_ENABLED`, `RERANKER_API_KEY` — RAG pipeline flags
- `LANGSMITH_API_KEY`, `LANGSMITH_TRACING_ENABLED`, `LANGSMITH_PROJECT` — observability
- `WEB_SEARCH_ENABLED`, `CALCULATOR_ENABLED` — feature flags to disable individual tools

### API Layer

| File | Purpose |
|------|---------|
| `app/api/v1/router.py` | Aggregates all routers under `/api/v1` |
| `app/api/v1/endpoints/chat.py` | `GET /chat/models`, `POST /chat/stream` (SSE), `POST /chat/bulk` |
| `app/api/v1/endpoints/documents.py` | CRUD for documents + PDF upload (`POST /documents/upload`) + `POST /documents/{id}/reprocess` |
| `app/api/v1/endpoints/rag_metrics.py` | `GET /rag/metrics` — proxies metrics from MCP server + total doc/chunk counts from DB |
| `app/api/v1/endpoints/eval.py` | `POST /eval/route` — lightweight routing-only endpoint for eval. Runs 1 LLM call (~800 tokens), returns `{tool, args}`. No tool execution, no answer generation. |

**Chat stream endpoint** (`POST /api/v1/chat/stream`):
- Accepts `ChatStreamRequest`: `messages`, `model`, `user_id`, `conversation_id`, `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `seed`
- Returns OpenAI-compatible SSE: `data: {...}\n\n` chunks, ends with `data: [DONE]\n\n`
- `conversation_id` is server-generated if not provided by client

### Services

| File | Purpose |
|------|---------|
| `app/services/chat_service.py` | Main orchestration: tool-calling loop, context budget, LangSmith spans, chat log persistence |
| `app/services/groq_client.py` | `LLMClient` — async SSE streaming to Groq/OpenRouter/Gemini. Emits typed events: `chunk`, `tool_call`, `done`, `error`, `usage` |
| `app/services/tracing.py` | LangSmith integration. `create_run()` / `end_run()` — safe no-ops when disabled |
| `app/services/embeddings.py` | `EmbeddingService` — retry/backoff, in-memory LRU cache, returns `EmbeddingResult(vector, model_name)` |
| `app/services/ingestion_service.py` | `IngestionService` — chunk + embed + persist documents atomically |
| `app/services/chunking.py` | Recursive text chunking: paragraphs → lines → sentences → token windows. Code fences kept intact. Returns `ChunkRecord(text, token_count, page_number, section_heading)` |
| `app/services/reranker.py` | `RerankerService` — optional Jina reranker, returns `[(orig_idx, score)]` |
| `app/services/rag_metrics.py` | `RagMetrics` singleton (lives in MCP process, NOT the API process). Tracks: `recent_similarity_avg`, `embedding_api_error_rate`, `retrieval_calls_total`, embedding/pgvector latency, avg chunks returned. API accesses these via `RemoteMCPClient.get_metrics()` proxy |
| `app/services/pdf_extractor.py` | PDF text extraction (used by ingestion for PDF uploads) |
| `app/services/mcp/remote_client.py` | `RemoteMCPClient` — HTTP client for `list_tools()` and `call_tool()` |

### MCP Tool Server

| File | Purpose |
|------|---------|
| `app/mcp_server/main.py` | FastAPI app mounting MCP routes, health check |
| `app/mcp_server/routes.py` | `GET /tools`, `POST /tools/call`, `GET /metrics` |
| `app/mcp_server/tools/registry.py` | `ToolRegistry` — instantiates and holds all 3 tools |
| `app/mcp_server/tools/base.py` | `Tool` ABC + `ToolResult(ok, content)`. `execute()` wraps `run()` with structured logging. Redacts sensitive keys (`metadata_filter`, `api_key`, etc.) from logs |
| `app/mcp_server/tools/calculator.py` | `CalculatorTool` — AST-based safe eval. Supports `+`, `-`, `*`, `/`, `**`, unary minus |
| `app/mcp_server/tools/web_search.py` | `WebSearchTool` — Tavily API, returns 3 results as `url: content` lines |
| `app/mcp_server/tools/rag_search.py` | `RagSearchTool` — hybrid pgvector search with optional Jina reranking |

### Database

| File | Purpose |
|------|---------|
| `app/db/base.py` | SQLAlchemy `Base` |
| `app/db/session.py` | `AsyncSessionLocal`, `get_db` dependency |
| `app/db/engine.py` | Engine creation |
| `app/db/models/document.py` | `Document` — metadata (filename, user_id, tags, chunk_count, created_at) |
| `app/db/models/document_chunk.py` | `DocumentChunk` — text + `pgvector(768)` embedding, page_number, section_heading |
| `app/db/models/chat_log.py` | `ChatLog` — full messages payload (JSONB), response, model params, conversation_id, turn_index, user_id |
| `app/db/repositories/document.py` | DB queries: `create_document`, `search_document_chunks` (hybrid search), `list_documents`, etc. |
| `app/db/repositories/chat_log.py` | `create_chat_log`, `list_chat_logs_by_conversation` |

### Utilities

| File | Purpose |
|------|---------|
| `app/utils/token_counter.py` | `count_tokens()`, `estimate_messages_tokens()`, `truncate_rag_chunks()`, `truncate_text_to_token_budget()` — all use tiktoken `cl100k_base` |
| `app/core/logging.py` | structlog setup |
| `app/core/chunking_config.py` | `TIKTOKEN_ENCODING = "cl100k_base"` — shared encoding constant for chunking and token counting |
| `app/services/mcp/client.py` | `MCPClient` ABC — enforces fail-soft contract: `list_tools()` returns `[]` on error, `call_tool()` never raises |

---

## LangSmith Tracing

**Span hierarchy per request:**

```
chat.stream  [chain]
├── llm.{model}.tool_routing  [llm]
│     inputs: messages, model, tool names
│     outputs: text, tool_calls, prompt_tokens, completion_tokens
├── tool.{name}  [tool]   (one per tool call)
│     inputs: name, args
│     outputs: success, content_preview, [chunk_count, chunks_preview for rag]
└── llm.{model}.finalization  [llm]
      inputs: messages with tool results, model
      outputs: text, token counts
```

**Optional spans:**
- `llm.{model}.fallback_no_tools` — created when Groq fails to serialize tool call JSON

**Configuration:**
```env
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=groq-stream-fastapi
LANGSMITH_TRACING_ENABLED=true
```

`tracing.py` lazy-imports `langsmith.run_trees.RunTree` and always returns `None` when disabled — all callers treat `None` as a no-op. Errors in tracing are caught and logged as warnings; they never affect the main request.

---

## Tool Routing System Prompt

Located in `chat_service.py:ROUTING_SYSTEM_MESSAGE`. Strict rules — model must call a tool or output nothing:

1. **rag_search** — ONLY when the question explicitly references the user's own uploaded documents (phrases like "my documents", "my files", "in my contract"). Never use for general knowledge.
2. **web_search** — when the answer depends on real-time or frequently changing data (prices, exchange rates, weather, news, current roles/positions, events after training cutoff). Do NOT use for stable general knowledge (history, geography, science, definitions).
3. **calculator** — ANY arithmetic the user explicitly asks to compute, regardless of simplicity. Never compute in the model's head.
4. **None** — static general-knowledge questions (historical dates, geography, science, classical authors, established formulas) and simple conversational messages. Output nothing; a separate step generates the answer.

**`<|python_tag|>` fallback:** Groq occasionally serializes tool calls as plain text starting with `<|python_tag|>` instead of proper JSON. `chat_service.py` detects this in the stream, parses the tool name via regex, and routes to the correct tool. The routing system message is stripped and replaced with `DIRECT_ANSWER_SYSTEM_MESSAGE` before the fallback LLM call.

**Finalization prompt** (`FINALIZATION_SYSTEM_MESSAGE`): forces model to use retrieved content exclusively, ends response with `Source: [filename]`, `Source: [URL]`, or `Source: calculator`.

---

## Eval Infrastructure

### Dataset

`eval/upload_dataset.py` — uploads 30 questions to LangSmith dataset `tool-routing-eval-v2`:
- calculator (4): explicit arithmetic with natural-language phrasing
- web_search (13): weather, exchange rates, current events, current president, current population, ambiguous/tricky cases
- rag_search (3): "my documents", "my files" phrasing — expected tool name is `rag_search` (matches MCP tool name exactly)
- none (9): Shakespeare, capital of France, H₂O, WWII, simple conversational, unknowable questions
- edge cases: "Find population of Turkey in my documents" → expected `rag_search` (model obeys literal "in my documents"); "Can you find me a good book?" → expected `none` ("find" ≠ RAG)

**Important:** `expected_tool` values must match the actual MCP tool names (`rag_search`, not `rag`).

Run once: `.venv\Scripts\python eval/upload_dataset.py`

### Eval Runner

`eval/run_eval.py` — POSTs each question to `/api/v1/eval/route`, gets the tool name directly, compares with expected, reports to LangSmith. No SSE streaming or heuristic parsing needed.

**Endpoint:** `POST /api/v1/eval/route` — runs only the routing LLM call (~800 tokens vs ~2600 for full `/chat/stream`). Returns `{tool: str, args: dict}`. Defined in `app/api/v1/endpoints/eval.py`.

**Model:** `meta-llama/llama-4-scout-17b-16e-instruct` (Groq). Separate token quota from the default chat model. Good tool-calling accuracy.

**Rate limiting:** `REQUEST_DELAY=2.0s` between questions, group pause of `15s` every 5 questions, up to 4 retries with exponential backoff. Stays well under Groq's 30 RPM limit.

**Warmup:** `GET /api/v1/chat/models` — zero tokens, just verifies the API is reachable.

**Run subset:** set `FILTER = ["substring1", "substring2"]` to run matching questions only.

**Run all 30:** `.venv\Scripts\python eval/run_eval.py` (requires live Docker stack). Takes ~3.5 minutes.

**Baseline accuracy:** 80% (24/30) with `llama-4-scout` and current routing prompt.

**LangSmith project:** `groq-stream-fastapi` → view at https://smith.langchain.com

---

## Multi-Provider LLM

`groq_client.py` resolves provider from `AVAILABLE_MODELS`:
- `groq` → `GROQ_API_KEY` + `https://api.groq.com/openai/v1`
- `openrouter` → `OPENROUTER_API_KEY` + `https://openrouter.ai/api/v1`
- `gemini` → `GEMINI_API_KEY` + `https://generativelanguage.googleapis.com/v1beta/openai`

All providers use the OpenAI-compatible `/chat/completions` endpoint. Gemini strips `frequency_penalty`, `presence_penalty`, `seed` before sending.

**Default model:** `llama-3.3-70b-versatile` (Groq). **Eval model:** `meta-llama/llama-4-scout-17b-16e-instruct`.

---

## Context Budget Management

`_apply_context_budget()` in `chat_service.py` trims history to stay under `max_context_tokens` (default 6000):
1. Trim rag_search tool payloads first (keep top chunks up to `rag_tool_max_context_tokens=2500`)
2. Evict oldest non-protected history messages (protected = system messages + current turn)
3. Last resort: truncate longest tool content blocks

---

## RAG Pipeline Details

**Embedding:** `nomic-embed-text` (768-dim), called via OpenAI-compatible API. In-memory SHA-256 cache.

**Hybrid search (default on):** Two legs merged via Reciprocal Rank Fusion (RRF, k=60):
- Dense leg: pgvector cosine similarity
- Sparse leg: PostgreSQL tsvector BM25-style full-text search
- `hybrid_fetch_multiplier=3` → fetches `top_k × 3` candidates per leg

**Reranker (default off):** Jina `jina-reranker-v2-base-multilingual`. Over-fetches `top_k × 3` candidates from pgvector, reranks, keeps top_k.

**User scoping:** `metadata_filter={"user_id": user_id}` is always injected server-side — the model never controls which user's documents are queried.

**metadata_filter supports 3 keys:** `user_id` (string), `document_type` (string: "pdf"/"text"/"json"/"code"), `tags` (list of strings — uses PostgreSQL JSONB `@>` containment).

**Required DB column:** `document_chunks.text_search` — a PostgreSQL `tsvector` generated column used by the sparse leg of hybrid search (`dc.text_search @@ plainto_tsquery('english', :q)`). Must exist in the DB schema for hybrid search to work.

---

## Key Design Decisions

- **MCP over HTTP:** The tool server is a separate FastAPI process so tools can be scaled, restarted, or swapped independently.
- **`_traced_llm_call` is an async generator:** Uses `try/finally` to always end the LangSmith span, even when the caller breaks out of the loop early.
- **Usage events consumed internally:** `groq_client.py` emits `{"type": "usage", ...}` events. `_traced_llm_call` consumes them for LangSmith and does not forward them to callers.
- **RAG result re-injection:** After tool execution, the last successful RAG result is added as a system message again before finalization. This prevents the model from ignoring retrieved content.
- **Grounding enforcement:** `FINALIZATION_SYSTEM_MESSAGE` instructs the model to answer exclusively from tool results and end with a `Source:` line. The `Source:` line is what the eval runner uses for tool detection.
- **Fail-all-or-nothing embedding:** `embed_batch()` returns `None` if any embedding fails; no partial ingestion.

---

## Development

### Database Migrations

DB schema is managed with **Alembic** (`alembic.ini` in project root). No auto-migration on startup — must run manually:

```powershell
# Apply all pending migrations
.venv\Scripts\alembic upgrade head
```

The schema includes the `text_search` tsvector column on `document_chunks` (required for hybrid search). If this column is missing, set `HYBRID_SEARCH_ENABLED=false` in `.env` as a fallback.

### Build & Run

```powershell
docker compose up --build
```

### Run Eval

```powershell
# Start Docker stack first
docker compose up -d

# Upload dataset once
.venv\Scripts\python eval/upload_dataset.py

# Run full eval (requires live API)
.venv\Scripts\python eval/run_eval.py
```

### Environment Variables (minimum required)

```env
GROQ_API_KEY=...
DATABASE_URL=postgresql+asyncpg://app:app@localhost/app
MCP_SERVER_URL=http://localhost:8001
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING_ENABLED=true
```

---

## Schema Quick Reference

**ChatStreamRequest** (`app/schemas/chat.py`):
`messages`, `model?`, `user_id?`, `conversation_id?`, `temperature?`, `max_tokens?`, `top_p?`, `frequency_penalty?`, `presence_penalty?`, `stop?`, `seed?`

**DocumentIngestRequest** (`app/schemas/document.py`):
`filename`, `text`, `source?`, `document_type?`, `tags?`, `user_id?`, `section_heading?`

**ToolResult** (`app/mcp_server/tools/base.py`):
`ok: bool`, `content: str`

---

*Last updated: 2026-05-25*
