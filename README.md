# groq-stream-fastapi

A learning-focused FastAPI project that demonstrates how to build a production-grade streaming LLM backend with tool use, a private RAG knowledge base, multi-provider model support, and full observability.

The project was built step by step to understand how streaming, async database access, tool-augmented generation, and retrieval-augmented generation work together in a real backend. The code is intentionally readable and explicit, making it suitable as a reference for anyone building their first serious LLM backend.

---

## What this project demonstrates

- **Streaming LLM responses** token-by-token via Server-Sent Events (SSE)
- **OpenAI-compatible chat API** (drop-in for any OpenAI client)
- **Tool-augmented generation** — the model can call a calculator, search the web, or query uploaded documents
- **Distributed tool execution** — tools live in a separate service; the chat API never owns or runs them
- **Private RAG knowledge base** — upload PDFs, chunk and embed them, search with pgvector
- **Hybrid search** — dense vector similarity + sparse BM25 full-text search merged with Reciprocal Rank Fusion
- **Optional reranking** — cross-encoder rescoring via the Jina API
- **Multi-provider LLM** — Groq, OpenRouter, and Google Gemini behind a single interface
- **Automatic fallback chain** — switches models on rate limits without any client changes
- **Conversation history** — persistent, multi-turn context stored in PostgreSQL
- **LangSmith tracing** — every LLM call, tool call, and RAG result recorded as child spans
- **Eval framework** — tool-routing accuracy measured against a 30-question LangSmith dataset
- **Async-first design** — all I/O is non-blocking; no thread pools, no sync ORM calls
- **Alembic migrations** — schema versioned and reproducible

---

## Architecture

Three Docker services:

| Service | Port | Responsibility |
|---|---|---|
| `api` | 8000 | FastAPI chat backend — handles requests, orchestrates LLM calls, persists history |
| `tool-server` | 8001 | Tool execution service — calculator, web search, RAG search |
| `db` | 5432 | PostgreSQL 15 with the pgvector extension |

The chat API never executes tools locally. It discovers available tools and calls them over HTTP. This means tools can be updated, replaced, or scaled independently.

---

## Request flow

Understanding this flow is the key to understanding the project.

```
Client
  |
  | POST /api/v1/chat/stream
  |
  v
FastAPI Endpoint (chat.py)
  |  - validate model
  |  - resolve or generate conversation_id
  |
  v
ChatService.stream_chat()
  |
  |-- 1. Load last 5 history turns from DB
  |-- 2. Fetch tool schemas from tool-server (/tools)
  |-- 3. Choose routing prompt
  |       - conversation has uploaded docs → "always call rag_search for factual questions"
  |       - no docs → standard tool-routing rules
  |
  |-- 4. LLM call 1 — ROUTING (temperature=0, max_tokens=256)
  |       The model decides whether a tool is needed. It outputs no text — only tool calls.
  |
  |         ┌── tool call(s) returned ─────────────────────────────────────────────┐
  |         │                                                                       │
  |         │  Execute each tool via RemoteToolClient (HTTP POST /tools/call)       │
  |         │    rag_search  → embed query → pgvector cosine + optional BM25 + rerank │
  |         │    web_search  → Tavily API                                           │
  |         │    calculator  → safe ast-based expression evaluator                 │
  |         │                                                                       │
  |         │  Inject tool results into message history                             │
  |         │  Re-inject last RAG result as a system message (prevents model skipping it) │
  |         │  Apply context budget (trim RAG payload → evict old history → truncate) │
  |         │                                                                       │
  |         │  LLM call 3 — FINALIZATION (no tools, user temperature)              │
  |         │    Model synthesizes a final answer from tool results.               │
  |         └───────────────────────────────────────────────────────────────────── │
  |
  |         ┌── no tool call ──────────────────────────────────────────────────────┐
  |         │  LLM call 2 — DIRECT ANSWER (no tools, user temperature)            │
  |         │    Model answers from its own knowledge. Routing output is discarded. │
  |         └─────────────────────────────────────────────────────────────────────-│
  |
  |         ┌── Groq failed_generation or <|python_tag|> leak ─────────────────────┐
  |         │  Parse intended tool + args from the leak text (regex + JSON).        │
  |         │    detected  → call that tool directly, inject result as system msg   │
  |         │    not found → no automatic tool call; LLM call 2 runs without tools  │
  |         └──────────────────────────────────────────────────────────────────────│
  |
  |-- 5. Stream response to client (OpenAI-compatible SSE chunks)
  |-- 6. Persist ChatLog to DB (full messages JSONB, model, params, user_id, turn_index)
  |
  v
data: {"choices": [{"delta": {"content": "..."}}]}
...
data: [DONE]
```

Key design rules embedded in this flow:

- The routing call always runs at `temperature=0` for determinism. User temperature only applies to the final answer.
- `rag_search` arguments (`user_id`, `conversation_id`) are always injected server-side. The model never controls whose documents are queried.
- `<think>...</think>` blocks are stripped before tokens reach the client (Qwen3 reasoning models).
- The agentic loop runs for at most 8 rounds and 10 total tool calls.

---

## Project structure

```
groq-stream-fastapi/
│
├── app/
│   ├── main.py                          # FastAPI app, lifespan, CORS, health check
│   │
│   ├── api/v1/
│   │   ├── router.py
│   │   └── endpoints/
│   │       ├── chat.py                  # POST /chat/stream, GET /chat/models
│   │       ├── documents.py             # POST /documents/upload, CRUD
│   │       ├── eval.py                  # POST /eval/route — lightweight routing test
│   │       └── rag_metrics.py           # GET /rag/metrics
│   │
│   ├── services/
│   │   ├── chat_service.py              # Main orchestrator — routing, tools, finalization
│   │   ├── groq_client.py               # LLM streaming client (Groq, OpenRouter, Gemini)
│   │   ├── ingestion_service.py         # PDF upload → chunk → embed → persist
│   │   ├── embeddings.py                # Embedding service with SHA-256 LRU cache
│   │   ├── chunking.py                  # Recursive token-aware text chunker
│   │   ├── pdf_extractor.py             # PyMuPDF-based PDF parser with heading detection
│   │   ├── reranker.py                  # Optional Jina cross-encoder reranker
│   │   ├── rag_metrics.py               # In-memory retrieval health tracking
│   │   ├── tracing.py                   # LangSmith span helpers
│   │   └── tool_client/
│   │       ├── client.py                # Abstract ToolClient interface
│   │       └── remote_client.py         # HTTP-based tool discovery and execution
│   │
│   ├── tool_server/                     # Separate FastAPI service (port 8001)
│   │   ├── main.py
│   │   ├── routes.py                    # GET /tools, POST /tools/call, GET /metrics
│   │   └── tools/
│   │       ├── base.py                  # Abstract Tool + ToolResult
│   │       ├── registry.py              # ToolRegistry
│   │       ├── calculator.py            # Safe ast-based expression evaluator
│   │       ├── web_search.py            # Tavily API integration
│   │       └── rag_search.py            # pgvector + hybrid BM25 + optional reranking
│   │
│   ├── db/
│   │   ├── engine.py                    # SQLAlchemy async engine
│   │   ├── session.py                   # AsyncSessionLocal, get_db dependency
│   │   ├── models/
│   │   │   ├── chat_log.py              # ChatLog — prompt, response, messages JSONB, metadata
│   │   │   ├── document.py              # Document — file metadata, tags, user_id
│   │   │   └── document_chunk.py        # DocumentChunk — text, pgvector(768), page, heading
│   │   └── repositories/
│   │       ├── chat_log.py              # Chat history CRUD
│   │       └── document.py              # Document CRUD + hybrid vector search
│   │
│   ├── schemas/
│   │   ├── chat.py                      # ChatStreamRequest, ChatMessage
│   │   └── document.py                  # DocumentIngestRequest/Response/Read
│   │
│   ├── core/
│   │   ├── config.py                    # Settings (BaseSettings), AVAILABLE_MODELS, FALLBACK_CHAIN
│   │   ├── logging.py                   # structlog setup
│   │   └── chunking_config.py           # TIKTOKEN_ENCODING constant
│   │
│   └── utils/
│       └── token_counter.py             # Token counting and text truncation (tiktoken cl100k_base)
│
├── alembic/
│   └── versions/                        # 15 migration files
│
├── eval/
│   ├── run_eval.py                      # Tool-routing accuracy evaluation
│   ├── run_multidoc_source_test.py      # Multi-document RAG attribution test
│   └── upload_dataset.py               # LangSmith dataset management
│
├── tests/
│   ├── unit/                            # Chunking, embeddings, PDF, token counter
│   ├── integration/                     # Embedding service, ingestion pipeline
│   └── e2e/                             # Document upload + RAG metrics API
│
├── scripts/                             # Exploratory and debug scripts
├── ui_gradio.py                         # Gradio chat UI (local development only)
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

---

## Setup

### 1. Create a `.env` file

Copy `.env.example` and fill in the required values. The variables are grouped below by category.

**Required:**

```env
# Primary LLM provider
GROQ_API_KEY=your_groq_api_key

# Database
DATABASE_URL=postgresql+asyncpg://app:app@db:5432/app

# Tool server (set to service name when running via Docker Compose)
TOOL_SERVER_URL=http://tool-server:8001

# Embeddings — point to an OpenAI-compatible embedding API
EMBEDDING_BASE_URL=http://your-embedding-server/v1
EMBEDDING_API_KEY=your_embedding_api_key
```

**Optional providers (used in fallback chain):**

```env
OPENROUTER_API_KEY=your_openrouter_api_key
GEMINI_API_KEY=your_gemini_api_key
```

**Web search:**

```env
TAVILY_API_KEY=your_tavily_api_key
```

**RAG settings:**

```env
RAG_DEFAULT_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.7
HYBRID_SEARCH_ENABLED=true
RERANKER_ENABLED=false
RERANKER_API_KEY=your_jina_api_key      # only needed if reranker is enabled
```

**Feature flags:**

```env
WEB_SEARCH_ENABLED=true
CALCULATOR_ENABLED=true
RAG_SYSTEM_PROMPT_ENABLED=true
```

**LangSmith tracing (optional):**

```env
LANGSMITH_TRACING_ENABLED=false
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=groq-stream-fastapi
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

This starts three containers:

- **api** — FastAPI chat backend on `localhost:8000`
- **tool-server** — tool execution service on `localhost:8001`
- **db** — PostgreSQL + pgvector on `localhost:5432`

### 3. Apply database migrations

```bash
docker compose exec api alembic upgrade head
```

---

## API reference

### Chat

#### `POST /api/v1/chat/stream`

Streams a chat completion as OpenAI-compatible SSE chunks.

Request body:

```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "conversation_id": "abc123",
  "user_id": "user-42",
  "model": null,
  "temperature": 0.0,
  "max_tokens": null,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "seed": null,
  "tags": []
}
```

All fields except `messages` are optional.

- `conversation_id` — if provided, the last 5 turns of history are loaded and prepended. If omitted, a new UUID is generated server-side.
- `user_id` — used to scope RAG search to the user's uploaded documents.
- `model` — must be a key in `AVAILABLE_MODELS`. Defaults to `GROQ_DEFAULT_MODEL`. If explicitly set, automatic fallback is disabled.
- `tags` — forwarded to LangSmith as trace tags.

Use `curl` to test (Swagger does not render SSE correctly):

```bash
curl -N http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 18% of 1250?"}],
    "user_id": "user-42"
  }'
```

#### `GET /api/v1/chat/models`

Returns the list of available models in OpenAI-compatible format.

---

### Documents

#### `POST /api/v1/documents/upload`

Uploads a PDF file, processes it through the ingestion pipeline, and makes it searchable via `rag_search`.

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf" \
  -F "user_id=user-42" \
  -F "conversation_id=abc123" \
  -F 'tags=["finance","q3"]'
```

#### `GET /api/v1/documents`

Lists all documents. Supports filtering by `user_id`, `tags`, and pagination via `limit` / `offset`.

#### `GET /api/v1/documents/{id}`

Returns metadata for a single document.

#### `PUT /api/v1/documents/{id}`

Updates document metadata (filename, tags, etc.).

#### `DELETE /api/v1/documents/{id}`

Deletes a document and all its chunks (cascade).

#### `POST /api/v1/documents/{id}/reprocess`

Re-embeds a document with updated text or section heading, replacing existing chunks.

---

### Other endpoints

#### `GET /api/v1/rag/metrics`

Returns RAG retrieval health metrics: average similarity score, average embedding and pgvector query latency, embedding API error rate, total retrieval call count, and document/chunk counts from the database.

#### `POST /api/v1/eval/route`

Lightweight routing test endpoint. Sends a message through the routing LLM call only and returns which tool (if any) was selected. Used by the eval framework.

#### `GET /health`

Basic liveness check. Returns `{"status": "ok"}`.

---

### Tool server

#### `GET /tools`

Returns all registered tools in OpenAI function-calling schema format.

#### `POST /tools/call`

Executes a single tool.

```json
{"name": "calculator", "arguments": {"expression": "18 * 1250 / 100"}}
```

```json
{"name": "web_search", "arguments": {"query": "current EUR/USD rate"}}
```

```json
{"name": "rag_search", "arguments": {"query": "Q3 revenue", "metadata_filter": {"user_id": "user-42"}}}
```

#### `GET /metrics`

Returns the same RAG metrics as `/api/v1/rag/metrics` (the API proxies this endpoint).

---

## RAG pipeline

### Ingestion

```
PDF upload
  → PDFExtractor (PyMuPDF)
      - heading levels detected from relative font sizes (H1/H2/H3)
      - repeated header/footer lines removed across pages
  → chunk_document()
      - text normalized, metadata sections (FUNDING, COI etc.) extracted as
        separate chunks, code blocks and tables kept intact
      - heading-aware: heading text prepended to each section body
      - oversized prose split: paragraph -> line -> sentence -> token window
      - noise chunks (heading-only, page numbers) dropped
      - configurable: chunk_size_tokens (default 300), overlap (default 75)
      - PDF pages merged before chunking so overlap can span page boundaries
  → EmbeddingService.embed_batch()
      - each chunk embedded with context_prefix prepended
      - nomic-embed-text 768-dim via OpenAI-compatible API
      - in-memory LRU cache; all-or-nothing (fail = document not persisted)
  → persist Document + N x DocumentChunk to PostgreSQL
```

### Retrieval (rag_search tool)

```
User question
  → embed query
  → dense search: pgvector cosine similarity (top_k, similarity_threshold)
  → hybrid search (if HYBRID_SEARCH_ENABLED=true):
      - sparse: PostgreSQL tsvector full-text search
      - merge both result sets with Reciprocal Rank Fusion (RRF, k=60)
      - note: similarity_threshold is not applied in hybrid mode (RRF scores are on a different scale)
  → reranking (if RERANKER_ENABLED=true and RERANKER_API_KEY is set):
      - overfetch_multiplier × top_k candidates fetched, then rescored
        by a Jina/Cohere-compatible cross-encoder (/rerank API)
      - default model: jina-reranker-v2-base-multilingual
      - on reranker failure, falls back to original retrieval order silently
  → format: chunk texts joined with "---" separators + source metadata
```

Retrieval is always scoped to the authenticated user via `user_id` injected server-side. The model has no way to change which user's documents are searched.

---

## Tools

### `calculator`

Evaluates arithmetic expressions safely using Python's `ast` module — no `eval()`. Supports standard operators and numeric literals.

```
Input:  "18 * 1250 / 100"
Output: "18 * 1250 / 100 = 225.0"
```

### `web_search`

Searches the web via the Tavily API and returns ranked results with titles, URLs, and content snippets.

### `rag_search`

Queries the user's uploaded documents using hybrid vector + full-text search. Returns the most relevant chunks with source filenames. Can be disabled per-conversation or at the service level.

### Tool routing rules

The routing system prompt instructs the model:

1. **`rag_search`** — when the question refers to the user's private documents, or when the conversation has uploaded docs (in doc mode, called for every factual question).
2. **`web_search`** — when the answer could have changed recently: prices, exchange rates, software versions, current events, role holders.
3. **`calculator`** — for any arithmetic the user explicitly asks to compute. Never compute in the model's head.
4. **No tool** — static general knowledge, geography, history, definitions, conversational messages.

---

## Multi-provider LLM and fallback chain

The `LLMClient` resolves the provider from the `provider` field in `AVAILABLE_MODELS` (`app/core/config.py`) at request time. Provider is not inferred from the model name — it is explicitly configured per model.

As a practical guide to what's in the registry:

| Provider | Pattern | Notes |
|---|---|---|
| Groq | no suffix (e.g. `llama-3.3-70b-versatile`, `qwen/qwen3-32b`, `openai/gpt-oss-120b`) | Primary provider |
| OpenRouter | `:free` suffix (e.g. `openai/gpt-oss-120b:free`) | Free-tier fallback models |
| Google Gemini | `gemini-` prefix (e.g. `gemini-2.5-flash`) | Direct API; disabled on free tier from some regions |

When a request hits a rate limit:

- **RPM (per-minute)**: `retry_after ≤ 60s` → wait and retry the same model.
- **RPD (daily)**: `retry_after > 60s` or missing → switch to the next model in the fallback chain.

Default fallback chain:

```
llama-3.3-70b-versatile          (Groq)
  → meta-llama/llama-3.3-70b-instruct:free   (OpenRouter)
  → openai/gpt-oss-120b:free                 (OpenRouter)
  → google/gemma-4-31b-it:free               (OpenRouter)
  → google/gemma-4-26b-a4b-it:free           (OpenRouter)
  → qwen/qwen3-32b                           (Groq)
  → llama-3.1-8b-instant                     (Groq)
```

Automatic fallback is disabled when the client explicitly sets the `model` field.

---

## Conversation history

Each completed turn is persisted to the `chat_logs` table with:

- `messages` — full message array including system prompts, tool calls, and tool results (JSONB)
- `conversation_id` — links turns into a conversation thread
- `turn_index` — position within the conversation
- `user_id` — used to scope history and document access
- `prompt`, `response` — plain-text copies for quick querying
- LLM parameters: `model_name`, `temperature`, `max_tokens`, `top_p`, `seed`

When a request includes a `conversation_id`, the last 5 turns are loaded and prepended to the current messages before any LLM call is made.

---

## LangSmith tracing

When `LANGSMITH_TRACING_ENABLED=true`, every `stream_chat` call creates a root trace with child spans for:

- Each LLM call (routing, direct answer, finalization, fallback)
- Each tool call
- Token usage per call

Spans include model name, round number, tool names, prompt/completion token counts, and total elapsed time.

---

## Evaluation framework

```bash
# Run tool-routing accuracy evaluation
python eval/run_eval.py

# Run multi-document RAG source attribution test
python eval/run_multidoc_source_test.py
```

`run_eval.py` sends 30 questions from the `tool-routing-eval-v2` LangSmith dataset through `POST /eval/route` and measures what percentage of questions are routed to the correct tool. Results are logged back to LangSmith with per-example correctness scores.

The eval model is set via the `MODEL` constant at the top of `eval/run_eval.py`. It is kept separate from the chat app's primary model to avoid consuming the same rate-limit quota during evaluation runs.

---

## Guardrails

The app ships a pre-flight `PromptInjectionGuard` (`app/services/guardrails.py`, powered by `meta-llama/llama-4-scout-17b-16e-instruct` via `settings.guard_model`) that screens incoming user messages for jailbreak / prompt-injection attempts before they reach the chat pipeline.

Three risk categories — prompt injection, harmful content, and hallucination — were each tested by running the real model(s) through a hand-written question set, recording the responses, and only adding a guard where the underlying model's own behavior left a real gap. The full methodology, test scripts, and per-question results are documented in [`guardrail.md`](guardrail.md):

- **Prompt injection** — guard added: the raw model leaked its system prompt and internal tool-call syntax on several prompts; Scout reliably catches these before the chat pipeline runs.
- **Harmful content** — no guard added: the primary model's built-in alignment already refuses direct harmful requests cleanly (8/8).
- **Hallucination** — no guard added: a full-pipeline RAG baseline (real ingestion → retrieval → generation, no guard) showed the primary model produced zero observed hallucinations across grounded, fabricated, distorted, and adversarial "hard hallucination" question sets — even though a guard classifier for this category is feasible (Scout scored 28/28 in isolation), the model it would protect doesn't currently need it.

Test scripts live in `scripts/guard/` and follow a shared convention: each question is sent as a fresh, history-free conversation, rate-limited between calls, with raw responses saved to `scripts/guard/responses/`.

---

## Local development (without Docker)

Run the API and tool server directly with hot reload:

```bash
# Terminal 1 — API server (port 8000)
python -m app.main

# Terminal 2 — Tool server (port 8001)
python -m app.tool_server.main
```

Both servers start with `--reload` enabled by default.

---

## Gradio UI (development only)

A minimal Gradio chat interface is included for local testing:

```bash
pip install gradio
python ui_gradio.py
```

The UI sends the full chat history with each request (client-managed context) and connects to the streaming endpoint at `http://localhost:8000/api/v1/chat/stream`. It is not part of the backend architecture and is not run inside Docker.

---

## Database migrations

Migrations are managed with Alembic and must be applied manually:

```bash
alembic upgrade head     # apply all pending migrations
alembic downgrade -1     # roll back the latest migration
```

There are 15 migrations covering: chat logs, documents and chunks, pgvector HNSW index, hybrid search tsvector column, embedding dimension changes, conversation tracking fields, and evaluation tables.

---

## Recommended models

| Use case | Model |
|---|---|
| Default (quality) | `llama-3.3-70b-versatile` |
| Balanced / general use | `openai/gpt-oss-120b` |
| Fast / low latency | `llama-3.1-8b-instant` |
| Long context | `gemini-2.5-flash` |

---

## License

This project is provided for educational and demonstration purposes.
