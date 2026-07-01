# groq-stream-fastapi

A learning-focused FastAPI project that demonstrates how to build a production-grade streaming LLM backend with tool use, a private RAG knowledge base, multi-provider model support, and full observability.

The project was built step by step to understand how streaming, async database access, tool-augmented generation, and retrieval-augmented generation work together in a real backend. The code is intentionally readable and explicit, making it suitable as a reference for anyone building their first serious LLM backend.

---

## What this project demonstrates

- **Streaming LLM responses** token-by-token via Server-Sent Events (SSE)
- **OpenAI-compatible chat API** (drop-in for any OpenAI client)
- **Tool-augmented generation** — the model can search the web or query uploaded documents via tool calls
- **Distributed tool execution** — tools live in a separate service; the chat API never owns or runs them
- **Private RAG knowledge base** — upload PDFs, chunk and embed them, search with pgvector
- **Hybrid search** — dense vector similarity + BM25-style keyword search (PostgreSQL tsvector) + grep, merged with Reciprocal Rank Fusion
- **Optional reranking** — cross-encoder rescoring via the Jina API
- **Multi-provider LLM** — Groq, OpenRouter, and SambaNova behind a single interface
- **Prompt injection guardrail** — pre-flight classifier screens user messages before the pipeline runs
- **Conversation history** — persistent, multi-turn context stored in PostgreSQL
- **LangSmith tracing** — every LLM call, tool call, and RAG result recorded as child spans
- **RAG evaluation** — per-leg retrieval diagnostics and RAGAS LLM-as-judge pipeline
- **Async-first design** — all I/O is non-blocking; no thread pools, no sync ORM calls
- **Alembic migrations** — schema versioned and reproducible

---

## Architecture

Three Docker services:

| Service | Port | Responsibility |
|---|---|---|
| `api` | 8000 | FastAPI chat backend — handles requests, orchestrates LLM calls, persists history |
| `tool-server` | 8001 | Tool execution service — web search, RAG search |
| `db` | 5432 | PostgreSQL 15 with the pgvector extension |

The chat API never executes tools locally. It discovers available tools and calls them over HTTP. This means tools can be updated, replaced, or scaled independently.

---

## Setup

### 1. Create a `.env` file

Copy `.env.example` and fill in the required values.

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

**Optional providers:**

```env
OPENROUTER_API_KEY=your_openrouter_api_key
SAMBANOVA_API_KEY=your_sambanova_api_key
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
RAG_SYSTEM_PROMPT_ENABLED=true
```

**LangSmith tracing (optional):**

```env
LANGSMITH_TRACING_ENABLED=false
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=groq-stream-fastapi
```

### 2. Run

**Option A — Docker Compose (recommended)**

```bash
docker compose up --build
```

Starts three containers: `api` on `localhost:8000`, `tool-server` on `localhost:8001`, `db` on `localhost:5432`.

**Option B — Local (without Docker)**

Requires a running PostgreSQL instance with pgvector. Set `TOOL_SERVER_URL=http://localhost:8001` in `.env`.

```bash
# Terminal 1 — API server (port 8000)
python -m app.main

# Terminal 2 — Tool server (port 8001)
python -m app.tool_server.main
```

Both servers start with `--reload` enabled by default. A minimal Gradio chat UI is also available for local testing:

```bash
pip install gradio
python ui_gradio.py
```

### 3. Apply database migrations

```bash
# Docker
docker compose exec api alembic upgrade head

# Local
alembic upgrade head
```

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
  |-- 2. Prompt injection guard (PromptInjectionGuard)
  |       Screens the incoming user message before the pipeline runs.
  |       If flagged → stream error message, stop.
  |
  |-- 3. Fetch tool schemas from tool-server (/tools)
  |-- 4. Choose routing prompt
  |       - conversation has uploaded docs → "always call rag_search for factual questions"
  |       - no docs → standard tool-routing rules
  |
  |-- 5. LLM call 1 — ROUTING (temperature=0, max_tokens=256)
  |       The model decides whether a tool is needed. It outputs no text — only tool calls.
  |
  |         ┌── tool call(s) returned ──────────────────────────────────────────────────┐
  |         │                                                                            │
  |         │  Execute each tool via RemoteToolClient (HTTP POST /tools/call)            │
  |         │    rag_search  → embed query → pgvector + BM25-style tsvector + grep + rerank │
  |         │    web_search  → Tavily API                                                │
  |         │                                                                            │
  |         │  Inject tool results into message history                                  │
  |         │  Re-inject last RAG result as a system message (prevents model skipping it)│
  |         │  Apply context budget (evict old history → trim RAG payload → truncate)    │
  |         │                                                                            │
  |         │  LLM call 3 — FINALIZATION (no tools, user temperature)                   │
  |         │    Model synthesizes a final answer from tool results.                    │
  |         └────────────────────────────────────────────────────────────────────────── │
  |
  |         ┌── no tool call ───────────────────────────────────────────────────────────┐
  |         │  LLM call 2 — DIRECT ANSWER (no tools, user temperature)                 │
  |         │    Model answers from its own knowledge. Routing output is discarded.     │
  |         └───────────────────────────────────────────────────────────────────────── -│
  |
  |         ┌── Groq failed_generation or <|python_tag|> leak ──────────────────────────┐
  |         │  Parse intended tool + args from the leak text (regex + JSON).             │
  |         │    detected  → call that tool directly, inject result as system msg        │
  |         │    not found → no automatic tool call; LLM call 2 runs without tools       │
  |         └───────────────────────────────────────────────────────────────────────────│
  |
  |-- 6. Stream response to client (OpenAI-compatible SSE chunks)
  |-- 7. Persist ChatLog to DB (full messages JSONB, model, params, user_id, turn_index)
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

### Trying it out

Swagger does not render SSE correctly — use `curl` with the `-N` flag to stream responses.

**Direct answer** (no tool call — model answers from its own knowledge):

```bash
curl -N http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the difference between REST and GraphQL?"}],
    "user_id": "user-42"
  }'
```

**RAG query** (upload a document first, then query it in the same conversation):

```bash
# 1. Upload a PDF
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf" \
  -F "user_id=user-42" \
  -F "conversation_id=conv-1"

# 2. Ask a question — rag_search is called automatically
curl -N http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What are the key findings in the report?"}],
    "user_id": "user-42",
    "conversation_id": "conv-1"
  }'
```

---

## Guardrails

A `PromptInjectionGuard` (`app/services/guardrails.py`) screens incoming user messages before they reach the routing pipeline. It uses `meta-llama/llama-4-scout-17b-16e-instruct` (configured via `settings.guard_model`) and expects a YES/NO response.

Three risk categories were evaluated — prompt injection, harmful content, and hallucination — by running the underlying model through a hand-written question set and recording raw responses. A guard was only added where the model's own behavior left a real gap:

- **Prompt injection** — guard added: the model leaked its system prompt and internal tool-call syntax on several prompts without intervention; Scout catches these reliably before the pipeline runs.
- **Harmful content** — no guard added: the primary model's built-in alignment already refuses direct harmful requests cleanly.
- **Hallucination** — no guard added: end-to-end RAG testing showed zero observed hallucinations across grounded, fabricated, distorted, and adversarial question sets.

The full methodology and per-question results are in [`docs/guardrail.md`](docs/guardrail.md). Test scripts live in `scripts/guard/`.

---

## Tools

### `calculator`

Evaluates arithmetic expressions safely using Python's `ast` module — no `eval()`.

### `web_search`

Searches the web via the Tavily API and returns ranked results with titles, URLs, and content snippets.

### `rag_search`

Queries the user's uploaded documents using hybrid vector + full-text search. Returns the most relevant chunks with source filenames. Can be disabled per-conversation or at the service level.

### Tool routing rules

The routing system prompt instructs the model:

1. **`rag_search`** — when the question refers to the user's private documents, or when the conversation has uploaded docs (in doc mode, called for every factual question).
2. **`web_search`** — when the answer could have changed recently: prices, exchange rates, software versions, current events, role holders.
3. **`calculator`** — any arithmetic the user explicitly asks to compute.
4. **No tool** — static general knowledge, geography, history, definitions, conversational messages.

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
      - oversized prose split: paragraph → line → sentence → token window
      - noise chunks (heading-only, page numbers) dropped
      - configurable: chunk_size_tokens (default 300), overlap (default 75)
      - PDF pages merged before chunking so overlap can span page boundaries
  → EmbeddingService.embed_batch()
      - each chunk embedded with context_prefix prepended
      - nomic-embed-text 768-dim via OpenAI-compatible API
      - in-memory LRU cache; all-or-nothing (fail = document not persisted)
  → persist Document + N x DocumentChunk to PostgreSQL
```

### Retrieval (`rag_search` tool)

```
User question
  → embed query
  → if HYBRID_SEARCH_ENABLED=true (default):
      Three search legs run independently, then their ranked result sets are
      merged with Reciprocal Rank Fusion (RRF, k=60):
        - dense leg:  pgvector cosine similarity on 768-dim embeddings
        - sparse leg: PostgreSQL tsvector ts_rank — keyword matching with
                      OR-based query expansion so a single missing term
                      does not cause the whole leg to return nothing
        - grep leg:   pg_trgm ILIKE substring match — catches structured
                      identifiers (SVC-X-0000, ERR::CODE) and keyword
                      bigrams that tsvector tokenisation fragments
      similarity_threshold is not applied in this mode (RRF scores are on
      a different scale than cosine similarity)
  → if HYBRID_SEARCH_ENABLED=false:
      - dense search only: pgvector cosine similarity
      - similarity_threshold applied to filter low-confidence results
  → reranking (if RERANKER_ENABLED=true and RERANKER_API_KEY is set):
      - overfetch_multiplier × top_k candidates fetched before RRF, then
        rescored by a Jina/Cohere-compatible cross-encoder (/rerank API)
      - default model: jina-reranker-v2-base-multilingual
      - on reranker failure, falls back to original retrieval order silently
  → format: chunk texts joined with "---" separators + source metadata
```

Retrieval is always scoped to the authenticated user via `user_id` injected server-side. The model has no way to change which user's documents are searched.

---

## Multi-provider LLM

The `LLMClient` (`app/services/groq_client.py`) resolves the provider from the `provider` field in `AVAILABLE_MODELS` (`app/core/config.py`) at request time. Provider is not inferred from the model name — it is explicitly configured per model.

| Provider | Pattern | Notes |
|---|---|---|
| Groq | no suffix (e.g. `llama-3.3-70b-versatile`, `qwen/qwen3-32b`) | Primary provider |
| OpenRouter | `:free` suffix (e.g. `openai/gpt-oss-120b:free`) | Free-tier models |
| SambaNova | `Meta-Llama-` prefix (e.g. `Meta-Llama-3.3-70B-Instruct`) | Alternative inference provider |

On rate limits (RPM), the service waits for the retry window and retries the same model. On daily limits (RPD), an error is returned and the user should select a different model.

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

- Each LLM call (routing, direct answer, finalization)
- Each tool call
- Token usage per call

Spans include model name, round number, tool names, prompt/completion token counts, and total elapsed time.

---

## Evaluation

### Tool-routing accuracy (`eval/`)

```bash
python eval/run_eval.py
python eval/run_multidoc_source_test.py
```

`run_eval.py` sends 30 questions from the `tool-routing-eval-v2` LangSmith dataset through `POST /eval/route` and measures what percentage of questions are routed to the correct tool. Results are logged back to LangSmith with per-example correctness scores.

### RAG evaluation (`rag-test/`)

A separate evaluation framework for the retrieval and answer generation pipeline, built iteratively alongside the code.

Retrieval was evaluated first: each RRF leg is tracked independently before fusion — dense (pgvector cosine), sparse (BM25-style tsvector), and grep (ILIKE) scored separately. This per-leg visibility revealed that the default AND-logic of `plainto_tsquery` caused full sparse leg failure when a single query term was absent from a chunk. The fix — OR-based `_build_or_tsquery` in `app/db/repositories/document.py` — is in the current production code.

Question sets are stratified by difficulty across categories: exact match, paraphrase, distractor, multi-chunk, inference, negation, and adversarial.

End-to-end answer quality is measured separately with a RAGAS LLM-as-judge pipeline (`rag-test/ragas/`): Faithfulness, Answer Correctness, Context Recall, and Context Precision. Runs compare two judge models to surface potential judge bias.

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
- `model` — must be a key in `AVAILABLE_MODELS`. Defaults to `GROQ_DEFAULT_MODEL`.
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
{"name": "web_search", "arguments": {"query": "current EUR/USD rate"}}
```

```json
{"name": "rag_search", "arguments": {"query": "Q3 revenue", "metadata_filter": {"user_id": "user-42"}}}
```

#### `GET /metrics`

Returns the same RAG metrics as `/api/v1/rag/metrics` (the API proxies this endpoint).

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
│   │   ├── groq_client.py               # LLM client (Groq, OpenRouter, SambaNova)
│   │   ├── guardrails.py                # PromptInjectionGuard
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
│   │       └── rag_search.py            # pgvector + hybrid tsvector + grep + optional reranking
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
│   │   ├── config.py                    # Settings (BaseSettings), AVAILABLE_MODELS
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
├── rag-test/
│   ├── sherlock/                        # Retrieval tests on an unstructured literary PDF
│   ├── techcorp/                        # Retrieval tests on a structured policy document
│   └── ragas/                           # RAGAS LLM-as-judge answer quality evaluation
│
├── tests/
│   ├── unit/                            # Chunking, embeddings, PDF, token counter
│   ├── integration/                     # Embedding service, ingestion pipeline
│   └── e2e/                             # Document upload + RAG metrics API
│
├── scripts/
│   ├── guard/                           # Guardrail test scripts and raw responses
│   ├── upload_document.py               # Upload a document via the API
│   ├── chunk_inspect.py                 # Inspect chunk boundaries for a given PDF
│   ├── pdf_inspect.py                   # Inspect raw PDF extraction output
│   ├── eval_grep_vs_bm25.py             # Compare grep and BM25 retrieval legs
│   └── test_threshold.py               # Threshold sensitivity analysis
│
├── docs/
│   ├── guardrail.md                     # Guardrail design, methodology, and test results
│   ├── guard_model_comparison.md        # Guard model evaluation across providers
│   ├── hybrid_search.md                 # Hybrid search design and grep leg rationale
│   ├── rag_test_results.md              # RAG retrieval test results summary
│   ├── rerank_threshold_analysis.md     # Reranker threshold sensitivity analysis
│   └── sherlock_questions.md            # Hand-written question set used in retrieval tests
│
├── ui_gradio.py                         # Gradio chat UI (local development only)
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

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
| Long context | `qwen/qwen3-32b` |

---

## License

This project is provided for educational and demonstration purposes.
