# groq-stream-fastapi

A learning-focused FastAPI project that demonstrates how to build a real-time streaming LLM API using Groq’s OpenAI-compatible API, with async PostgreSQL persistence managed by Alembic.

This project was originally built step by step to understand how streaming, async database access, and clean backend architecture work together. It has since evolved into a more advanced architecture that includes **remote tool execution via MCP (Model Context Protocol)**, while still keeping the code readable and educational.

It remains intentionally simple, explicit, and structured, making it suitable as a first serious backend project — now extended to demonstrate **distributed agent patterns**.

---

## What this project demonstrates (in plain terms)

This project shows how to:

* **Build async FastAPI endpoints**
* **Stream LLM responses** token-by-token using `StreamingResponse`
* **Integrate Groq’s OpenAI-compatible LLM API**
* **Persist data** (prompt + response + metadata) into PostgreSQL
* **Use async SQLAlchemy** correctly
* **Manage database schema changes** with Alembic
* **Separate responsibilities cleanly:**

  * API layer
  * Service / orchestration layer
  * LLM client
  * Database access
* **Add remote tool orchestration**
* **Use MCP-based architecture**
* **Design distributed agent systems**
* **Isolate tools into a separate service**

---

## New Architecture: MCP Integration

This project now includes an MCP-based architecture.

**MCP (Model Context Protocol)** is a simple abstraction layer that allows an LLM-driven service to discover and execute tools through a standard interface.

In this project:

* The **Chat API no longer owns or runs tools**
* Tools live inside a separate **MCP Server**
* ChatService calls tools through a **Remote MCP Client**
* Tool execution happens over HTTP

### Why MCP was added

To demonstrate:

* Remote tool execution
* Clean separation of responsibilities
* Distributed agent design
* Microservice-ready architecture

### Tool ownership separation

Before:

* ChatService had direct access to tools.

Now:

* Tools belong to the MCP server.
* ChatService only communicates via MCP.
* ToolRegistry lives only in the MCP service.

This makes the system:

* Easier to scale
* Easier to isolate
* Safer to evolve
* Suitable for multi-agent systems

---

## Updated High-level Request Flow (important)

Understanding this flow is the key to understanding the project:

```text
Client
  |
  | POST /api/v1/chat/stream
  |
  v
FastAPI Endpoint
  |
  v
ChatService
  |
  |----> Groq Streaming LLM
  |
  |----> MCP Client
           |
           v
        MCP Server
           |
           v
        Tools (Calculator, WebSearch)
  |
  v
Async PostgreSQL (chat_logs)
```

Streaming and database persistence still happen safely and asynchronously.

The only difference is that **tools are now executed remotely**.

---

## Project Structure

```text
app/
├─ main.py
├─ api/
│  └─ v1/
│     └─ endpoints/
│        └─ chat.py
├─ services/
│  ├─ groq_client.py
│  ├─ chat_service.py
│  └─ mcp/
│     ├─ client.py
│     ├─ remote_client.py
├─ mcp_server/
│  ├─ main.py
│  ├─ routes.py
│  └─ tools/
│     ├─ base.py
│     ├─ calculator.py
│     ├─ web_search.py
│     └─ registry.py
├─ db/
│  ├─ base.py
│  ├─ engine.py
│  ├─ session.py
│  ├─ models/
│  │  └─ chat_log.py
│  └─ repositories/
│     └─ chat_log.py
├─ schemas/
│  └─ chat.py
├─ core/
│  └─ config.py
alembic/
docker-compose.yml
Dockerfile
```

Key change:

* Tools now live under `app/mcp_server/tools/`
* Chat API is **tool-agnostic**
* ToolRegistry belongs to MCP server

---

## Setup (Docker-based, recommended)

### 1. Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_DEFAULT_MODEL=llama-3.3-70b-versatile

DATABASE_URL=postgresql+asyncpg://app:app@db:5432/app

MCP_SERVER_URL=http://mcp:8001
```

### 2. Build and run with Docker

```bash
docker compose up --build
```

This starts:

* **api** → FastAPI chat backend
* **mcp** → MCP server (tool execution)
* **db** → PostgreSQL database

---

## Streaming Chat Endpoint

**Endpoint:** `POST /api/v1/chat/stream`

### Request body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain async Python briefly"
    }
  ]
}
```

* Uses OpenAI-compatible chat format
* Additional parameters like model, temperature, etc. are optional

### Important note

* The Chat API does **not** execute tools directly.
* Tool calls are routed through the MCP server.
* ChatService communicates with tools via RemoteMCPClient.

### Test with curl (recommended)

Swagger UI does not display streaming responses correctly. Use curl instead:

```bash
curl -N http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  --data-raw '{
    "messages": [
      {"role": "user", "content": "Hello, explain async Python briefly"}
    ]
  }'
```

**Expected behavior:**

* Response arrives incrementally
* Output appears chunk by chunk
* Stream ends with: `data: [DONE]`

---

## MCP Server Endpoints

The MCP server exposes tools over HTTP.

### List available tools

```
GET /tools
```

Response:

```json
{
  "tools": [...]
}
```

### Call a tool

```
POST /tools/call
```

Example:

```json
{
  "name": "calculator",
  "arguments": { "expression": "5*9" }
}
```

Response:

```json
{ "result": "5*9 = 45" }
```

---

## Database Persistence

Each completed chat interaction is saved in PostgreSQL:

* Prompt (input)
* Final response (output)
* Model name
* Timestamp

**Example table query:**

```sql
SELECT id, prompt, model_name, created_at
FROM chat_logs
ORDER BY id DESC;
```

---

## Why this architecture?

This project intentionally avoids shortcuts:

* **LLM client** does not touch the database.
* **Endpoint** does not contain business logic.
* **Database writes** happen only after streaming completes to ensure data integrity.
* **Async sessions** are passed explicitly.

New architectural goals:

* **Tool ownership separation**
* **Microservice-friendly design**
* **Fail-soft remote execution**
* **Streaming-safe tool calls**
* **Distributed agent readiness**

By moving tools to a separate MCP service:

* Chat API becomes lighter
* Tools can scale independently
* New tools can be added without touching ChatService
* Multiple APIs can share the same MCP server

---

## Recommended Models

* **For development and debugging:** `llama-3.3-8b-instant`
* **For higher-quality responses:** `llama-3.3-70b-versatile`

---

## Optional: Gradio Demo UI (Development Only)

For local development and experimentation, this project can be used with a simple **Gradio-based UI** that connects to the streaming chat endpoint.

The Gradio UI is **not part of the backend architecture** and is intentionally kept separate:

* It runs as a local Python process (outside Docker)
* It does not add any backend state or persistence
* It sends the full chat history with each request (client-managed context)
* The FastAPI backend remains fully stateless

Conceptual flow:

```text
Gradio UI (local)
  |
  | POST /api/v1/chat/stream
  | (full messages array)
  v
FastAPI Streaming Endpoint
```

Tool calls triggered during chat will be executed remotely via MCP.

---

## Who this project is for

* Developers learning **async Python**
* Developers learning **LLM streaming**
* Developers learning **FastAPI + PostgreSQL**
* Developers exploring **tool-augmented LLMs**
* Developers interested in **distributed agent architectures**

This repository is meant to be read, modified, and learned from.

---

## License

This project is provided for educational and demonstration purposes.
