# groq-stream-fastapi

A learning-focused FastAPI project that demonstrates how to build a real-time streaming LLM API using Groq’s OpenAI-compatible API, with async PostgreSQL persistence managed by Alembic.

This project was built step by step to understand how streaming, async database access, and clean backend architecture work together. It is intentionally simple, explicit, and readable, making it suitable as a first serious backend project.

## What this project demonstrates (in plain terms)

This project shows how to:

*   **Build async FastAPI endpoints**
*   **Stream LLM responses** token-by-token using `StreamingResponse`
*   **Integrate Groq’s OpenAI-compatible LLM API**
*   **Persist data** (prompt + response + metadata) into PostgreSQL
*   **Use async SQLAlchemy** correctly
*   **Manage database schema changes** with Alembic
*   **Separate responsibilities cleanly:**
    *   API layer
    *   Service / orchestration layer
    *   LLM client
    *   Database access

## High-level request flow (important)

Understanding this flow is the key to understanding the project:

```text
Client
  |
  |  POST /api/v1/chat/stream
  |
  v
FastAPI Endpoint
  |
  |  (receives request + DB session)
  v
ChatService
  |
  |  starts LLM streaming
  |  collects final response
  |
  |--> streamed chunks → client (real-time)
  |
  |  when stream finishes:
  v
Async PostgreSQL (chat_logs table)
```

Streaming and database persistence happen in a controlled and safe way.

## Project Structure

```text
app/
├─ main.py                     # FastAPI application entrypoint
├─ api/
│  └─ v1/
│     └─ endpoints/
│        └─ chat.py             # Streaming chat endpoint (SSE)
├─ services/
│  ├─ groq_client.py            # Low-level Groq streaming client
│  └─ chat_service.py           # Orchestrates streaming + DB persistence
├─ db/
│  ├─ base.py                   # SQLAlchemy Base
│  ├─ engine.py                 # Async engine
│  ├─ session.py                # AsyncSession dependency
│  ├─ models/
│  │  └─ chat_log.py            # ChatLog ORM model
│  └─ repositories/
│     └─ chat_log.py            # DB write logic (isolated)
├─ schemas/
│  └─ chat.py                   # Pydantic request schemas
├─ core/
│  └─ config.py                 # Environment-based settings
alembic/
├─ versions/                    # Alembic migrations
alembic.ini
docker-compose.yml
Dockerfile
```

## Setup (Docker-based, recommended)

### 1. Environment variables
Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_DEFAULT_MODEL=llama-3.3-70b-versatile

DATABASE_URL=postgresql+asyncpg://app:app@db:5432/app
```

### 2. Build and run with Docker
```bash
docker compose up --build
```

This starts:
*   FastAPI backend
*   PostgreSQL database
*   Async DB connection between them

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

*   Uses OpenAI-compatible chat format
*   Additional parameters like model, temperature, etc. are optional

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
*   Response arrives incrementally
*   Output appears chunk by chunk
*   Stream ends with: `data: [DONE]`

## Database Persistence

Each completed chat interaction is saved in PostgreSQL:
*   Prompt (input)
*   Final response (output)
*   Model name
*   Timestamp

**Example table query:**
```sql
SELECT id, prompt, model_name, created_at
FROM chat_logs
ORDER BY id DESC;
```

## Why this architecture?

This project intentionally avoids shortcuts:
*   **LLM client** does not touch the database.
*   **Endpoint** does not contain business logic.
*   **Database writes** happen only after streaming completes to ensure data integrity.
*   **Async sessions** are passed explicitly (no globals), making the system easier to test and extend.

## Recommended Models

*   **For development and debugging:** `llama-3.3-8b-instant`
*   **For higher-quality responses:** `llama-3.3-70b-versatile`

## Who this project is for

*   Developers learning **async Python**
*   Developers learning **LLM streaming**
*   Developers learning **FastAPI + PostgreSQL**
*   Anyone building their first serious backend project

This repository is meant to be read, modified, and learned from.

## License

This project is provided for educational and demonstration purposes.