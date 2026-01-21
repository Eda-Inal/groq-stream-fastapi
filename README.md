# groq-stream-fastapi

A minimal, learning-focused demo project that shows how to build an **async streaming API** with **FastAPI** using **Groq’s OpenAI-compatible LLM API**.

This project demonstrates how to stream LLM responses **token-by-token** from a backend service to a client in real time.

---

## What this project demonstrates

* **Async FastAPI** endpoints
* **Real-time streaming** with `StreamingResponse`
* **Streaming LLM responses** using Groq’s OpenAI-compatible API
* Clean separation of **API**, **service**, and **config** layers
* **Environment-based configuration** with `.env`

This is a **submission-ready demo** while still being easy to read and learn from.

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
│  └─ groq_client.py
├─ schemas/
│  └─ chat.py
├─ core/
│  └─ config.py
```

---

## Setup

### 1️- Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv/Scripts/activate
```

### 2- Install dependencies

```bash
pip install -r requirements.txt
```

### 3- Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_DEFAULT_MODEL=llama-3.3-70b-versatile
```

**Notes:**

* `.env` is ignored by git
* `.env.example` documents required variables

---

##  Run the application

```bash
uvicorn app.main:app --reload
```

### Health check

```http
GET http://127.0.0.1:8000/health
```

---

## Streaming Chat Endpoint

### Endpoint

```http
POST /api/v1/chat/stream
```

### Request body

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain streaming responses briefly"
    }
  ]
}
```

* `messages` uses the **OpenAI-compatible chat format**
* `model` is optional and defaults to `GROQ_DEFAULT_MODEL`

---

## Test with curl (recommended)

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

### Expected behavior

* The response is **streamed progressively**
* Output appears **incrementally** in the terminal
* Short prompts may return in a **single chunk**, depending on the model

---

## Notes

* **Swagger UI** does not properly display streaming responses
* Use `curl -N`, a frontend client, or any HTTP client that supports **streaming**
* Streaming behavior depends on **model output** and **prompt length**

---

## Recommended Models

For development and debugging:

```text
llama-3.3-8b-instant
```

For higher-quality production responses:

```text
llama-3.3-70b-versatile
```

---

## License

This project is provided for **educational and demo purposes**.
