import json
import requests
import gradio as gr

BACKEND_URL = "http://localhost:8000/api/v1/chat/stream"


def stream_chat(user_input, messages):
    if messages is None:
        messages = []

    # Add user message
    messages.append({"role": "user", "content": user_input})

    payload = {"messages": messages}
    assistant_text = ""

    with requests.post(
        BACKEND_URL,
        json=payload,
        stream=True,
        timeout=60,
    ) as response:

        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue

            data = raw_line.replace("data:", "").strip()

            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            delta = (
                chunk.get("choices", [{}])[0]
                .get("delta", {})
                .get("content")
            )

            if delta:
                assistant_text += delta

                yield (
                    messages + [{"role": "assistant", "content": assistant_text}],
                    messages,
                    ""  # clear input box while streaming
                )

    # Finalize assistant message
    messages.append({"role": "assistant", "content": assistant_text})

    yield messages, messages, ""


def clear_chat():
    return [], []


with gr.Blocks() as demo:
    gr.Markdown("# LLM Streaming Demo (FastAPI + Gradio)")

    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Type your message and press Enter",
            show_label=False,
            scale=4,
        )
        clear_btn = gr.Button("Clear chat", scale=1)

    user_input.submit(
        stream_chat,
        inputs=[user_input, state],
        outputs=[chatbot, state, user_input],
    )

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, state],
    )

demo.launch()
