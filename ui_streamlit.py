import json
import httpx
import streamlit as st

RESEARCH_URL = "http://localhost:8000/api/v1/research/stream"

MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
    "gemini-2.5-flash",
]

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "report_text" not in st.session_state:
    st.session_state.report_text = ""
if "searches" not in st.session_state:
    st.session_state.searches = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "done_meta" not in st.session_state:
    st.session_state.done_meta = None

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
left, right = st.columns([1, 2])

with left:
    st.title("Research Agent")
    st.caption("Enter a topic and the agent will search the web and produce a structured report.")

    topic = st.text_area(
        "Research topic",
        placeholder="e.g. quantum computing, CRISPR gene editing, large language models",
        height=100,
        disabled=st.session_state.is_running,
    )

    model = st.selectbox(
        "Model",
        options=MODELS,
        disabled=st.session_state.is_running,
    )

    language = st.radio(
        "Report language",
        options=["English", "Turkish"],
        horizontal=True,
        disabled=st.session_state.is_running,
    )

    run_button = st.button(
        "Research" if not st.session_state.is_running else "Researching…",
        type="primary",
        disabled=st.session_state.is_running,
        use_container_width=True,
    )

    if st.session_state.done_meta:
        st.caption(
            f"✓ {st.session_state.done_meta['total_searches']} searches · "
            f"{st.session_state.done_meta['elapsed_seconds']}s"
        )

with right:
    progress_placeholder = st.empty()
    report_placeholder = st.empty()
    download_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Show previous results if not running
# ---------------------------------------------------------------------------
if not st.session_state.is_running:
    if st.session_state.error_message:
        report_placeholder.error(st.session_state.error_message)
    elif st.session_state.report_text:
        report_placeholder.markdown(st.session_state.report_text)
        download_placeholder.download_button(
            label="Download report (.md)",
            data=st.session_state.report_text,
            file_name=f"report_{topic[:40].replace(' ', '_') if topic else 'research'}.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Run research
# ---------------------------------------------------------------------------
if run_button and topic.strip():
    # Reset state for new run
    st.session_state.report_text = ""
    st.session_state.searches = []
    st.session_state.error_message = None
    st.session_state.done_meta = None
    st.session_state.is_running = True
    st.rerun()

if st.session_state.is_running and topic.strip():
    lang_code = "en" if language == "English" else "tr"
    payload = {"topic": topic.strip(), "model": model, "language": lang_code}

    progress_lines: list[str] = []
    report_parts: list[str] = []

    def render_progress():
        progress_placeholder.markdown("\n\n".join(progress_lines) if progress_lines else "")

    def render_report():
        if report_parts:
            report_placeholder.markdown("".join(report_parts))

    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", RESEARCH_URL, json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "start":
                        progress_lines.append(
                            f"**Starting research:** {event.get('topic')}  \n"
                            f"Model: `{event.get('model')}`"
                        )
                        render_progress()

                    elif etype == "searching":
                        progress_lines.append(
                            f"🔍 **Round {event.get('round')}:** {event.get('query')}"
                        )
                        render_progress()

                    elif etype == "found":
                        # Replace the last "searching" line with found status
                        if progress_lines:
                            progress_lines[-1] += (
                                f"  \n&nbsp;&nbsp;&nbsp;&nbsp;✓ {event.get('source_count')} sources found"
                            )
                        render_progress()

                    elif etype == "nudge":
                        progress_lines.append(
                            f"↩ Searching more angles ({event.get('search_count')} / 3 minimum)…"
                        )
                        render_progress()

                    elif etype == "finalizing":
                        progress_lines.append(
                            f"\n---\n✍️ **Generating report** from {event.get('total_searches')} searches…"
                        )
                        render_progress()

                    elif etype == "report_chunk":
                        text = event.get("text", "")
                        report_parts.append(text)
                        render_report()

                    elif etype == "done":
                        st.session_state.done_meta = {
                            "total_searches": event.get("total_searches"),
                            "elapsed_seconds": event.get("elapsed_seconds"),
                        }

                    elif etype == "error":
                        st.session_state.error_message = (
                            f"Error ({event.get('phase', 'unknown')}): {event.get('message')}"
                        )
                        break

    except httpx.HTTPStatusError as e:
        st.session_state.error_message = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        st.session_state.error_message = f"Connection error: {str(e)}"

    st.session_state.report_text = "".join(report_parts)
    st.session_state.is_running = False
    st.rerun()
