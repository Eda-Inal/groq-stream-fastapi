from app.utils.token_counter import (
    count_tokens,
    estimate_messages_tokens,
    truncate_rag_chunks,
    truncate_text_to_token_budget,
)


def test_count_tokens_non_empty() -> None:
    assert count_tokens("hello world") > 0


def test_estimate_messages_tokens_grows_with_content() -> None:
    short = [{"role": "user", "content": "hi"}]
    long = [{"role": "user", "content": "hi " * 200}]
    assert estimate_messages_tokens(long) > estimate_messages_tokens(short)


def test_truncate_text_to_token_budget() -> None:
    text = "word " * 500
    out = truncate_text_to_token_budget(text, 50)
    assert out
    assert count_tokens(out) <= 50


def test_truncate_rag_chunks_respects_budget_order() -> None:
    chunks = [
        "alpha " * 50,
        "beta " * 50,
        "gamma " * 50,
    ]
    out = truncate_rag_chunks(chunks, budget=count_tokens(chunks[0]) + 10)
    assert len(out) >= 1
    # First chunk should remain first (preserve ordering)
    assert out[0].startswith("alpha")
