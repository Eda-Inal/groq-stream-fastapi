"""
Chunking helpers that are not environment-driven.

Numeric limits (chunk size, overlap, max document size, etc.) live in
`app.core.config.Settings` so they can be set via `.env` (see `.env.example`).
"""

# tiktoken encoding for `count_tokens` / chunk boundaries (not typically overridden per env)
TIKTOKEN_ENCODING = "cl100k_base"
