"""
Verifies that embed_batch() correctly splits large inputs into mini-batches.

Sends 120 texts (> default batch_size=50) and checks:
  - All 120 results are returned
  - Each result is a valid vector of the correct dimension
  - No result slot is None

Run: .venv\Scripts\python scripts\test_embed_batch.py
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.services.embeddings import EmbeddingService


async def run() -> None:
    svc = EmbeddingService()

    batch_size = settings.embedding_batch_size
    total = batch_size * 2 + 20  # e.g. 120 for batch_size=50 → 3 mini-batches

    texts = [f"This is test sentence number {i} about marine biology and ocean ecosystems." for i in range(total)]

    print(f"\n{'='*55}")
    print(f"embed_batch() mini-batch test")
    print(f"  batch_size setting : {batch_size}")
    print(f"  texts to embed     : {total}")
    print(f"  expected batches   : {-(-total // batch_size)}")  # ceil div
    print(f"  embedding model    : {settings.embedding_model_name}")
    print(f"  embedding dim      : {settings.embedding_dim}")
    print(f"{'='*55}\n")

    print("Calling embed_batch()...")
    results = await svc.embed_batch(texts)

    print()
    if results is None:
        print("FAIL — embed_batch() returned None")
        print("       Check that the embedding service is reachable.")
        sys.exit(1)

    print(f"Results received : {len(results)}")

    errors: list[str] = []

    if len(results) != total:
        errors.append(f"Expected {total} results, got {len(results)}")

    for i, r in enumerate(results):
        if r is None:
            errors.append(f"Result [{i}] is None")
        elif len(r.vector) != settings.embedding_dim:
            errors.append(f"Result [{i}] has wrong dim: {len(r.vector)} (expected {settings.embedding_dim})")

    print(f"None slots       : {sum(1 for r in results if r is None)}")
    print(f"Correct dim      : {sum(1 for r in results if r and len(r.vector) == settings.embedding_dim)}/{total}")

    print()
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL PASS — mini-batching works correctly")


if __name__ == "__main__":
    asyncio.run(run())
