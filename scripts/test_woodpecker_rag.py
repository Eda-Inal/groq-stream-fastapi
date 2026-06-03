"""
End-to-end RAG test with woodpecker.md content using gpt-oss-120b.

Tests whether context_prefix embedding change broke or improved retrieval.
Compares against known-good results from rag_test_results.md.

Run: .venv\Scripts\python scripts\test_woodpecker_rag.py
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from uuid import uuid4

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:8000/api/v1"
MODEL = "llama-3.3-70b-versatile"
USER_ID = "test_user"
CONVERSATION_ID = f"test-woodpecker-{uuid4().hex[:8]}"
FILENAME = f"woodpecker_{uuid4().hex[:6]}.md"

WOODPECKER_TEXT = """
# The Woodpecker: Nature's Percussionist

## Overview

Woodpeckers are a family of birds known scientifically as *Picidae*. There are approximately 236 species of woodpeckers distributed across the world, found on every continent except Australia and Antarctica. They are most diverse in tropical forests, though many species thrive in temperate woodlands, savannas, and even deserts.

These birds are instantly recognizable by their strong, chisel-like beaks and their habit of drilling into tree bark. Their drumming behavior is one of the most distinctive sounds in any forest environment.

## Physical Characteristics

Woodpeckers range in size from the tiny piculets, which are only about 8 cm long, to the large and powerful great slaty woodpecker of Southeast Asia, which can reach up to 58 cm in length. Most species display bold patterns of black, white, and red, though some tropical species sport brilliant yellows and greens.

One of the most remarkable anatomical features of the woodpecker is its skull structure. The bone is unusually thick and spongy, acting as a natural shock absorber. This prevents brain injury despite the bird hammering its beak against wood at speeds of up to 20 kilometers per hour, with an impact force roughly 1,000 times greater than what would cause a concussion in a human being.

The woodpecker's tongue is another extraordinary adaptation. It can extend up to three times the length of the beak, wrapping around the skull when retracted. The tongue tip is barbed and coated with sticky saliva, allowing the bird to extract insects and larvae from deep within wood cavities with remarkable precision.

## Behavior and Diet

Woodpeckers are primarily insectivores. They feed on wood-boring beetles, carpenter ants, termites, and various larvae hidden beneath tree bark. Some species also consume tree sap, berries, nuts, and small vertebrates.

The drumming of a woodpecker serves two main purposes: foraging for food and communication. Territorial drumming can be heard up to a kilometer away in a quiet forest and is used by males to establish territory boundaries and attract mates.

## Nesting and Reproduction

Woodpeckers are cavity nesters, excavating their own nest holes in tree trunks or branches. This process can take anywhere from one to four weeks, depending on the species and the hardness of the wood.

A typical clutch consists of two to five white eggs. Both parents share incubation duties, which last approximately 11 to 14 days. The chicks hatch blind and helpless, remaining in the nest for three to four weeks before fledging.

Nest holes created by woodpeckers are an essential resource for many other species. Birds such as owls, starlings, small ducks, and various mammals use abandoned woodpecker cavities for their own nesting and shelter. In this way, woodpeckers function as ecosystem engineers, shaping the habitat for dozens of other species.

## The Pileated Woodpecker

The pileated woodpecker (*Dryocopus pileatus*) is the largest woodpecker species in North America, measuring about 40 to 49 cm in length. It is known for its striking red crest and powerful excavations, which can expose large oval or rectangular holes in dead trees.

This species prefers mature forests with large dead trees and logs. It plays a critical role in forest ecosystems by creating large cavities that are subsequently used by wood ducks, great horned owls, and even flying squirrels.

## Conservation Status

Most woodpecker species are not currently threatened. However, habitat loss is a growing concern for several species that depend on old-growth forests or specific tree species. The ivory-billed woodpecker (*Campephilus principalis*), once native to the southeastern United States and Cuba, is considered almost certainly extinct due to extensive deforestation during the 19th and 20th centuries.

## Summary Table

| Feature | Detail |
|---|---|
| Family | Picidae |
| Number of species | ~236 |
| Largest in North America | Pileated woodpecker (40-49 cm) |
| Pecking speed | Up to 20 km/h |
| Tongue length | Up to 3x the beak length |
| Primary diet | Insects, especially wood-boring beetles and ants |
| Conservation concern | Ivory-billed woodpecker (likely extinct) |
"""

QUESTIONS = [
    {
        "id": "Q1 (4.3 — spesifik detay)",
        "text": "How long does woodpecker egg incubation take?",
        "expected": "11 to 14 days",
    },
    {
        "id": "Q2 (4.2 — semantik paraphrase)",
        "text": "How does a woodpecker pull insects out of trees with its tongue?",
        "expected": "barbed + sticky saliva + 3x beak length",
    },
    {
        "id": "Q3 (4.4 — çok bölüm sentezi)",
        "text": "What role do woodpeckers play in the ecosystem for other animals?",
        "expected": "ecosystem engineers, cavity nesting for owls/ducks/squirrels",
    },
]


def upload_document() -> int:
    payload = json.dumps({
        "filename": FILENAME,
        "text": WOODPECKER_TEXT.strip(),
        "document_type": "text",
        "user_id": USER_ID,
        "conversation_id": CONVERSATION_ID,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/documents",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    print(f"  document_id : {data['document_id']}")
    print(f"  chunks      : {data['chunks_created']}")
    print(f"  tokens      : {data['tokens_processed']}")
    print(f"  model       : {data['embedding_model']}")
    return data["document_id"]


def ask(question: str) -> str:
    payload = json.dumps({
        "messages": [{"role": "user", "content": question}],
        "model": MODEL,
        "user_id": USER_ID,
        "conversation_id": CONVERSATION_ID,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    answer_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    answer_parts.append(delta)
            except Exception:
                pass

    return "".join(answer_parts).strip()


def main() -> None:
    print(f"\n{'='*60}")
    print("Woodpecker RAG — context_prefix embedding regression test")
    print(f"Model    : {MODEL}")
    print(f"Conv ID  : {CONVERSATION_ID}")
    print(f"{'='*60}\n")

    print("Uploading woodpecker.md...")
    try:
        upload_document()
    except Exception as e:
        print(f"UPLOAD FAILED: {e}")
        sys.exit(1)

    print("\nWaiting 2s for embedding to settle...\n")
    time.sleep(2)

    for q in QUESTIONS:
        print(f"{'─'*60}")
        print(f"{q['id']}")
        print(f"Question : {q['text']}")
        print(f"Expected : {q['expected']}")
        print()
        try:
            answer = ask(q["text"])
            print(f"Answer   : {answer}")
        except Exception as e:
            print(f"ERROR    : {e}")
        print()
        time.sleep(3)

    print(f"{'='*60}")
    print("Done. Compare answers against rag_test_results.md Test 4.")


if __name__ == "__main__":
    main()
