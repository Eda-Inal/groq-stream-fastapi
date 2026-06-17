"""
Upload a text file to the document ingestion endpoint.

user_id used for Sherlock Holmes upload: sherlock-rag-test
  - This document is permanent in the DB (not scheduled for deletion).

Usage:
    python scripts/upload_document.py "Sherlock Holmes.txt" --user-id sherlock-rag-test --half
    python scripts/upload_document.py "Sherlock Holmes.txt" --user-id sherlock-rag-test
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

API_URL = "http://localhost:8000/api/v1/documents"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the text file")
    parser.add_argument("--user-id", default="sherlock-rag-test")
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--half", action="store_true", help="Upload only the first half of the file")
    parser.add_argument("--percent", type=int, default=None, help="Upload first N%% of the file (e.g. 15)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    original_len = len(text)

    if args.percent is not None:
        text = text[: int(len(text) * args.percent / 100)]
        print(f"{args.percent}% slice: {len(text):,} / {original_len:,} chars")
    elif args.half:
        text = text[: len(text) // 2]
        print(f"Half mode: {len(text):,} / {original_len:,} chars")
    else:
        print(f"Full file: {original_len:,} chars")

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    payload = {
        "text": text,
        "filename": path.name,
        "document_type": "text",
        "user_id": args.user_id,
        "tags": tags,
    }
    if args.conversation_id:
        payload["conversation_id"] = args.conversation_id

    print(f"Uploading '{path.name}' -> {API_URL}")
    print(f"  user_id: {args.user_id}")
    if args.conversation_id:
        print(f"  conversation_id: {args.conversation_id}")

    t0 = time.monotonic()
    try:
        r = httpx.post(API_URL, json=payload, timeout=600.0)
    except httpx.ConnectError:
        print("Connection refused — is the API running on localhost:8000?")
        sys.exit(1)

    elapsed = round(time.monotonic() - t0, 1)

    if r.status_code == 200:
        data = r.json()
        print(f"\nDone in {elapsed}s")
        print(f"  document_id   : {data['document_id']}")
        print(f"  chunks_created: {data['chunks_created']}")
        print(f"  tokens        : {data['tokens_processed']}")
        print(f"  embedding     : {data['embedding_model']}")
        print(f"  ingest time   : {data['elapsed_ms']}ms")
    else:
        print(f"\nFailed ({r.status_code}) in {elapsed}s")
        print(r.text[:500])
        sys.exit(1)


if __name__ == "__main__":
    main()
