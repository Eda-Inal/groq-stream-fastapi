import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymupdf4llm, fitz

pdf_path = r"C:\Users\edain\Downloads\Whales- Giants of the Ocean.pdf"
doc = fitz.open(pdf_path)
chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)

# Print all pages
for i, chunk in enumerate(chunks):
    meta = chunk.get("metadata", {})
    text = chunk.get("text", "")
    tables = chunk.get("tables", [])
    print(f"\n{'='*60}")
    print(f"Chunk {i}  page={meta.get('page')}  tables={len(tables)}")
    print(f"{'='*60}")
    print(text[:600])
    if tables:
        print(f"\n  [TABLE DATA]: {tables}")
