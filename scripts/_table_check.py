import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pdf_extractor import extract_pages
from app.services.chunking import chunk_document
from dataclasses import replace

pdf_path = r"C:\Users\edain\Downloads\Whales- Giants of the Ocean.pdf"
with open(pdf_path, "rb") as f:
    content = f.read()

pages = extract_pages(content)

page_char_ranges, merged_parts, char_pos = [], [], 0
for p in pages:
    page_char_ranges.append((char_pos, char_pos + len(p["text"]), p["page"]))
    merged_parts.append(p["text"])
    char_pos += len(p["text"]) + 2
merged_text = "\n\n".join(merged_parts)

all_chunks = list(chunk_document(merged_text))

def _page_for(t):
    idx = merged_text.rfind(t.rstrip()[-60:])
    if idx == -1: idx = merged_text.rfind(t.rstrip()[-20:])
    if idx == -1: return None
    for s,e,n in page_char_ranges:
        if s <= idx < e: return n

all_chunks = [replace(c, chunk_index=i, total_chunks=len(all_chunks),
    page_number=_page_for(c.text)) for i, c in enumerate(all_chunks)]

# Tablo ile ilgili chunk'ları bul
for c in all_chunks:
    if "table" in (c.section_heading or "").lower() or "tablo" in c.text.lower():
        print(f"\n{'='*60}")
        print(f"Chunk {c.chunk_index+1}  [{c.token_count} token]  section_heading={c.section_heading!r}")
        print(f"{'='*60}")
        print(repr(c.text[:500]))
        print(f"\n--- GORSEL ---")
        print(c.text[:500])
