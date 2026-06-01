# PDF chunking gorsel test scripti.
# Kullanim: .venv\Scripts\python scripts/pdf_inspect.py dosya.pdf
#
# Gosterilen bilgiler:
#   - Her sayfanin PyMuPDF markdown ciktisi (# heading, tablo, liste)
#   - Birlestirilmis metinden uretilen chunk'lar
#   - Her chunk'ta section_heading, page_number, token sayisi
#   - Heading ve tablo tespiti ozeti

import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if len(sys.argv) < 2:
    print("Kullanim: python scripts/pdf_inspect.py dosya.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]
if not os.path.isfile(pdf_path):
    print(f"Dosya bulunamadi: {pdf_path}")
    sys.exit(1)

from app.services.pdf_extractor import extract_pages, PDFExtractionError
from app.services.chunking import chunk_document, count_tokens
from dataclasses import replace

SEP  = "─" * 70
SEP2 = "━" * 70

# ── PDF'i oku ────────────────────────────────────────────────────────────────
with open(pdf_path, "rb") as f:
    content = f.read()

print(f"\n{SEP2}")
print(f"  PDF: {os.path.basename(pdf_path)}  ({len(content):,} byte)")
print(SEP2)

try:
    pages = extract_pages(content)
except PDFExtractionError as e:
    print(f"\n  HATA: {e}")
    sys.exit(1)

print(f"\n  {len(pages)} sayfa bulundu\n")

# ── Sayfa markdown ciktilarini goster ────────────────────────────────────────
for p in pages:
    print(f"{SEP}")
    print(f"  SAYFA {p['page']}  ({count_tokens(p['text'])} token, {len(p['text'])} karakter)")
    print(SEP)

    lines = p["text"].split("\n")
    headings_found = [l for l in lines if l.startswith("#")]
    tables_found   = [l for l in lines if l.startswith("|")]

    print(f"  Tespit: {len(headings_found)} heading, {len(tables_found)} tablo satiri\n")

    # Markdown ciktisinin ilk 40 satirini goster
    preview_lines = lines[:40]
    for line in preview_lines:
        tag = ""
        if line.startswith("#"):
            tag = "  [HEADING] "
        elif line.startswith("|"):
            tag = "  [TABLO]   "
        elif line.startswith("-") or line.startswith("*"):
            tag = "  [LISTE]   "
        else:
            tag = "             "
        display = line[:100] + ("..." if len(line) > 100 else "")
        print(f"{tag}{display}")

    if len(lines) > 40:
        print(f"\n  ... (+{len(lines) - 40} satir daha)")

print()

# ── Sayfalar birlestirilip chunk'lanir ───────────────────────────────────────
page_char_ranges = []
merged_parts = []
char_pos = 0
for p in pages:
    text = p["text"]
    page_char_ranges.append((char_pos, char_pos + len(text), p["page"]))
    merged_parts.append(text)
    char_pos += len(text) + 2

merged_text = "\n\n".join(merged_parts)
all_chunks = list(chunk_document(merged_text))

def _page_for(chunk_text):
    tail = chunk_text.rstrip()[-60:]
    idx = merged_text.rfind(tail)
    if idx == -1:
        tail = chunk_text.rstrip()[-20:]
        idx = merged_text.rfind(tail)
    if idx == -1:
        return None
    for c_start, c_end, page_num in page_char_ranges:
        if c_start <= idx < c_end:
            return page_num
    return None

total = len(all_chunks)
all_chunks = [
    replace(c, chunk_index=i, total_chunks=total, page_number=_page_for(c.text))
    for i, c in enumerate(all_chunks)
]

print(f"{SEP2}")
print(f"  CHUNK SONUCLARI  —  {total} chunk uretildi")
print(f"  Toplam token: {sum(c.token_count for c in all_chunks)}")
print(SEP2)

headings_with_section = sum(1 for c in all_chunks if c.section_heading)
tables_in_chunks = sum(1 for c in all_chunks if "|" in c.text)

print(f"\n  section_heading dolu : {headings_with_section}/{total} chunk")
print(f"  Tablo iceren chunk   : {tables_in_chunks}/{total} chunk\n")

for c in all_chunks:
    bar = "#" * min(c.token_count // 10, 50)
    page_info  = f"sayfa {c.page_number}" if c.page_number else "sayfa ?"
    heading_info = f'  heading="{c.section_heading}"' if c.section_heading else ""
    print(f"{SEP}")
    print(f"  Chunk {c.chunk_index + 1:02d}/{total}  [{c.token_count:>4} token]  {page_info}{heading_info}")
    print(f"  {bar}")

    preview = c.text[:300].replace("\n", " | ")
    print(f"  {preview}")
    if len(c.text) > 300:
        print(f"  ... (+{len(c.text) - 300} karakter daha)")
    print()

print(SEP2)
print("  Tamamlandi.")
print(SEP2)
