import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import fitz
print("pymupdf version:", fitz.__version__)

doc = fitz.open(r"C:\Users\edain\Downloads\Whales- Giants of the Ocean.pdf")
page = doc[0]

for mode in ("text", "md", "markdown", "html", "dict"):
    try:
        result = page.get_text(mode)
        print(f"  mode '{mode}': OK ({len(str(result))} chars)")
    except Exception as e:
        print(f"  mode '{mode}': HATA — {e}")
