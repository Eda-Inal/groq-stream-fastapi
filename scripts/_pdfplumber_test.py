import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import pdfplumber
except ImportError:
    print("pdfplumber yuklu degil: pip install pdfplumber")
    sys.exit(1)

pdf_path = r"C:\Users\edain\Downloads\Whales- Giants of the Ocean.pdf"

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()
        if tables:
            print(f"\nSayfa {page_num}: {len(tables)} tablo bulundu")
            for t_idx, table in enumerate(tables):
                print(f"\n  Tablo {t_idx+1} ({len(table)} satir x {len(table[0]) if table else 0} sutun):")
                for row in table[:5]:
                    print(f"    {row}")
        else:
            print(f"Sayfa {page_num}: tablo yok")
