from pathlib import Path

import fitz 

BASE_DIR = Path(__file__).resolve().parent
src_pdf = BASE_DIR / "data" / "Understanding_Climate_Change.pdf"
dst_pdf = BASE_DIR / "data" / "Understanding_Climate_Change_fixed.pdf"

doc = fitz.open(src_pdf)
doc.save(dst_pdf, garbage=4, deflate=True, clean=True)
doc.close()