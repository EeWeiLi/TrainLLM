# -*- coding: utf-8 -*-
"""
Prepare dataset directly from scanned PDFs (massive-corpus friendly):
1) OCR pages with pytesseract
2) Auto-clean and build (input, target) pairs
3) Chunk, dedupe, split -> dataset/train|valid|test.jsonl

NEW:
- --page_stride: use every Nth page (sample long PDFs)
- --max_pages: cap pages per PDF (0 = no cap)

Usage:
  python scripts/prepare_dataset.py \
    --pdfs "data_raw/*.pdf" \
    --out_dir dataset \
    --lang "eng+msa" --dpi 300 \
    --page_stride 5 --max_pages 120 \
    --min_chars 1500 --max_chars 7500 --overlap_ratio 0.12 \
    --val_ratio 0.02 --test_ratio 0.02
"""
import argparse, re, json, hashlib, random, unicodedata, os
from pathlib import Path
from typing import List, Dict
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# ---------- Optional Windows helpers ----------
# If you prefer not to edit PATH, uncomment & adjust these:
 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# POPPLER_PATH = r"C:\Program Files\poppler-24.02.0\Library\bin"

def nfkc(s): return unicodedata.normalize("NFKC", s)
def sha1(s): return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# Lines to DROP in TARGET (keep in input so the model learns to remove)
SERIAL_PAT = re.compile(r'^\s*(S/N|Note:?|Serial number|THIS PAGE IS LEFT BLANK).*$', re.I)
HEADER_FOOT_PAT = re.compile(r'^\s*(Page \d+|--- PAGE \d+ ---|\[\s*Signature block\s*\]).*$', re.I)
NRIC_PAT = re.compile(r'\b(\d{6}-\d{2}-\d{4})\b')  # Malaysia NRIC

def strip_boilerplate_lines(t: str) -> str:
    out=[]
    for line in t.splitlines():
        if SERIAL_PAT.match(line) or HEADER_FOOT_PAT.match(line): continue
        out.append(line)
    return "\n".join(out)

def dehyphen_wraps(s: str) -> str:
    s = re.sub(r'(\w+)-\n(\w+)', r'\1\2', s)      # pro-\nceed -> proceed
    s = re.sub(r'([^\n])\n(?!\n)', r'\1 ', s)     # unwrap single newlines inside paragraph
    return s

def clean_text(s: str) -> str:
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r' ?([,.;:])', r'\1', s)
    s = re.sub(r'\s+\n', '\n', s)
    s = re.sub(r'(\d{1,2})(st|nd|rd|th)\s*day', r'\1\2 day', s, flags=re.I)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = NRIC_PAT.sub("NRIC-REDACTED", s)
    return s.strip()

def build_target(raw: str) -> str:
    t = nfkc(raw)
    t = strip_boilerplate_lines(t)
    t = dehyphen_wraps(t)
    t = clean_text(t)
    return t

# Structure-aware split, then size windows with overlap
HEAD_SPLIT = re.compile(
    r'\n(?=(?:[A-Z][A-Z ]{4,}AGREEMENT|TABLE OF CONTENTS|CLAUSE\s+\d+|Execution Page|IN WITNESS WHEREOF))',
    re.M
)

def split_by_headings(text: str) -> List[str]:
    parts = HEAD_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()] or [text]

def window_with_overlap(s: str, min_chars: int, max_chars: int, overlap_ratio: float):
    s = s.strip()
    if len(s) <= max_chars:
        return [s] if len(s) >= min_chars else []
    chunks, step, i = [], int(max_chars*(1-overlap_ratio)), 0
    while i < len(s):
        piece = s[i:i+max_chars]
        last_para = piece.rfind("\n\n")
        end = i + last_para if last_para >= int(min_chars*0.6) else i + max_chars
        chunk = s[i:end].strip()
        if len(chunk) >= min_chars: chunks.append(chunk)
        i = max(i + step, end)
    return chunks

def chunk_text(raw_text, min_chars, max_chars, overlap_ratio):
    blocks = split_by_headings(raw_text)
    out=[]
    for b in blocks:
        if len(b) < min_chars: continue
        if len(b) > max_chars:
            out.extend(window_with_overlap(b, min_chars, max_chars, overlap_ratio))
        else:
            out.append(b)
    return out

def convert_from_path_winaware(pdf_path: str, dpi: int):
    # use poppler_path if provided via environment or constant
    poppler_path = os.environ.get("POPPLER_PATH", None)
    # If you uncommented the constant at the top, prefer it:
    try:
        POPPLER_PATH  # type: ignore # noqa
        poppler_path = POPPLER_PATH  # type: ignore # noqa
    except NameError:
        pass
    if poppler_path:
        return convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return convert_from_path(pdf_path, dpi=dpi)

def ocr_pdf_to_text(pdf_path: Path, dpi: int, lang: str, page_stride: int, max_pages: int) -> str:
    pages = convert_from_path_winaware(str(pdf_path), dpi=dpi)
    out=[]
    used = 0
    for idx, img in enumerate(pages):
        # Page sampling for huge PDFs
        if (idx % page_stride) != 0:
            continue
        if max_pages and used >= max_pages:
            break
        # Simple Pillow preprocessing; swap to OpenCV if you need deskew/adaptive threshold
        img = img.convert("L").point(lambda x: 0 if x < 200 else 255, "1")  # binarize
        text = pytesseract.image_to_string(img, lang=lang)
        out.append(text.strip() + f"\n\n<<<PAGE_BREAK:{idx+1}>>>")
        used += 1
    return "\n".join(out)

def split_by_document(pairs, val_ratio, test_ratio):
    random.seed(42)
    by_doc = {}
    for p in pairs: by_doc.setdefault(p["doc"], []).append(p)
    docs = list(by_doc.keys()); random.shuffle(docs)
    n = len(docs)
    n_test = max(1, int(n*test_ratio)) if n > 0 else 0
    n_val  = max(1, int(n*val_ratio))  if n > 1 else 0
    test_docs = set(docs[:n_test])
    val_docs  = set(docs[n_test:n_test+n_val])
    train_docs= set(docs[n_test+n_val:])
    train, val, test = [], [], []
    for d, items in by_doc.items():
        if d in test_docs: test += items
        elif d in val_docs: val += items
        else: train += items
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True)
    ap.add_argument("--out_dir", default="dataset")
    ap.add_argument("--lang", default="eng+msa")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--page_stride", type=int, default=1, help="Use every Nth page (1=all pages)")
    ap.add_argument("--max_pages", type=int, default=0, help="Cap pages per PDF (0=no cap)")
    ap.add_argument("--min_chars", type=int, default=1500)
    ap.add_argument("--max_chars", type=int, default=7500)
    ap.add_argument("--overlap_ratio", type=float, default=0.12)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--test_ratio", type=float, default=0.02)
    args = ap.parse_args()

    from glob import glob
    pdfs = sorted(glob(args.pdfs))
    if not pdfs:
        print("[WARN] No PDFs matched."); return

    pairs=[]
    for pdf in pdfs:
        p = Path(pdf)
        print(f"[OCR] {p.name} (dpi={args.dpi}, lang={args.lang}, stride={args.page_stride}, max_pages={args.max_pages or '∞'})")
        raw = nfkc(ocr_pdf_to_text(p, args.dpi, args.lang, args.page_stride, args.max_pages))
        for i,ch in enumerate(chunk_text(raw, args.min_chars, args.max_chars, args.overlap_ratio)):
            tgt = build_target(ch)
            if len(tgt) < 200: continue
            pairs.append({"doc": p.name, "chunk_id": i, "input": ch, "target": tgt})

    train, val, test = split_by_document(pairs, args.val_ratio, args.test_ratio)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name,data in [("train.jsonl",train),("valid.jsonl",val),("test.jsonl",test)]:
        with (out_dir/name).open("w",encoding="utf-8") as f:
            for r in data: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        print(f"[OK] {name}: {len(data)}")

if __name__ == "__main__":
    main()
