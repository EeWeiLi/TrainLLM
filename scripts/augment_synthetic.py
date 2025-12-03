# -*- coding: utf-8 -*-
"""
Create synthetic (input=noisy, target=clean) pairs from a clean legal corpus.
Input: one large text file with paragraphs separated by blank lines.
Usage:
  python scripts/augment_synthetic.py \
    --clean_file synthetic/clean_corpus.txt \
    --out synthetic/synthetic.jsonl \
    --min_chars 300 --seed 42
"""
import argparse, re, json, random, unicodedata

def noise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    # random hyphenation (simulate wrap)
    s = re.sub(r'(\w{4,12}) (\w{4,12})',
               lambda m: m.group(1) + ('-\n' if random.random()<0.18 else ' ') + m.group(2), s)
    # ligatures
    if random.random() < 0.35: s = s.replace("fi","ﬁ")
    if random.random() < 0.35: s = s.replace("fl","ﬂ")
    # stray spaces before punctuation
    if random.random() < 0.25: s = re.sub(r' +([,.;:])', r' \1', s)
    # fake headers/footers
    if random.random() < 0.22:
        s = f"--- PAGE {random.randint(1,300)} ---\n{s}\nNote: Serial number will be used..."
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_file", required=True)
    ap.add_argument("--out", default="synthetic/synthetic.jsonl")
    ap.add_argument("--min_chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = open(args.out, "w", encoding="utf-8")
    buf = []
    with open(args.clean_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip()=="":
                if buf:
                    para = "\n".join(buf).strip()
                    buf = []
                    if len(para) >= args.min_chars:
                        out.write(json.dumps({"input": noise(para), "target": para}, ensure_ascii=False)+"\n")
            else:
                buf.append(line.rstrip("\n"))
        if buf:
            para = "\n".join(buf).strip()
            if len(para) >= args.min_chars:
                out.write(json.dumps({"input": noise(para), "target": para}, ensure_ascii=False)+"\n")
    out.close()
    print(f"[OK] Wrote → {args.out}")

if __name__ == "__main__":
    main()
