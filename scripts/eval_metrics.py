# -*- coding: utf-8 -*-
"""
Evaluate on held-out test.jsonl (CER/WER/BLEU) and save a few sample predictions.
Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_metrics.py \
    --model_dir ocr_qlora/final \
    --base_model Qwen/Qwen2-7B-Instruct \
    --test_file dataset/test.jsonl \
    --out_dir results --max_new_tokens 1200
"""
import argparse, os, json
import jiwer, evaluate, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = (
"You are a document restorer. Clean and reconstruct the OCR text faithfully.\n"
"### Raw OCR:\n{raw}\n\n### Cleaned text:"
)

def fmt(raw: str) -> str:
    return PROMPT.format(raw=raw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples_eval"; samples_dir.mkdir(exist_ok=True)

    tok_src = args.model_dir if args.base_model is None else args.base_model
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    bleu = evaluate.load("bleu")

    refs, hyps = [], []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            prompt = fmt(ex["input"])
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = tok.decode(out[0], skip_special_tokens=True)
            pred = text.split("### Cleaned text:")[-1].strip()

            refs.append(ex["target"])
            hyps.append(pred)

            if i < 50:  # save first 50 qualitative samples
                with open(samples_dir / f"sample_{i:05d}.txt","w",encoding="utf-8") as sf:
                    sf.write("=== INPUT (OCR) ===\n"+ex["input"][:8000]+"\n\n")
                    sf.write("=== PREDICTION ===\n"+pred[:8000]+"\n\n")
                    sf.write("=== TARGET (CLEAN) ===\n"+ex["target"][:8000]+"\n")

    cer = jiwer.cer(refs, hyps)
    wer = jiwer.wer(refs, hyps)
    bleu_score = bleu.compute(predictions=hyps, references=[[r] for r in refs])["bleu"]

    with open(out_dir / "metrics.txt","w",encoding="utf-8") as mf:
        mf.write(f"CER:  {cer:.6f}\nWER:  {wer:.6f}\nBLEU: {bleu_score:.6f}\n")

    print(f"CER:  {cer:.6f}\nWER:  {wer:.6f}\nBLEU: {bleu_score:.6f}")
    print(f"[OK] Wrote metrics and samples to {out_dir}")

if __name__ == "__main__":
    main()
