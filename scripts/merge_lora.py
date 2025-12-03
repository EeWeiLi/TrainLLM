# -*- coding: utf-8 -*-
"""
Merge LoRA adapter into full model weights (for inference/export).
Usage:
  python scripts/merge_lora.py \
    --adapter_dir ocr_qlora/final \
    --base_model Qwen/Qwen2-7B-Instruct \
    --out_dir ocr_merged_fp16
"""
import argparse, torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--out_dir", default="ocr_merged_fp16")
    args = ap.parse_args()

    print(f"[MERGE] Base={args.base_model}  Adapter={args.adapter_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter_dir,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto").merge_and_unload()
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[OK] Merged weights saved to {args.out_dir}")

if __name__ == "__main__":
    main()
