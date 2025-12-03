# -*- coding: utf-8 -*-
"""
LoRA fine-tuning (raw OCR -> cleaned text) WITHOUT bitsandbytes / 4-bit.

- Uses bf16 weights on GPU (Qwen2-7B fits in 48 GB with LoRA).
- No gradient checkpointing (simpler & avoids grad issues).
- No packing (one sample per sequence).

Usage example:

  python scripts/train_data.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --data_file dataset/train.jsonl \
    --eval_file dataset/valid.jsonl \
    --out_dir ocr_qlora \
    --seq_len 2048 \
    --epochs 2 \
    --batch 1 \
    --grad_accum 16 \
    --lora_r 16
"""
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

PROMPT = (
    "You are a document restorer. Clean and reconstruct the OCR text faithfully.\n"
    "### Raw OCR:\n{raw}\n\n### Cleaned text:"
)


def fmt(raw: str) -> str:
    return PROMPT.format(raw=raw)


def load_text_dataset(file, val_file=None):
    ds = load_dataset("json", data_files=file, split="train")
    val = load_dataset("json", data_files=val_file, split="train") if val_file else None
    return ds, val


def map_field(ds):
    def _map(batch):
        texts = [fmt(i) + "\n" + t for i, t in zip(batch["input"], batch["target"])]
        return {"text": texts}
    return ds.map(_map, batched=True, remove_columns=list(ds.features))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2-7B-Instruct")
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--eval_file", default=None)
    ap.add_argument("--out_dir", default="ocr_qlora")
    ap.add_argument("--seq_len", type=int, default=2048)   # safer default for your setup
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- LoRA config ---
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --- Base model (bf16, no quantization) ---
    print("[INFO] Loading base model without 4-bit quantization (bf16, device_map='auto').")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = get_peft_model(model, lora)

    # Disable cache for training stability
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # --- Dataset prep ---
    train_ds, val_ds = load_text_dataset(args.data_file, args.eval_file)
    train_ds = map_field(train_ds)
    if val_ds is not None:
        val_ds = map_field(val_ds)

    # --- Training config (no checkpointing, no packing) ---
    cfg = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(1, args.batch // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        logging_steps=20,
        evaluation_strategy="steps" if val_ds is not None else "no",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        gradient_checkpointing=False,
        max_seq_length=args.seq_len,
        packing=False,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=cfg,
    )

    trainer.train()
    trainer.save_model(f"{args.out_dir}/final")
    print("[OK] Training complete.")


if __name__ == "__main__":
    main()
