# OCR LLM Fine-Tuning Pipeline (Qwen2-7B + LoRA)

This repository contains a complete, end-to-end workflow for training a custom OCR text restoration model using LoRA fine-tuning on Qwen2-7B-Instruct.
The model learns to convert raw OCR text from scanned legal PDFs into clean, corrected, normalized text.

This project supports:

Full-page OCR of scanned PDFs (Poppler + Tesseract)

Automatic dataset generation (train / val / test)

LoRA fine-tuning (bf16, no bitsandbytes)

Evaluation on held-out test set

Model merging for deployment

Fully tested on Windows + 48GB GPU (CUDA).

## Features

Train your own OCR-correction LLM from scanned PDFs

End-to-end automated dataset builder

Supports Malay + English OCR (eng+msa)

No bitsandbytes required (compatible with Windows CUDA 12+)

Produces a single merged model ready for deployment

Perfect for legal documents, agreements, contracts, financial PDFs, etc.

## Project Structure
FineTune/

│

├── data_raw/               # Put your scanned PDFs here

├── dataset/                # Auto-generated JSONL dataset (train/valid/test)

│

├── scripts/

│   ├── prepare_dataset.py  # OCR + cleaning + chunking

│   ├── train_data.py       # Qwen2-7B LoRA fine-tuning

│   ├── eval_metrics.py     # Evaluation on test.jsonl

│   ├── merge_lora.py       # Merge LoRA → full model

│
├── requirements.txt

├── README.md

└── .venv/                  # Python virtual environment (recommended)

## System Requirements
Hardware

NVIDIA GPU with ≥ 24GB VRAM

Recommended: 48GB GPU (verified)

Software

Windows 10/11 or WSL2 Ubuntu

Python 3.10–3.12

CUDA toolkit + compatible GPU drivers

Poppler (PDF → image)

Tesseract OCR (image → text)

## Installation
1. Clone project / download folder
cd C:\Users\user\Downloads

2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

3. Install requirements
pip install -r requirements.txt

4. Uninstall bitsandbytes (Windows incompatible)
pip uninstall -y bitsandbytes

5. Install Poppler & Tesseract

Poppler → C:\Program Files\poppler-24.02.0\

Tesseract → C:\Program Files\Tesseract-OCR\

Make sure prepare_dataset.py points to correct paths.

1. Add Your PDF Files

Place all scanned PDFs into:

data_raw/


Example files:

data_raw/125. CBD (Part B - Vol 1) +002_unlocked.pdf
data_raw/242. CBOD Part B Volume 12 +002_unlocked.pdf
...

2. Build Training Dataset (OCR → JSONL)

This performs:

Full OCR of every page

Text normalization

Cleans artifacts (line breaks, spacing, noise)

Splits into chunks

Generates train/valid/test JSONL

Run:

python scripts/prepare_dataset.py --pdfs "data_raw/*.pdf" --out_dir dataset --lang "eng+msa" --dpi 300 --page_stride 1 --max_pages 0 --min_chars 1000 --max_chars 9000 --overlap_ratio 0.12 --val_ratio 0.02 --test_ratio 0.02


You will get:

dataset/train.jsonl
dataset/valid.jsonl
dataset/test.jsonl

3. Train Qwen2-7B-Instruct with LoRA (bf16)

Because Windows cannot use bitsandbytes reliably, we use bf16 LoRA, which fits comfortably in 48GB VRAM.

Recommended training command:

python scripts/train_data.py --base_model Qwen/Qwen2-7B-Instruct --data_file dataset/train.jsonl --eval_file dataset/valid.jsonl --out_dir ocr_qlora --seq_len 2048 --epochs 2 --batch 1 --grad_accum 16 --lora_r 16


This will:

Load Qwen2-7B in bf16

Attach LoRA adapters to attention + MLP layers

Train on your dataset

Save results to:

ocr_qlora/final/

4. Evaluate Model on Test Set

Run:

python scripts/eval_metrics.py --model_dir ocr_qlora/final --base_model Qwen/Qwen2-7B-Instruct --test_file dataset/test.jsonl --out_dir results --max_new_tokens 1200


Produces:

results/metrics.txt
results/samples_eval/


You will see:

WER (word error rate)

CER (character error rate)

BLEU score

Before/after examples

5. Merge LoRA Into Full Model (Deployable)
python scripts/merge_lora.py --adapter_dir ocr_qlora/final --base_model Qwen/Qwen2-7B-Instruct --out_dir ocr_merged_fp16


Produces a folder:

ocr_merged_fp16/


This folder acts like a single HF model:

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("ocr_merged_fp16")
model = AutoModelForCausalLM.from_pretrained("ocr_merged_fp16", torch_dtype="bfloat16").cuda()

6. Use Your Custom OCR-Restoration Model

Example usage:

prompt = "You are a document restorer. Clean and reconstruct the OCR text faithfully.\n### Raw OCR:\n" + raw_ocr + "\n\n### Cleaned text:"

out = model.generate(
    **tok(prompt, return_tensors="pt").to(model.device),
    max_new_tokens=800
)

print(tok.decode(out[0], skip_special_tokens=True))


You now have your own legal-document OCR correction transformer.

## Tips & Best Practices

For giant PDFs (500+ pages), consider using page_stride=2 or 3 to save time.

Larger seq_len improves quality but increases VRAM usage.

Your 48GB GPU can do 7B models easily, but Windows memory fragmentation requires conservative settings.

If you switch to Linux/WSL2, you may enable 4-bit QLoRA.

## Troubleshooting
❗ CUDA Out Of Memory

Use:

--seq_len 2048 --batch 1 --grad_accum 16

❗ bitsandbytes / CUDA126 errors

Solution: uninstall bitsandbytes.

❗ OCR too noisy

Increase DPI:

--dpi 300

❗ Dataset too small

Add more PDFs to data_raw/ and rerun prepare_dataset.py.

## Summary

You now have a full pipeline:

Drop PDFs into data_raw/

Build dataset with prepare_dataset.py

Train Qwen2-7B LoRA with train_data.py

Evaluate with eval_metrics.py

Merge model with merge_lora.py

Deploy your custom OCR restoration model
