📘 OCR LLM Fine-Tuning Pipeline (Qwen2-7B + LoRA)

This repository contains a complete, end-to-end workflow for training a custom OCR text restoration model using LoRA fine-tuning on Qwen2-7B-Instruct.
The model learns to convert raw OCR text from scanned legal PDFs into clean, corrected, normalized text.

This project supports:

Full-page OCR of scanned PDFs (Poppler + Tesseract)

Automatic dataset generation (train / val / test)

LoRA fine-tuning (bf16, no bitsandbytes)

Evaluation on held-out test set

Model merging for deployment

Fully tested on Windows + 48GB GPU (CUDA).
