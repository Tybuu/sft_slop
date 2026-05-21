# Qwen 3.5 SFT & Distillation Workspace

This workspace is focused on Supervised Fine-Tuning (SFT) and Knowledge Distillation (KD) experiments using **Qwen 3.5** models.

## Core Focus
- **Target Models:** Qwen 3.5 series (e.g., `Qwen/Qwen3.5-0.8B` as student, `Qwen/Qwen3.5-2B` as teacher).
- **Task Domains:** Tool-use and Science-related reasoning.
- **Goal:** Improving small model performance via KD and SFT.

## Primary Files
- `train_sft_qwen.py`: Standard Supervised Fine-Tuning for Qwen models.
- `generate_qwen_teacher_logits.py`: Pre-computes top-100 teacher logits from a larger Qwen model (e.g., 2B) for distillation.
- `train_distilled_qwen.py`: Trains a smaller Qwen model (e.g., 0.8B) using the pre-computed teacher logits.
- `sdft_repo/`: Used **exclusively** for datasets (located in `sdft_repo/data/`).

## Irrelevant / Legacy Files
The following files are part of previous experiments (often Llama-based or generic) and should be ignored unless explicitly requested:
- `train_sft.py`, `train_distilled.py`, `train_student.py`
- `generate_teacher_logits.py`
- `data_loader.py`, `eval.py`, `utils.py` (older versions)

## Workflow Conventions
- **Hardware Target:** Optimized for Kaggle T4x2 environments. Often uses `CUDA_VISIBLE_DEVICES` to manage GPU allocation manually.
- **Distillation:** Uses a two-phase approach:
    1. Generate logits with `generate_qwen_teacher_logits.py`.
    2. Train student with `train_distilled_qwen.py`.
- **Tokenization:** Since teacher and student share the same Qwen 3.5 tokenizer, no cross-vocabulary mapping is required.

## Technical Details
- **Teacher Logits:** Saved as `.pt` files containing top-100 probabilities and indices.
- **Training:** Uses `trl.SFTTrainer` for SFT and custom loss functions for distillation.
