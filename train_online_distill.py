"""
train_online_distill.py

Online knowledge distillation from a teacher model (with a fine-tuned LoRA)
to a student model (output LoRA). Both teacher and student run in BF16.

Usage:
    # First, SFT-train a LoRA on the teacher (e.g. 2B model):
    python train_sft_qwen.py --dataset tooluse --model_path Qwen/Qwen3.5-2B \
        --output_dir ./qwen-2b-expert --epochs 2 --batch_size 1 --grad_acc 8

    # Then distill to 0.8B using the trained teacher LoRA:
    python train_online_distill.py --dataset tooluse \
        --teacher_model Qwen/Qwen3.5-2B \
        --teacher_lora_path ./qwen-2b-expert \
        --dataset_path data/tooluse_data/train_data_fixed \
        --output_dir ./qwen-distilled-online
"""

import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Allows TF32 matmul on Tensor Cores (RTX 5090 supports this natively)
torch.set_float32_matmul_precision("medium")


# ── Helpers ───────────────────────────────────────────────────────────────────

def prepare_online_distill_dataset(raw_hf_dataset, dataset_name, tokenizer, max_length):
    from utils import format_target

    def _tokenize_and_align(example, idx):
        if dataset_name == "science":
            messages = [
                {"role": "system",    "content": example["messages"][0]["content"]},
                {"role": "user",      "content": example["messages"][1]["content"]},
                {"role": "assistant", "content": example["output_text"]},
            ]
        elif dataset_name == "tooluse":
            target = format_target(example["golden_response"][0])
            messages = [
                {"role": "system",    "content": "You are a helpful assistant."},
                {"role": "user",      "content": example["prompt"]},
                {"role": "assistant", "content": target},
            ]
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        prompt_msgs = messages[:-1]
        prompt_ids = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, enable_thinking=False)
        full_ids = tokenizer.apply_chat_template(messages, enable_thinking=False)

        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        elif isinstance(prompt_ids, dict) and "input_ids" in prompt_ids:
            prompt_ids = prompt_ids["input_ids"]
        if hasattr(full_ids, "input_ids"):
            full_ids = full_ids.input_ids
        elif isinstance(full_ids, dict) and "input_ids" in full_ids:
            full_ids = full_ids["input_ids"]

        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if hasattr(full_ids, "tolist"):
            full_ids = full_ids.tolist()

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        resp_start = len(prompt_ids)
        resp_end = min(len(full_ids), max_length)

        labels = [-100] * len(full_ids)
        for i in range(resp_start, resp_end):
            labels[i] = full_ids[i]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return raw_hf_dataset.map(
        _tokenize_and_align,
        with_indices=True,
        remove_columns=raw_hf_dataset.column_names,
        num_proc=1,
    )


# ── Distillation trainer ──────────────────────────────────────────────────────

class OnlineDistillationTrainer(SFTTrainer):
    def __init__(self, teacher_model, alpha=0.3, temp=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temp = temp

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # ── Teacher forward (no gradients, native BF16) ────────────────────
        with torch.inference_mode():
            t_outputs = self.teacher_model(**inputs)
            t_logits = t_outputs.logits  # (B, L, V) — stays BF16

        # ── Student forward ─────────────────────────────────────────────────
        s_outputs = model(**inputs)
        s_logits = s_outputs.logits  # (B, L, V) — BF16
        loss_ce = s_outputs.loss

        # ── Full-vocab KL over response tokens ─────────────────────────────
        response_mask = (labels != -100)
        response_positions = response_mask.nonzero(as_tuple=True)

        if response_positions[0].numel() > 0:
            # Gather response-position logits; shift by -1 to align predictions
            t_resp = t_logits[response_positions[0], response_positions[1] - 1].float()  # (N, V)
            s_resp = s_logits[response_positions[0], response_positions[1] - 1].float()

            t_resp /= self.temp
            s_resp /= self.temp

            t_probs = F.softmax(t_resp, dim=-1)
            s_log_probs = F.log_softmax(s_resp, dim=-1)

            loss_kd = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (self.temp ** 2)

            loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd
        else:
            loss = loss_ce

        if torch.isnan(loss) or torch.isinf(loss):
            return loss_ce

        return (loss, s_outputs) if return_outputs else loss


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",            type=str, default="tooluse")
    parser.add_argument("--teacher_model",      type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--teacher_lora_path",  type=str, default=None,
                        help="Path to fine-tuned teacher LoRA (from train_sft_qwen.py output). "
                             "If not set, uses the base teacher model without LoRA.")
    parser.add_argument("--student_model",      type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output_dir",         type=str, default="./qwen-distilled-online")
    parser.add_argument("--epochs",             type=int, default=3)
    parser.add_argument("--batch_size",         type=int, default=1)
    parser.add_argument("--grad_acc",           type=int, default=16)
    parser.add_argument("--lr",                 type=float, default=2e-4)
    parser.add_argument("--max_seq_length",     type=int, default=2048)
    parser.add_argument("--alpha",              type=float, default=0.3)
    parser.add_argument("--temp",               type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank for student.")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha scaling for student.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate for student.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for LR scheduler.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="LR scheduler type (cosine, linear, etc.).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Custom path to training dataset.")
    args = parser.parse_args()

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Teacher model — BF16 (RTX 5090 has enough memory) ────────────────────────
    print(f"Loading teacher model (BF16): {args.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if args.teacher_lora_path is not None:
        print(f"Loading teacher LoRA from: {args.teacher_lora_path}")
        teacher = PeftModel.from_pretrained(teacher, args.teacher_lora_path)
        teacher = teacher.merge_and_unload()
    teacher.eval()
    print("  Teacher ready (eval mode, frozen, BF16).")

    # ── Student model (0.8B) — BF16 ────────────────────────────────────────────
    print(f"Loading student model: {args.student_model}")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )

    student_peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, student_peft_config)
    print("  Student LoRA configured.")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    raw_path = args.dataset_path or f"sdft_repo/data/{args.dataset}_data/train_data"
    raw_dataset = load_from_disk(raw_path)
    formatted_dataset = prepare_online_distill_dataset(
        raw_dataset, args.dataset, tokenizer, args.max_seq_length
    )
    print(f"Formatted dataset: {len(formatted_dataset)} examples")

    # ── Training config ────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        packing=False,
        max_length=args.max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to="none",
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = OnlineDistillationTrainer(
        teacher_model=teacher,
        alpha=args.alpha,
        temp=args.temp,
        model=student,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
    )

    print("Starting online distillation training …")

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is None and os.path.isdir(args.output_dir):
        checkpoints = [
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_from_checkpoint = checkpoints[-1]
            print(f"Auto-resuming from latest checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Saving student LoRA to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
