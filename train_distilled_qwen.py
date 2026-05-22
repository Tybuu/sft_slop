"""
train_distilled_qwen.py

Trains Qwen/Qwen3.5-0.8B (student) using knowledge distillation from
pre-computed top-100 teacher logits produced by generate_qwen_teacher_logits.py.

Two-phase run on Kaggle T4x2:
    1. python generate_qwen_teacher_logits.py --dataset tooluse
    2. python train_distilled_qwen.py --dataset tooluse

Because teacher and student share the same Qwen3.5 tokenizer there is no
cross-vocabulary mapping: teacher top-100 indices are used directly.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ── Distillation trainer ─────────────────────────────────────────────────────

class CachedLogitsDistillationTrainer(SFTTrainer):
    """
    SFTTrainer subclass that blends standard cross-entropy loss with a
    sparse KL-divergence loss computed from pre-cached teacher top-100 logits.

    The KL is computed over the entire batch at once for efficiency.
    """

    def __init__(self, teacher_logits: dict, alpha: float = 0.5,
                 temp: float = 2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_logits = teacher_logits
        self.alpha = alpha
        self.temp = temp

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop the example id we injected via the collator
        example_ids = inputs.pop("example_id", None)

        # ── Student forward ──────────────────────────────────────────────────
        outputs = model(**inputs)
        student_logits = outputs.logits   # (B, L, V)
        loss_ce = outputs.loss

        if example_ids is None or self.teacher_logits is None:
            return (loss_ce, outputs) if return_outputs else loss_ce

        # ── Vectorized KL distillation ──────────────────────────────────────
        labels = inputs.get("labels")
        if labels is None:
            return (loss_ce, outputs) if return_outputs else loss_ce

        batch_t_probs = []
        batch_t_indices = []
        batch_s_logits = []
        
        response_mask = (labels != -100) # (B, L)

        for b_idx, ex_id in enumerate(example_ids):
            ex_id_str = ex_id if isinstance(ex_id, str) else str(ex_id.item())
            if ex_id_str not in self.teacher_logits:
                continue

            t_data = self.teacher_logits[ex_id_str]
            t_probs = t_data["top_probs"]   # (L_t, 100)
            t_indices = t_data["top_indices"] # (L_t, 100)

            resp_positions = response_mask[b_idx].nonzero(as_tuple=True)[0]
            n_aligned = min(t_probs.size(0), resp_positions.size(0))

            if n_aligned > 0:
                batch_t_probs.append(t_probs[:n_aligned])
                batch_t_indices.append(t_indices[:n_aligned])
                batch_s_logits.append(student_logits[b_idx, resp_positions[:n_aligned]])

        if not batch_t_probs:
            return (loss_ce, outputs) if return_outputs else loss_ce

        # Concatenate all response tokens from the batch
        all_t_probs = torch.cat(batch_t_probs, dim=0).to(student_logits.device, non_blocking=True)
        all_t_indices = torch.cat(batch_t_indices, dim=0).to(student_logits.device, dtype=torch.long, non_blocking=True)
        all_s_logits = torch.cat(batch_s_logits, dim=0).float() # (Total_N, V)

        if self.temp != 1.0:
            t_log_scaled = torch.log(all_t_probs.clamp(min=1e-9)) / self.temp
            all_t_probs = torch.softmax(t_log_scaled, dim=-1)

        s_log_probs_full = F.log_softmax(all_s_logits / self.temp, dim=-1)
        
        V = s_log_probs_full.size(-1)
        all_t_indices_clamped = all_t_indices.clamp(0, V - 1)
        s_log_probs_sparse = s_log_probs_full.gather(1, all_t_indices_clamped)

        # KL: Token-level average
        loss_kd = F.kl_div(
            s_log_probs_sparse,
            all_t_probs,
            reduction="batchmean",
        ) * (self.temp ** 2)

        loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd

        return (loss, outputs) if return_outputs else loss



# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_formatted_dataset(raw_hf_dataset, dataset_name: str):
    """Convert raw HF dataset rows into {messages, example_id} dicts."""
    from utils import format_target

    def _format(examples):
        out_messages = []
        out_ids = []

        if dataset_name == "science":
            for i, (msgs, out) in enumerate(
                zip(examples["messages"], examples["output_text"])
            ):
                out_messages.append([
                    {"role": "system",    "content": msgs[0]["content"]},
                    {"role": "user",      "content": msgs[1]["content"]},
                    {"role": "assistant", "content": out},
                ])
                out_ids.append(f"science_train_{i}")

        elif dataset_name == "tooluse":
            for i, (prompt, golden) in enumerate(
                zip(examples["prompt"], examples["golden_response"])
            ):
                target = format_target(golden[0])
                out_messages.append([
                    {"role": "system",    "content": "You are a helpful assistant."},
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": target},
                ])
                out_ids.append(f"tooluse_train_{i}")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        return {"messages": out_messages, "example_id": out_ids}

    print(f"Mapping dataset (num_proc=1 for stability) …")
    return raw_hf_dataset.map(
        _format,
        batched=True,
        batch_size=256,
        remove_columns=raw_hf_dataset.column_names,
        num_proc=1, # Multiprocessing often hangs on Kaggle/Colab
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str,   default="tooluse")
    parser.add_argument("--student_model", type=str,   default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--logits_file",   type=str,   default=None,
                        help="Path to .pt file from generate_qwen_teacher_logits.py. "
                             "Defaults to teacher_logits_qwen_<dataset>.pt")
    parser.add_argument("--output_dir",    type=str,   default="./qwen-distilled-tooluse")
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch_size",    type=int,   default=2)
    parser.add_argument("--grad_acc",      type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=5e-5)
    parser.add_argument("--max_seq_length",type=int,   default=1024)
    parser.add_argument("--alpha",         type=float, default=0.8,
                        help="Weight of KD loss (0 = pure CE, 1 = pure KD).")
    parser.add_argument("--temp",          type=float, default=2.0,
                        help="Distillation temperature.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Specific checkpoint path to resume from.")
    args = parser.parse_args()

    if args.logits_file is None:
        args.logits_file = f"teacher_logits_qwen_{args.dataset}.pt"

    # ── Load cached teacher logits ───────────────────────────────────────────
    if not os.path.exists(args.logits_file):
        raise FileNotFoundError(
            f"Teacher logits file not found: {args.logits_file}\n"
            f"Run generate_qwen_teacher_logits.py first."
        )
    print(f"Loading teacher logits from {args.logits_file} …")
    logits_list = torch.load(args.logits_file, weights_only=False)
    print(f"  Raw logits loaded. Converting to dictionary …")
    # Key by id for O(1) lookup in compute_loss
    teacher_logits = {item["id"]: item for item in logits_list}
    print(f"  {len(teacher_logits)} examples loaded and indexed.")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Student model ─────────────────────────────────────────────────────────
    print(f"Loading student: {args.student_model}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.float16,   # T4 is FP16-only (no BF16 support)
        trust_remote_code=True,
        attn_implementation="sdpa",  # Fast path for T4
        device_map="auto",           # Ensure model is on GPU
    )
    print("  Student model loaded.")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    raw_path = f"sdft_repo/data/{args.dataset}_data/train_data"
    raw_dataset = load_from_disk(raw_path)
    formatted_dataset = build_formatted_dataset(raw_dataset, args.dataset)
    print(f"Formatted dataset: {len(formatted_dataset)} examples")

    # ── Training config ───────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        # T4 has no BF16 hardware support
        bf16=False,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        # 8-bit Adam halves optimizer-state memory (~2 GB saved)
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Data loading (set to 0 for Kaggle stability)
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        # Disable packing so we preserve 'example_id' for teacher logits lookup
        packing=False,
        max_length=args.max_seq_length,
        # Keep example_id column so compute_loss can look up teacher logits
        dataset_kwargs={"skip_prepare_dataset": False},
        remove_unused_columns=False,
        report_to="none",
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = CachedLogitsDistillationTrainer(
        teacher_logits=teacher_logits,
        alpha=args.alpha,
        temp=args.temp,
        model=student_model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting distillation training …")
    
    # Check for existing checkpoints to resume from
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is None and os.path.isdir(args.output_dir):
        checkpoints = [
            os.path.join(args.output_dir, d) 
            for d in os.listdir(args.output_dir) 
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            # Sort by checkpoint number
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_from_checkpoint = checkpoints[-1]
            print(f"Auto-resuming from latest checkpoint: {resume_from_checkpoint}")
    elif resume_from_checkpoint is not None:
        print(f"Resuming from specified checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
