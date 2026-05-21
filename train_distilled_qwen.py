import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


class OnTheFlyDistillationTrainer(SFTTrainer):
    def __init__(self, teacher_model, alpha=0.5, temp=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temp = temp

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ── 1. Student forward ──────────────────────────────────────────────
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits   # (B, L, V_student)
        loss_ce = student_outputs.loss

        # ── 2. Teacher forward (no grad, inference_mode for max speed) ──────
        # non_blocking=True overlaps PCIe transfer with GPU work
        teacher_inputs = {
            k: v.to(self.teacher_model.device, non_blocking=True)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            # Move teacher logits back to student device; keep as FP16 to save BW
            teacher_logits = teacher_outputs.logits.to(
                student_logits.device, non_blocking=True
            )

        # ── 3. Full-vocab KL divergence on unmasked (non-padding) tokens ────
        labels = inputs.get("labels")
        if labels is not None:
            mask = labels != -100                        # (B, L)

            s_logits_valid = student_logits[mask]        # (N, V_student)
            t_logits_valid = teacher_logits[mask]        # (N, V_teacher)

            if s_logits_valid.numel() > 0:
                # Truncate teacher vocab to student vocab size if needed
                v_student = s_logits_valid.size(-1)
                if t_logits_valid.size(-1) > v_student:
                    t_logits_valid = t_logits_valid[..., :v_student]

                # Compute softmax / log-softmax in fp32 for numerical stability
                s_log_probs = F.log_softmax(
                    s_logits_valid.float() / self.temp, dim=-1
                )
                t_probs = F.softmax(
                    t_logits_valid.float() / self.temp, dim=-1
                )

                loss_kd = (
                    F.kl_div(s_log_probs, t_probs, reduction="batchmean")
                    * (self.temp ** 2)
                )
                loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd
            else:
                loss = loss_ce
        else:
            loss = loss_ce

        return (loss, student_outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tooluse")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output_dir", type=str, default="./qwen-distilled-tooluse")
    parser.add_argument("--epochs", type=int, default=3)
    # T4 has 16 GB VRAM; batch_size=2 with LoRA+gc fits comfortably
    parser.add_argument("--batch_size", type=int, default=2)
    # Keep effective batch = 16; halve grad_acc since batch doubled
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=2.0)
    # Compile teacher with torch.compile for CUDA graph / kernel fusion speedup
    parser.add_argument("--compile_teacher", action="store_true", default=True)
    args = parser.parse_args()

    # ── Device assignments ──────────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    teacher_device = "cuda:1" if num_gpus > 1 else "cuda:0"
    student_device = "cuda:0"

    # T4 does NOT support BF16; force FP16 explicitly
    dtype = torch.float16

    # ── Tokenizer ───────────────────────────────────────────────────────────
    print(f"Loading tokenizer from {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Teacher model ────────────────────────────────────────────────────────
    print(f"Loading Teacher ({args.teacher_model}) → {teacher_device} ...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(teacher_device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # torch.compile fuses kernels and builds CUDA graphs; dynamic=True handles
    # variable sequence lengths without recompilation storms.
    if args.compile_teacher:
        print("Compiling teacher with torch.compile (dynamic=True, reduce-overhead)...")
        teacher_model = torch.compile(
            teacher_model,
            mode="reduce-overhead",
            dynamic=True,
        )

    # ── Student model ─────────────────────────────────────────────────────────
    print(f"Loading Student ({args.student_model}) → {student_device} ...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(student_device)

    # ── Dataset ────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    raw_path = f"sdft_repo/data/{args.dataset}_data/train_data"
    dataset = load_from_disk(raw_path)

    from utils import format_target

    def format_dataset(examples):
        """Batched map: faster than per-example mapping."""
        results = {"messages": []}
        if args.dataset == "science":
            for msgs, out in zip(examples["messages"], examples["output_text"]):
                results["messages"].append([
                    {"role": "system",    "content": msgs[0]["content"]},
                    {"role": "user",      "content": msgs[1]["content"]},
                    {"role": "assistant", "content": out},
                ])
        elif args.dataset == "tooluse":
            for prompt, golden in zip(examples["prompt"], examples["golden_response"]):
                target = format_target(golden[0])
                results["messages"].append([
                    {"role": "system",    "content": "You are a helpful assistant."},
                    {"role": "user",      "content": prompt},
                    {"role": "assistant", "content": target},
                ])
        else:
            raise ValueError(f"Dataset format not supported: {args.dataset}")
        return results

    formatted_dataset = dataset.map(
        format_dataset,
        batched=True,              # vectorised → much faster
        batch_size=256,
        remove_columns=dataset.column_names,
        num_proc=4,                # parallel CPU workers
    )
    print(f"Formatted dataset: {len(formatted_dataset)} examples")

    # ── Training config ──────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        # Force FP16 — T4 has no BF16 hardware support
        bf16=False,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,           # keep only the latest checkpoint
        # 8-bit Adam: halves optimizer state memory (~2 GB saved on T4)
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required for LoRA
        # Data loading
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Pack multiple short examples per sequence → less padding waste
        packing=True,
        max_length=args.max_seq_length,
        report_to="none",
    )

    # ── LoRA config ──────────────────────────────────────────────────────
    # Expand to all linear projections for better gradient flow
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",   # MLP projections
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = OnTheFlyDistillationTrainer(
        teacher_model=teacher_model,
        alpha=args.alpha,
        temp=args.temp,
        model=student_model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting Dual-GPU On-The-Fly Distillation...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
