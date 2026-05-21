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
        # 1. Forward pass on Student
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Standard Cross-Entropy loss from student
        loss_ce = student_outputs.loss

        # 2. Forward pass on Teacher (no gradients)
        teacher_inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            # Move teacher logits to student device for calculation
            teacher_logits = teacher_outputs.logits.to(student_logits.device)

        # 3. KL Divergence calculation
        # We only compute KD on the unmasked tokens. SFTTrainer uses -100 for ignored labels.
        labels = inputs.get("labels")
        if labels is not None:
            # Mask out the padding/ignored tokens
            mask = labels != -100
            
            s_logits_valid = student_logits[mask]
            t_logits_valid = teacher_logits[mask]

            if s_logits_valid.numel() > 0:
                s_log_probs = F.log_softmax(s_logits_valid / self.temp, dim=-1)
                t_probs = F.softmax(t_logits_valid / self.temp, dim=-1)

                loss_kd = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (self.temp ** 2)
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temp", type=float, default=2.0)
    args = parser.parse_args()

    # Device assignments:
    # Teacher on GPU 1 (or cpu if not available)
    # Student on GPU 0
    num_gpus = torch.cuda.device_count()
    teacher_device = "cuda:1" if num_gpus > 1 else "cuda:0"
    student_device = "cuda:0"

    print(f"Loading tokenizer from {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading Teacher Model ({args.teacher_model}) onto {teacher_device}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    ).to(teacher_device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print(f"Loading Student Model ({args.student_model}) onto {student_device}...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    ).to(student_device)

    print(f"Loading dataset: {args.dataset}")
    raw_path = f"sdft_repo/data/{args.dataset}_data/train_data"
    dataset = load_from_disk(raw_path)

    from utils import format_target
    
    def format_dataset(example):
        if args.dataset == "science":
            return {
                "messages": [
                    {"role": "system", "content": example["messages"][0]["content"]},
                    {"role": "user", "content": example["messages"][1]["content"]},
                    {"role": "assistant", "content": example["output_text"]}
                ]
            }
        elif args.dataset == "tooluse":
            target = format_target(example['golden_response'][0])
            return {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": target}
                ]
            }
        else:
            raise ValueError(f"Dataset format not supported.")

    formatted_dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)
    print(f"Formatted dataset: {len(formatted_dataset)} examples")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        max_length=args.max_seq_length,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

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

    print(f"Starting Dual-GPU On-The-Fly Distillation...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
