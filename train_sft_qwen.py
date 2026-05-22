import argparse
import os
# Restrict to a single GPU for maximum speed on small models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tooluse")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output_dir", type=str, default="./qwen-sft-tooluse")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Specific checkpoint path to resume from.")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Qwen tokenizer needs a pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16, # T4 is FP16-only
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa", # Force the fast/memory-efficient path
    )

    print(f"Loading dataset: {args.dataset}")
    raw_path = f"sdft_repo/data/{args.dataset}_data/train_data"
    print(f"Loading raw dataset from {raw_path}")
    dataset = load_from_disk(raw_path)

    # Format dataset for conversational SFT
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
            raise ValueError(f"Dataset {args.dataset} format not supported.")

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
        fp16=True, # T4 doesn't support BF16
        gradient_checkpointing=True,
        report_to="none",
        max_length=args.max_seq_length,
        packing=True, # Pack multiple examples into one sequence for speed
        optim="adamw_8bit",
        dataloader_num_workers=0,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting Normal Fine-Tuning (SFT) on Qwen...")
    
    # Check for existing checkpoints to resume from
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
    elif resume_from_checkpoint is not None:
        print(f"Resuming from specified checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
