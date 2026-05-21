import argparse
import os
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    print(f"Loading dataset: {args.dataset}")
    raw_path = f"sdft_repo/data/{args.dataset}_data/train_data"
    print(f"Loading raw dataset from {raw_path}")
    dataset = load_from_disk(raw_path)

    # Format dataset for conversational SFT
    # Qwen expects a list of messages. We'll build the conversation history:
    # System prompt, User question, Assistant answer (reasoning + final choice).
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

    # Use standard SFTTrainer logic for causal LMs
    # In newer TRL versions, the completion only data collator is removed or implicit.
    # We will use the default data collator provided by SFTTrainer.

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

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting Normal Fine-Tuning (SFT) on Qwen...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
