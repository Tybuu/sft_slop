import argparse
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from data_loader import load_sdft_dataset, format_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tooluse")
    parser.add_argument("--model_path", type=str, default="google/flan-t5-small")
    parser.add_argument("--output_dir", type=str, default="./t5-sft-tooluse")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading model and tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    # T5 tokenizer does not natively support '<' as a token, resulting in <unk>.
    # Add explicit XML tags as custom vocabulary tokens to avoid mangling and enable perfect tokenization.
    special_tokens = ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]
    num_added_toks = tokenizer.add_tokens(special_tokens)
    if num_added_toks > 0:
        print(f"Added {num_added_toks} special tokens to the tokenizer.")
        model.resize_token_embeddings(len(tokenizer))


    print(f"Loading dataset: {args.dataset}")
    raw_train_data = load_sdft_dataset(args.dataset, 'train')
    
    from utils import format_target
    
    # Format data for HuggingFace Dataset
    train_list = []
    for item in raw_train_data:
        if args.dataset == "tooluse":
            target = format_target(item['target'])
        else:
            target = item['target']
            
        train_list.append({
            "input_text": format_prompt(item),
            "target_text": target
        })
    
    train_dataset = Dataset.from_list(train_list)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"], 
            max_length=args.max_input_length, 
            truncation=True, 
            padding="max_length"
        )

        labels = tokenizer(
            text_target=examples["target_text"], 
            max_length=args.max_target_length, 
            truncation=True, 
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in model_inputs["labels"]
        ]
        
        # Debug: Print a non-empty label count for the first batch
        if not hasattr(preprocess_function, "debug_done"):
            non_pad = [l for l in model_inputs["labels"][0] if l != -100]
            print(f"Debug: Example 0 has {len(non_pad)} non-pad tokens in target.")
            preprocess_function.debug_done = True
            
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_train = train_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=["input_text", "target_text"]
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        fp16=False, # T5 is unstable with FP16
        gradient_checkpointing=True,
        push_to_hub=False,
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    print("Starting Normal Fine-Tuning (SFT)...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
