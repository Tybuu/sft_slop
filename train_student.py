import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from data_loader import load_lamp_data, format_prompt
import json
from datasets import Dataset
import argparse
import os

def prepare_dataset(tokenizer, labels_path=None, questions_path='train_questions.json', outputs_path='train_outputs.json', use_ground_truth=False, max_length=512):
    # Load original questions
    questions = load_lamp_data(questions_path, outputs_path)
    
    if use_ground_truth:
        print("Using ground truth labels for baseline training.")
        label_map = {item['id']: item['target'] for item in questions}
    else:
        print(f"Using teacher labels from {labels_path} for distillation.")
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        label_map = {item['id']: item['teacher_output'] for item in labels}
    
    formatted_data = []
    for item in questions:
        if item['id'] in label_map:
            prompt = format_prompt(item)
            target = label_map[item['id']]
            # Concatenate prompt and target for CLM
            full_text = f"{prompt} {target}{tokenizer.eos_token}"
            formatted_data.append({"text": full_text})
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Train on ground truth instead of teacher labels")
    parser.add_argument("--labels", type=str, default="teacher_labels.json", help="Path to teacher labels")
    parser.add_argument("--output_dir", type=str, default="./gpt2-distilled", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_acc", type=int, default=8)
    args = parser.parse_args()

    model_id = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_id)

    print("Preparing dataset...")
    train_dataset = prepare_dataset(
        tokenizer=tokenizer, 
        labels_path=args.labels, 
        use_ground_truth=args.baseline
    )
    
    # Also prepare dev set (using ground truth for evaluation)
    dev_data = load_lamp_data('dev_questions.json', 'dev_outputs.json')
    dev_formatted = [{"text": f"{format_prompt(item)} {item['target']}{tokenizer.eos_token}"} for item in dev_data[:100]] # Limit dev for speed
    dev_dataset = Dataset.from_list(dev_formatted).map(
        lambda e: tokenizer(e["text"], truncation=True, max_length=512, padding="max_length"),
        batched=True,
        remove_columns=["text"]
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
