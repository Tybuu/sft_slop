import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json
import argparse
import os
from data_loader import load_sdft_dataset, format_prompt
import torch.nn.functional as F

class DistillationTrainer(Trainer):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, 
                 eval_dataset=None, processing_class=None, model_init=None, compute_metrics=None, 
                 callbacks=None, optimizers=(None, None), preprocess_logits_for_metrics=None,
                 teacher_logits=None, vocab_map=None, alpha=0.5, temp=1.0, item_ids_map=None):
        super().__init__(model=model, args=args, data_collator=data_collator, 
                         train_dataset=train_dataset, eval_dataset=eval_dataset, 
                         processing_class=processing_class, model_init=model_init, 
                         compute_metrics=compute_metrics, callbacks=callbacks, 
                         optimizers=optimizers, 
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.teacher_logits = teacher_logits
        self.vocab_map = vocab_map
        self.alpha = alpha
        self.temp = temp
        self.item_ids_map = item_ids_map
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

        if vocab_map is not None:
            # Build pre-mapped vocabulary tensor
            # Gemma-2-2B has 256000 tokens
            self.vocab_map_tensor = torch.full((260000,), -1, dtype=torch.long)
            for t_id_str, s_id in vocab_map.items():
                t_id = int(t_id_str)
                if t_id < 260000:
                    self.vocab_map_tensor[t_id] = s_id

        if teacher_logits is not None and vocab_map is not None:
            print("Pre-mapping teacher logits to student vocabulary...")
            for idx_str, item in self.teacher_logits.items():
                t_indices = item["top_indices"].long()
                t_indices = torch.clamp(t_indices, 0, 259999)
                item["top_indices_mapped"] = self.vocab_map_tensor[t_indices]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        item_indices = inputs.pop("item_idx", None)
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        student_logits = outputs.logits # (batch, seq, vocab)

        # Standard Cross-Entropy loss
        loss_ce = outputs.loss
        
        if self.teacher_logits is None or item_indices is None or self.item_ids_map is None:
            return (loss_ce, outputs) if return_outputs else loss_ce

        # Distillation loss
        loss_kd = 0
        valid_kd_count = 0
        device = student_logits.device
        student_vocab_size = student_logits.size(-1)
        
        for i, item_idx in enumerate(item_indices):
            idx = item_idx.item()
            idx_str = self.item_ids_map[idx]
            if idx_str not in self.teacher_logits:
                continue
                
            teacher_data = self.teacher_logits[idx_str]
            s_indices = teacher_data["s_indices"].long().to(device)
            top_probs = teacher_data["top_probs"].to(device) # (seq_t, top_k)
            top_indices_mapped = teacher_data["top_indices_mapped"].to(device) # (seq_t, top_k)

            # Filter indices that exceed current student logits sequence length
            valid_s_mask = s_indices < student_logits.size(1)
            s_indices = s_indices[valid_s_mask]
            if len(s_indices) == 0:
                continue
            
            top_probs = top_probs[valid_s_mask]
            top_indices_mapped = top_indices_mapped[valid_s_mask]
            seq_t = len(s_indices)

            # Get student log probabilities for these selected tokens
            s_logits_selected = student_logits[i, s_indices, :] # (seq_t, vocab)
            s_log_probs = F.log_softmax(s_logits_selected / self.temp, dim=-1)

            # Create mask of valid mappings
            mask = (top_indices_mapped >= 0) & (top_indices_mapped < student_vocab_size)

            # Filter out invalid indices by replacing them with 0 (dummy) and zeroing out their probability
            valid_indices = torch.where(mask, top_indices_mapped, torch.zeros_like(top_indices_mapped))
            valid_probs = torch.where(mask, top_probs, torch.zeros_like(top_probs))

            # Calculate sum of mapped probabilities before renormalization (Purity Thresholding)
            mapped_sums = valid_probs.sum(dim=-1) # (seq_t,)
            valid_step_mask = mapped_sums >= 0.5

            if not valid_step_mask.any():
                continue

            # Only compute loss for valid steps
            s_log_probs_valid = s_log_probs[valid_step_mask]
            valid_indices = valid_indices[valid_step_mask]
            valid_probs = valid_probs[valid_step_mask]
            seq_t_valid = valid_step_mask.sum().item()

            # Renormalize the mapped probabilities for valid tokens
            row_sums = valid_probs.sum(dim=-1, keepdim=True) + 1e-7
            valid_probs = valid_probs / row_sums

            # Create target distribution tensor for valid steps
            t_dist = torch.zeros(seq_t_valid, student_vocab_size, dtype=valid_probs.dtype, device=device)

            # Scatter the probabilities
            t_dist.scatter_(1, valid_indices, valid_probs)

            # KL Divergence (summed over valid tokens)
            loss_kd += F.kl_div(s_log_probs_valid, t_dist, reduction="sum") * (self.temp ** 2)
            valid_kd_count += seq_t_valid

        if valid_kd_count > 0:
            loss_kd /= valid_kd_count
            loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd
        else:
            loss = loss_ce

        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="science")
    parser.add_argument("--model_path", type=str, default="google/flan-t5-small")
    parser.add_argument("--logits", type=str, default=None)
    parser.add_argument("--vocab_map", type=str, default="vocab_map.json")
    parser.add_argument("--output_dir", type=str, default="./t5-distilled-science")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.logits is None:
        args.logits = f"teacher_logits_{args.dataset}.pt"

    print(f"Loading teacher logits from {args.logits}")
    teacher_logits_list = torch.load(args.logits, weights_only=False)
    # Convert list to dict keyed by item['id']
    teacher_logits = {item['id']: item for item in teacher_logits_list}
    
    with open(args.vocab_map, "r") as f:
        vocab_map = json.load(f)["teacher_to_student"]

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
    raw_data = load_sdft_dataset(args.dataset, 'train')
    
    from utils import format_target
    formatted_data = []
    item_ids_map = []
    for i, item in enumerate(raw_data):
        item_ids_map.append(item['id'])
        if args.dataset == "tooluse":
            target = format_target(item['target'])
        else:
            target = item['target']
            
        formatted_data.append({
            "input_text": format_prompt(item),
            "target_text": target,
            "item_idx": i
        })
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        inputs = tokenizer(examples["input_text"], truncation=True, max_length=1024, padding="max_length")
        targets = tokenizer(text_target=examples["target_text"], truncation=True, max_length=512, padding="max_length")
        
        labels = targets["input_ids"]
        labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        
        inputs["labels"] = labels
        inputs["item_idx"] = examples["item_idx"]
        return inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])

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
        fp16=False,
        gradient_checkpointing=True,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False # Crucial for custom inputs like item_id
    )

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        teacher_logits=teacher_logits,
        vocab_map=vocab_map,
        alpha=args.alpha,
        temp=args.temp,
        item_ids_map=item_ids_map
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model()

if __name__ == "__main__":
    main()
