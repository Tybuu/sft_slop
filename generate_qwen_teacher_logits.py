"""
generate_qwen_teacher_logits.py

Pre-computes top-100 teacher logits from Qwen/Qwen3.5-2B for each training
example and saves them to disk.  Because teacher and student share the same
Qwen3.5 tokenizer there is no cross-vocab alignment step.

Usage:
    python generate_qwen_teacher_logits.py \
        --dataset tooluse \
        --output teacher_logits_qwen_tooluse.pt

The output file is a list of dicts:
    {
        "id":        str,              # example id, e.g. "tooluse_train_0"
        "top_probs": Tensor[L, 100],  # FP16, already softmax'd
        "top_indices": Tensor[L, 100] # INT32 token indices in shared vocab
    }
where L is the number of non-padding (non-prompt) response tokens.
"""

import argparse
import os
# Restrict to a single GPU for maximum speed on small models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import load_sdft_dataset, format_prompt
from utils import format_target

TOP_K = 100

class LogitDataset(Dataset):
    def __init__(self, data, tokenizer, dataset_name, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = format_prompt(item)
        if self.dataset_name == "tooluse":
            response = format_target(item["target"])
        else:
            response = item["target"]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        full_ids = prompt_ids + response_ids
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        resp_start = len(prompt_ids)
        resp_end = min(len(prompt_ids) + len(response_ids), self.max_length)
        
        return {
            "id": item["id"],
            "input_ids": full_ids,
            "resp_start": resp_start,
            "resp_end": resp_end
        }

def collate_fn(batch, tokenizer):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    padded_ids = []
    attention_masks = []
    ids = []
    resp_starts = []
    resp_ends = []
    original_lens = []

    for ex in batch:
        original_len = len(ex["input_ids"])
        pad_len = max_len - original_len
        # Left padding is standard for generation, but here we just need a batch
        padded_ids.append([tokenizer.pad_token_id] * pad_len + ex["input_ids"])
        attention_masks.append([0] * pad_len + [1] * original_len)
        ids.append(ex["id"])
        resp_starts.append(ex["resp_start"])
        resp_ends.append(ex["resp_end"])
        original_lens.append(original_len)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "ids": ids,
        "resp_starts": torch.tensor(resp_starts, dtype=torch.long),
        "resp_ends": torch.tensor(resp_ends, dtype=torch.long),
        "original_lens": torch.tensor(original_lens, dtype=torch.long),
        "max_len": max_len
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tooluse",
                        choices=["tooluse", "science"],
                        help="Dataset to process.")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-2B",
                        help="HuggingFace model ID for the teacher.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt file path.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU(s).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Truncate dataset.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers. Set to 0 to avoid hangs on Kaggle.")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the teacher model.")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"teacher_logits_qwen_{args.dataset}.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, GPUs: {torch.cuda.device_count()}")

    print(f"Loading teacher: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    teacher.eval()
    
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        teacher = torch.nn.DataParallel(teacher)

    if args.compile:
        print("Compiling teacher …")
        teacher = torch.compile(teacher, mode="reduce-overhead", dynamic=True)

    print(f"Loading dataset: {args.dataset}")
    train_data = load_sdft_dataset(args.dataset, "train")
    if args.limit:
        train_data = train_data[: args.limit]

    results: list[dict] = []
    processed_ids: set[str] = set()
    if os.path.exists(args.output):
        results = torch.load(args.output, weights_only=False)
        processed_ids = {item["id"] for item in results}
        train_data = [ex for ex in train_data if ex["id"] not in processed_ids]
        print(f"Resuming — {len(processed_ids)} done, {len(train_data)} remaining.")

    dataset = LogitDataset(train_data, tokenizer, args.dataset, args.max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        pin_memory=True
    )

    save_every = 500
    print(f"Processing {len(train_data)} examples with batch_size={args.batch_size} …")

    for idx, batch in enumerate(tqdm(dataloader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.inference_mode():
            outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits # (B, L, V)

        max_len = batch["max_len"]
        for i in range(len(batch["ids"])):
            example_id = batch["ids"][i]
            resp_start = batch["resp_starts"][i].item()
            resp_end = batch["resp_ends"][i].item()
            original_len = batch["original_lens"][i].item()
            
            pad_offset = max_len - original_len
            first_logit_idx = pad_offset + resp_start - 1
            last_logit_idx  = pad_offset + resp_end - 1

            if first_logit_idx < 0 or last_logit_idx <= first_logit_idx:
                continue

            resp_logits = logits[i, first_logit_idx:last_logit_idx, :]
            probs = torch.softmax(resp_logits.float(), dim=-1)
            top_probs, top_indices = torch.topk(probs, TOP_K, dim=-1)

            results.append({
                "id":          example_id,
                "top_probs":   top_probs.cpu().half(),
                "top_indices": top_indices.cpu().to(torch.int32),
            })

        if (idx + 1) % (save_every // args.batch_size + 1) == 0:
            torch.save(results, args.output)

    torch.save(results, args.output)
    print(f"\nDone. {len(results)} examples saved.")



if __name__ == "__main__":
    main()
