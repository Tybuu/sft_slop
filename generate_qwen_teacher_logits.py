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
# Restrict to a single GPU by default, but allow user override via environment variable
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data_loader import load_sdft_dataset, format_prompt
from utils import format_target

TOP_K = 100

class TeacherWithTopK(torch.nn.Module):
    """
    Wrapper module to compute softmax and top-k on each GPU locally.
    This prevents gathering the massive (B, L, V) logits tensor back to GPU 0,
    which causes out-of-memory (OOM) errors and severe communication overhead.
    """
    def __init__(self, model, top_k=TOP_K):
        super().__init__()
        self.model = model
        self.top_k = top_k

    def forward(self, input_ids, attention_mask):
        # Disable cache to optimize memory during full-sequence forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits # (B, L, V)
        
        # Softmax is monotonic (preserves relative order), so top-k logits index matches top-k probs index.
        # Computing softmax ONLY on the top-k logits uses ~1500x less memory than the full vocab,
        # which completely avoids CUDA OOMs and is mathematically identical after trainer normalization.
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_probs = torch.softmax(top_logits.float(), dim=-1).half()
        
        return top_probs, top_indices.to(torch.int32)

def build_example_inputs(tokenizer, prompt: str, response: str, max_length: int):
    """
    Tokenize using the Qwen chat template to match SFT training.
    """
    # SYSTEM PROMPT: Matches eval.py and train_sft_qwen.py exactly
    SYSTEM_PROMPT = "You are a helpful assistant."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    # We need to find where the assistant response starts
    # Get prompt-only tokens
    prompt_msgs = messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True)
    
    # Get full tokens
    full_ids = tokenizer.apply_chat_template(messages)
    
    # Handle BatchEncoding or dict return types from apply_chat_template
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    elif isinstance(prompt_ids, dict) and "input_ids" in prompt_ids:
        prompt_ids = prompt_ids["input_ids"]
        
    if hasattr(full_ids, "input_ids"):
        full_ids = full_ids.input_ids
    elif isinstance(full_ids, dict) and "input_ids" in full_ids:
        full_ids = full_ids["input_ids"]

    # Convert tensors to list if needed
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    resp_start = len(prompt_ids)
    resp_end = min(len(full_ids), max_length)
    
    return full_ids, resp_start, resp_end

class LogitDataset(Dataset):
    def __init__(self, data, tokenizer, dataset_name, max_length):
        self.features = []
        print(f"Pre-tokenizing dataset (size: {len(data)}) to prevent CPU bottlenecks...")
        for item in tqdm(data, desc="Tokenizing"):
            prompt = format_prompt(item)
            if dataset_name == "tooluse":
                response = format_target(item["target"])
            else:
                response = item["target"]

            full_ids, resp_start, resp_end = build_example_inputs(
                tokenizer, prompt, response, max_length
            )
            self.features.append({
                "id": item["id"],
                "input_ids": full_ids,
                "resp_start": resp_start,
                "resp_end": resp_end
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

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

    # Quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    teacher.eval()
    
    # Wrap model to compute top-K probabilities locally on each GPU
    teacher = TeacherWithTopK(teacher, TOP_K)
    
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
            # Wrapper returns top_probs and top_indices directly
            top_probs, top_indices = teacher(input_ids=input_ids, attention_mask=attention_mask)

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

            # Slice from pre-computed top_probs and top_indices
            example_top_probs = top_probs[i, first_logit_idx:last_logit_idx, :]
            example_top_indices = top_indices[i, first_logit_idx:last_logit_idx, :]

            results.append({
                "id":          example_id,
                "top_probs":   example_top_probs.cpu(),
                "top_indices": example_top_indices.cpu(),
            })

        if (idx + 1) % (save_every // args.batch_size + 1) == 0:
            torch.save(results, args.output)

    torch.save(results, args.output)
    print(f"\nDone. {len(results)} examples saved.")


if __name__ == "__main__":
    main()
