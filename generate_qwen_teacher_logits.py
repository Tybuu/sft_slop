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
# Restrict to a single GPU on Kaggle T4x2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import load_sdft_dataset, format_prompt
from utils import format_target

TOP_K = 100


def build_example_inputs(tokenizer, prompt: str, response: str, max_length: int):
    """
    Tokenize [prompt + response] as a causal LM would see it.
    Returns input_ids and the index of the first response token so we know
    which logit positions correspond to response tokens.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # Concatenate; teacher sees the full sequence
    full_ids = prompt_ids + response_ids
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    response_start = len(prompt_ids)           # index of first response token
    response_end = min(len(prompt_ids) + len(response_ids), max_length)
    return full_ids, response_start, response_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tooluse",
                        choices=["tooluse", "science"],
                        help="Dataset to process.")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-2B",
                        help="HuggingFace model ID for the teacher.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt file path. Defaults to "
                             "teacher_logits_qwen_<dataset>.pt")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length passed to the teacher.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of examples to process per GPU forward pass.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Truncate dataset (for testing).")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the teacher model using torch.compile (can take a long time to start).")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"teacher_logits_qwen_{args.dataset}.pt"

    # ── Device ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use only a single GPU (GPU 0) for teacher inference.
    print(f"Teacher will run on: {device}")

    # ── Load teacher ─────────────────────────────────────────────────────────
    print(f"Loading teacher: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.float16,   # T4 is FP16-only
        trust_remote_code=True,
        attn_implementation="sdpa",  # scaled-dot-product attention — faster on T4
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Optional: compile teacher for faster repeated inference
    if args.compile:
        print("Compiling teacher with torch.compile …")
        teacher = torch.compile(teacher, mode="reduce-overhead", dynamic=True)

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    train_data = load_sdft_dataset(args.dataset, "train")
    if args.limit:
        train_data = train_data[: args.limit]

    # ── Resume support ────────────────────────────────────────────────────────
    results: list[dict] = []
    processed_ids: set[str] = set()
    if os.path.exists(args.output):
        results = torch.load(args.output, weights_only=False)
        processed_ids = {item["id"] for item in results}
        train_data = [ex for ex in train_data if ex["id"] not in processed_ids]
        print(f"Resuming — {len(processed_ids)} examples already done, "
              f"{len(train_data)} remaining.")

    print(f"Processing {len(train_data)} examples  (top-k = {TOP_K}) …")

    # ── Main loop ─────────────────────────────────────────────────────────────
    batch_inputs: list[dict] = []   # accumulate raw examples
    batch_meta: list[tuple] = []    # (example_id, response_start, response_end)

    def flush_batch():
        """Run one batched teacher forward and append results."""
        if not batch_inputs:
            return

        # Pad to same length for batched forward
        max_len = max(len(ex["input_ids"]) for ex in batch_inputs)
        padded_ids = []
        attention_masks = []
        for ex in batch_inputs:
            pad_len = max_len - len(ex["input_ids"])
            padded_ids.append([tokenizer.pad_token_id] * pad_len + ex["input_ids"])
            attention_masks.append([0] * pad_len + [1] * len(ex["input_ids"]))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.inference_mode():
            outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            # logits: (B, L, V)
            logits = outputs.logits

        for i, (example_id, resp_start, resp_end) in enumerate(batch_meta):
            # Offset into padded sequence
            pad_offset = max_len - len(batch_inputs[i]["input_ids"])
            # Response token logits: position [resp_start-1 … resp_end-2] predict
            # tokens at [resp_start … resp_end-1].  We shift by 1 (next-token pred).
            # padded index of the first response token prediction logit:
            first_logit_idx = pad_offset + resp_start - 1
            last_logit_idx  = pad_offset + resp_end - 1    # exclusive

            if first_logit_idx < 0 or last_logit_idx <= first_logit_idx:
                batch_inputs.clear()
                batch_meta.clear()
                continue

            resp_logits = logits[i, first_logit_idx:last_logit_idx, :]  # (L_resp, V)

            # Softmax in FP32, then take top-100
            probs = torch.softmax(resp_logits.float(), dim=-1)
            top_probs, top_indices = torch.topk(probs, TOP_K, dim=-1)   # (L_resp, 100)

            results.append({
                "id":          example_id,
                "top_probs":   top_probs.cpu().half(),     # FP16 to save disk space
                "top_indices": top_indices.cpu().to(torch.int32),
            })

        batch_inputs.clear()
        batch_meta.clear()

    save_every = 200   # checkpoint to disk every N examples
    for idx, item in enumerate(tqdm(train_data)):
        prompt = format_prompt(item)
        if args.dataset == "tooluse":
            response = format_target(item["target"])
        else:
            response = item["target"]

        full_ids, resp_start, resp_end = build_example_inputs(
            tokenizer, prompt, response, args.max_length
        )

        if resp_start >= resp_end:
            # Nothing to learn from this example (response was fully truncated)
            continue

        batch_inputs.append({"input_ids": full_ids})
        batch_meta.append((item["id"], resp_start, resp_end))

        if len(batch_inputs) >= args.batch_size:
            flush_batch()

        if (idx + 1) % save_every == 0:
            torch.save(results, args.output)
            print(f"  checkpoint: {len(results)} examples saved to {args.output}")

    # Final partial batch
    flush_batch()

    torch.save(results, args.output)
    print(f"\nDone. {len(results)} examples saved to {args.output}")


if __name__ == "__main__":
    main()
