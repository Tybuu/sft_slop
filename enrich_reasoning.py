"""
enrich_reasoning.py

For each training example, use the teacher model to generate a detailed
reasoning trace explaining WHY the action/parameters were chosen,
replacing the shallow "I need to use X tool" with genuine reasoning.

Usage:
    python enrich_reasoning.py --teacher_model Qwen/Qwen3.5-27B-FP8 --batch_size 1

Output: data/reasoning_dataset/
"""

import argparse
import os
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def build_reasoning_prompt(tool_prompt, instruction, action, action_input):
    return f"""Given a user query and available tools, explain step by step why you would choose a specific action and its input parameters.

AVAILABLE TOOLS:
{tool_prompt}

USER QUERY:
{instruction}

CORRECT ACTION:
Action: {action}
Action Input: {action_input}

Now provide a detailed reasoning trace explaining WHY this action and input are correct. Consider:
- What does the user want?
- Which tool function should be used and why?
- How do the parameters map to the user's request?

Your reasoning (be detailed and thorough):"""


def extract_reasoning(text):
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith('action:') or stripped.lower().startswith('action input:'):
            break
        if stripped.lower().startswith('your reasoning'):
            continue
        if stripped:
            filtered.append(stripped)
    return ' '.join(filtered) if filtered else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset_path", type=str, default="data/tooluse_data/train_data_fixed")
    parser.add_argument("--output_dir", type=str, default="data/reasoning_dataset")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading teacher model: {args.teacher_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    print("Teacher loaded.")

    print(f"Loading dataset: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    if args.limit:
        ds = ds.select(range(args.limit))
    print(f"Dataset: {len(ds)} examples")

    def process_batch(start_idx):
        batch = ds.select(range(start_idx, min(start_idx + args.batch_size, len(ds))))
        prompts = []
        for ex in batch:
            action = ex['golden_answer'][0]['Action']
            action_input = ex['golden_answer'][0]['Action_Input']
            prompts.append(build_reasoning_prompt(ex['prompt'], ex['instruction'], action, action_input))

        messages_batch = []
        for p in prompts:
            messages_batch.append([
                {"role": "user", "content": p},
            ])

        texts = tokenizer.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        generated = outputs[:, input_len:]
        raw_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

        enriched = []
        for raw in raw_outputs:
            reasoning = extract_reasoning(raw)
            enriched.append(reasoning)

        return enriched

    enriched_reasoning = [None] * len(ds)

    for start in tqdm(range(0, len(ds), args.batch_size), desc="Enriching reasoning"):
        enriched = process_batch(start)
        for i, r in enumerate(enriched):
            enriched_reasoning[start + i] = r
        if start % 10 == 0:
            torch.cuda.empty_cache()

    def build_new_example(original, reasoning):
        response = original['golden_response'][0]
        if 'Action:' in response:
            parts = response.split('Action:', 1)
            action_part = 'Action:' + parts[1]
        else:
            action_part = response

        new_response = f"Thought: {reasoning}\n{action_part.strip()}"
        while '\n\n' in new_response:
            new_response = new_response.replace('\n\n', '\n')

        return {
            **dict(original),
            "golden_response": [new_response],
        }

    new_data = []
    for i in tqdm(range(len(ds)), desc="Building output"):
        new_data.append(build_new_example(ds[i], enriched_reasoning[i]))

    new_ds = Dataset.from_list(new_data)
    new_ds.save_to_disk(args.output_dir)
    print(f"Saved enriched dataset ({len(new_ds)} examples) to {args.output_dir}")

    old = ds[0]['golden_response'][0]
    new = new_ds[0]['golden_response'][0]
    print("\n=== Before ===")
    print(old[:300])
    print("\n=== After ===")
    print(new[:600])


if __name__ == "__main__":
    main()
