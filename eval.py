import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data_loader import load_sdft_dataset, format_prompt
from peft import PeftModel
import json
from tqdm import tqdm
import argparse
import re
import os

def extract_xml_answer(text: str) -> str:
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = text.strip()
    if len(text) == 1:
        return text
    match = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text

def extract_actions(text):
    return re.findall(r'Action:\s*(\w+)', text)

def extract_action_inputs(text):
    matches = re.findall(r'Action Input:\s*(.*?)(?=Action:|$)', text, re.DOTALL)

    merged_inputs = {}
    for content in matches:
        content = content.strip()
        if not content:
            continue
        if not content.startswith('{') and ':' in content:
            content = '{' + content + '}'
        try:
            json_match = re.search(r'({.*?})', content, re.DOTALL)
            if json_match:
                c = json_match.group(1)
                merged_inputs.update(json.loads(c))
        except:
            pass
    return merged_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model or LoRA adapter.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-0.8B",
                        help="Base model for LoRA adapter.")
    parser.add_argument("--dataset", type=str, default="science")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load base model in 4-bit (for QLoRA adapters).")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking mode for Qwen3.5.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device}")

    print(f"Loading tokenizer from {args.model_path} (or {args.base_model})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.base_model}")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    ).to(device)

    if args.model_path != args.base_model:
        print(f"Loading LoRA adapter from {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        print("Merging LoRA weights for faster inference...")
        model = model.merge_and_unload()
        model = model.to(device)

    model.eval()
    print(f"Model is on: {model.device}")

    print(f"Loading evaluation data: {args.dataset}")
    eval_data = load_sdft_dataset(args.dataset, 'eval')
    if args.limit:
        eval_data = eval_data[:args.limit]

    from datasets import load_from_disk
    raw_eval_ds = load_from_disk(f'sdft_repo/data/{args.dataset}_data/eval_data')

    results = []
    correct = 0

    print(f"Running evaluation (batch_size={args.batch_size}) ...")

    for i in tqdm(range(0, len(eval_data), args.batch_size)):
        batch_items = eval_data[i : i + args.batch_size]

        formatted_prompts = []
        for b_idx, item in enumerate(batch_items):
            global_idx = i + b_idx
            if args.dataset == "tooluse":
                system_content = "You are a helpful assistant."
            else:
                system_content = raw_eval_ds[global_idx]['prompt'][0]['content']

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": format_prompt(item)}
            ]
            formatted_prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
            )

        inputs = tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(device)

        with torch.no_grad():
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stop_strings=["<|im_start|>", "<|im_end|>", "User:", "Question:"],
                tokenizer=tokenizer,
            )
            if args.enable_thinking:
                gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
            else:
                gen_kwargs["do_sample"] = False
            outputs = model.generate(**gen_kwargs)

            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]
            batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for j, prediction in enumerate(batch_predictions):
            clean_prediction = prediction.split("Question:")[0].split("User:")[0].strip()

            item = batch_items[j]
            target = item['target']
            item_idx = i + j

            if args.dataset == "science":
                extracted_pred = extract_xml_answer(clean_prediction)
                is_correct = (extracted_pred.lower() == target.lower())
            elif args.dataset == "tooluse":
                from collections import Counter
                pred_actions = extract_actions(clean_prediction)
                pred_inputs = extract_action_inputs(clean_prediction)

                golden_answer = raw_eval_ds[item_idx]['golden_answer']
                gt_actions = [ga['Action'] for ga in golden_answer]

                max_allowed = max(len(gt_actions) * 2, 10)
                if len(pred_actions) > max_allowed:
                    pred_actions = pred_actions[:len(gt_actions)]

                gt_inputs = {}
                for ga in golden_answer:
                    try:
                        loaded = json.loads(ga['Action_Input'])
                        filtered_loaded = {k: v for k, v in loaded.items() if v != "" and v is not None}
                        gt_inputs.update(filtered_loaded)
                    except:
                        pass

                filtered_pred_inputs = {k: v for k, v in pred_inputs.items() if v != "" and v is not None}

                actions_match = Counter(pred_actions) == Counter(gt_actions)
                inputs_match = filtered_pred_inputs == gt_inputs
                is_correct = actions_match and inputs_match
                extracted_pred = f"Actions: {pred_actions}, Inputs: {pred_inputs}"

            if is_correct:
                correct += 1

            results.append({
                'id': item['id'],
                'prompt': batch_items[j]['input'],
                'target': target,
                'prediction': clean_prediction,
                'extracted_pred': extracted_pred,
                'correct': is_correct
            })

    accuracy = correct / len(eval_data) if eval_data else 0
    print(f"\nResults for {args.model_path} on {args.dataset}:")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(eval_data)})")

    output_name = os.path.basename(args.model_path.rstrip('/'))
    with open(f"eval_results_{args.dataset}_{output_name}.json", "w") as f:
        json.dump({
            "model": args.model_path,
            "accuracy": accuracy,
            "num_correct": correct,
            "num_total": len(eval_data),
            "results": results
        }, f, indent=2)

if __name__ == "__main__":
    main()
