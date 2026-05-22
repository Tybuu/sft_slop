import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from data_loader import load_sdft_dataset, format_prompt
from peft import PeftModel
import json
from tqdm import tqdm
import argparse
import re
import os

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    # Look for <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: just take the last part if it looks like a single letter
    text = text.strip()
    if len(text) == 1:
        return text
    # Sometimes it might be just "A" or "Answer: A"
    match = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text

def extract_actions(text):
    """Extract all actions from model response."""
    return re.findall(r'Action:\s*(\w+)', text)

def extract_action_inputs(text):
    """Extract and merge all action inputs from model response."""
    # Find all "Action Input: ..." segments
    matches = re.findall(r'Action Input:\s*(.*?)(?=Action:|$)', text, re.DOTALL)
    
    merged_inputs = {}
    for content in matches:
        content = content.strip()
        if not content:
            continue
            
        # Heuristic: if it's missing brackets, add them
        if not content.startswith('{') and ':' in content:
            content = '{' + content + '}'
            
        try:
            # Find the first json-like block
            json_match = re.search(r'({.*?})', content, re.DOTALL)
            if json_match:
                # Basic cleanup for repeating keys or words
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
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    # Force single GPU for evaluation of small models to avoid communication overhead/offloading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device}")
    
    print(f"Loading tokenizer from {args.model_path} (or {args.base_model})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # Don't use device_map="auto" for 0.8B - it often offloads to CPU incorrectly
        # We will move it manually to ensure it's on the GPU
        attn_implementation="sdpa"
    ).to(device)

    # If model_path is different from base_model, we assume it's a LoRA adapter
    if args.model_path != args.base_model:
        print(f"Loading LoRA adapter from {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        print("Merging LoRA weights for faster inference...")
        model = model.merge_and_unload()
        model = model.to(device) # Re-ensure it's on GPU after merge
    
    model.eval()
    print(f"Model is on: {model.device}")

    print(f"Loading evaluation data: {args.dataset}")
    eval_data = load_sdft_dataset(args.dataset, 'eval')
    if args.limit:
        eval_data = eval_data[:args.limit]

    # For ToolUse, we need the original dataset items for golden_answer
    raw_eval_ds = None
    if args.dataset == "tooluse":
        from datasets import load_from_disk
        raw_eval_ds = load_from_disk('sdft_repo/data/tooluse_data/eval_data')

    results = []
    correct = 0

    print("Running evaluation...")
    for i, item in enumerate(tqdm(eval_data)):
        prompt = format_prompt(item)
        target = item['target']
        
        # Format with chat template if needed (optional, but keep it consistent with training)
        # For now, keep the simple format_prompt from data_loader
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_tokens, 
                do_sample=False,
                repetition_penalty=1.1, # Slightly lower to avoid degrading Qwen
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt
            if full_text.startswith(prompt):
                prediction = full_text[len(prompt):].strip()
            else:
                prediction = full_text.strip()
        
        if args.dataset == "science":
            extracted_pred = extract_xml_answer(prediction)
            is_correct = (extracted_pred.lower() == target.lower())
        elif args.dataset == "tooluse":
            from collections import Counter
            pred_actions = extract_actions(prediction)
            pred_inputs = extract_action_inputs(prediction)
            
            # Use raw_eval_ds for golden_answer
            golden_answer = raw_eval_ds[i]['golden_answer']
            gt_actions = [ga['Action'] for ga in golden_answer]
            
            # Merge all ground truth inputs and filter out empty strings for lenient comparison
            gt_inputs = {}
            for ga in golden_answer:
                try:
                    loaded = json.loads(ga['Action_Input'])
                    # Only keep non-empty string values for comparison
                    filtered_loaded = {k: v for k, v in loaded.items() if v != "" and v is not None}
                    gt_inputs.update(filtered_loaded)
                except:
                    pass
            
            # Filter predicted inputs too
            filtered_pred_inputs = {k: v for k, v in pred_inputs.items() if v != "" and v is not None}
            
            actions_match = Counter(pred_actions) == Counter(gt_actions)
            inputs_match = filtered_pred_inputs == gt_inputs
            is_correct = actions_match and inputs_match
            extracted_pred = f"Actions: {pred_actions}, Inputs: {pred_inputs}"
        
        if is_correct:
            correct += 1
            
        results.append({
            'id': item['id'],
            'prompt': prompt,
            'target': target,
            'prediction': prediction,
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
