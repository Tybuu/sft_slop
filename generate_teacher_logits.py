import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from data_loader import load_sdft_dataset, format_prompt
import json
from tqdm import tqdm
import os
import argparse

def get_token_alignments(student_tokenizer, teacher_tokenizer, prompt, target):
    """
    Finds the mapping from student decoder token indices to teacher token indices.
    We align the 'target' part only.
    """
    # Teacher sees full text
    full_text = prompt + target
    t_enc = teacher_tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    t_offsets = t_enc['offset_mapping']
    
    # Student decoder sees target
    s_enc = student_tokenizer(target, return_offsets_mapping=True, add_special_tokens=False)
    s_offsets = s_enc['offset_mapping']
    
    # Find where target starts in teacher offsets
    prompt_len = len(prompt)
    t_target_start_idx = 0
    for idx, (start, end) in enumerate(t_offsets):
        if start >= prompt_len:
            t_target_start_idx = idx
            break
            
    # map: student_decoder_idx -> teacher_idx (where student token ends)
    alignment = {}
    
    t_ptr = t_target_start_idx
    for s_idx, (s_start, s_end) in enumerate(s_offsets):
        # s_start and s_end are relative to 'target'. We make them relative to 'full_text'
        s_end_abs = s_end + prompt_len
        
        while t_ptr < len(t_offsets) and t_offsets[t_ptr][1] < s_end_abs:
            t_ptr += 1
            
        if t_ptr < len(t_offsets) and t_offsets[t_ptr][1] == s_end_abs:
            alignment[s_idx] = t_ptr
            
    return alignment, s_enc['input_ids'], t_enc['input_ids'], t_target_start_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="science")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_id = "google/gemma-2-2b-it"
    student_id = "google/flan-t5-small"
    output_file = f"teacher_logits_{args.dataset}.pt"
    top_k = 50

    print(f"Loading teacher: {teacher_id}")
    tokenizer_t = AutoTokenizer.from_pretrained(teacher_id)
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    model_t = AutoModelForCausalLM.from_pretrained(
        teacher_id, 
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="sdpa"
    )
    
    print(f"Loading student tokenizer: {student_id}")
    tokenizer_s = AutoTokenizer.from_pretrained(student_id)
    
    # Load vocab map to verify top-k matches
    with open("vocab_map.json", "r") as f:
        vocab_map = json.load(f)
    t_to_s = {int(k): int(v) for k, v in vocab_map["teacher_to_student"].items()}

    print(f"Loading data: {args.dataset}")
    train_data = load_sdft_dataset(args.dataset, 'train')
    if args.limit:
        train_data = train_data[:args.limit]
    
    results = []
    
    # Resume check
    if os.path.exists(output_file):
        results = torch.load(output_file, weights_only=False)
        processed_ids = set(item['id'] for item in results)
        train_data = [item for item in train_data if item['id'] not in processed_ids]
        print(f"Resuming from {len(processed_ids)} items.")

    print(f"Processing {len(train_data)} items...")
    
    from utils import format_target
    for item in tqdm(train_data):
        torch.cuda.empty_cache()
        prompt = format_prompt(item)
        if args.dataset == "tooluse":
            target = format_target(item['target'])
        else:
            target = item['target']
        
        # Align with teacher
        # Use add_special_tokens=True for student to match training (adds EOS)
        s_enc = tokenizer_s(target, return_offsets_mapping=True, add_special_tokens=True)
        s_ids = s_enc['input_ids']
        s_offsets = s_enc['offset_mapping']
        
        # Teacher alignment remains similar but must match absolute offsets
        full_text = prompt + target
        t_enc = tokenizer_t(full_text, return_offsets_mapping=True, add_special_tokens=False)
        t_ids = t_enc['input_ids']
        t_offsets = t_enc['offset_mapping']
        
        # Find where target starts in teacher offsets
        prompt_len = len(prompt)
        t_target_start_idx = 0
        for idx, (start, end) in enumerate(t_offsets):
            if start >= prompt_len:
                t_target_start_idx = idx
                break
                
        # align_map: student_decoder_idx -> teacher_idx
        align_map = {}
        t_ptr = t_target_start_idx
        for s_idx, (s_start, s_end) in enumerate(s_offsets):
            s_end_abs = s_end + prompt_len
            while t_ptr < len(t_offsets) and t_offsets[t_ptr][1] < s_end_abs:
                t_ptr += 1
            if t_ptr < len(t_offsets) and t_offsets[t_ptr][1] == s_end_abs:
                align_map[s_idx] = t_ptr
        
        # Prepare teacher input
        inputs_t = tokenizer_t(
            full_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1536
        ).to(device)
        
        with torch.no_grad():
            outputs = model_t(**inputs_t)
            logits = outputs.logits[0] # [seq_len, vocab_size]
            
        item_top_probs = []
        item_top_indices = []
        item_s_indices = [] # Which student decoder token this refers to
        
        for s_idx in range(len(s_ids)):
            # To predict student decoder token s_idx, we need teacher logit at position align_map[s_idx-1] + 1
            # or if s_idx is 0, we need teacher logit at the position just before the target starts + 1.
            
            if s_idx == 0:
                is_aligned = (align_map.get(0) == t_target_start_idx)
                t_pos = t_target_start_idx - 1
            else:
                prev_t_pos = align_map.get(s_idx - 1)
                curr_t_pos = align_map.get(s_idx)
                is_aligned = (prev_t_pos is not None and curr_t_pos is not None and curr_t_pos == prev_t_pos + 1)
                t_pos = prev_t_pos if prev_t_pos is not None else None
            
            if is_aligned and t_pos is not None and t_pos < len(logits):
                # Get top-k from teacher distribution
                p = torch.softmax(logits[t_pos].float(), dim=-1)
                top_p, top_i = torch.topk(p, top_k)
                
                item_top_probs.append(top_p.cpu().half())
                item_top_indices.append(top_i.cpu().int())
                item_s_indices.append(s_idx)
        
        if item_top_probs:
            results.append({
                'id': item['id'],
                's_indices': torch.tensor(item_s_indices, dtype=torch.int32),
                'top_probs': torch.stack(item_top_probs),
                'top_indices': torch.stack(item_top_indices)
            })

        # Save periodically
        if len(results) % 50 == 0:
            torch.save(results, output_file)

    if results:
        torch.save(results, output_file)
        
    print(f"Done. Logits saved to {output_file}")

if __name__ == "__main__":
    main()
