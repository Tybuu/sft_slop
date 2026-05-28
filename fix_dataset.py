import argparse
import os
import json
import re
from datasets import load_from_disk

def merge_golden_responses(example):
    gr = example["golden_response"]
    if len(gr) <= 1:
        return example

    merged = "\n\n".join(gr)
    example["golden_response"] = [merged]
    return example

def check_example(example, idx):
    issues = []
    gr = example["golden_response"]
    ga = example["golden_answer"]

    ga_action_count = len(ga)

    thought_actions = set()
    resp_actions = set()
    for resp in gr:
        found = re.findall(r"Action:\s*(\w+)", resp)
        resp_actions.update(found)
    for ga_item in ga:
        thought_actions.add(ga_item["Action"])

    if resp_actions != thought_actions:
        issues.append(
            f"Action mismatch: golden_answer={sorted(thought_actions)} vs golden_response={sorted(resp_actions)}"
        )

    return issues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="sdft_repo/data/tooluse_data/train_data")
    parser.add_argument("--output", type=str, default="data/tooluse_data/train_data_fixed")
    parser.add_argument("--verify", action="store_true", help="Only verify the dataset, don't fix")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    dataset = load_from_disk(args.input)
    print(f"Loaded {len(dataset)} examples with columns: {dataset.column_names}")

    before_multi = sum(1 for ex in dataset if len(ex["golden_response"]) > 1)

    if args.verify:
        print(f"\nMulti-step examples before fix: {before_multi}")
        print(f"Single-step examples: {len(dataset) - before_multi}")
        print(f"\nSample multi-step (raw):")
        count = 0
        for i, ex in enumerate(dataset):
            if len(ex["golden_response"]) > 1:
                print(f"\n--- Example {i} ({len(ex['golden_response'])} steps) ---")
                for j, resp in enumerate(ex["golden_response"]):
                    print(f"  Step {j}: {resp[:200]}...")
                count += 1
                if count >= 3:
                    break
        return

    print(f"\nFixing dataset: {before_multi} multi-step examples to merge...")
    fixed_dataset = dataset.map(merge_golden_responses)

    after_multi = sum(1 for ex in fixed_dataset if len(ex["golden_response"]) > 1)
    print(f"After fix - multi-step: {after_multi} (should be 0)")

    print(f"\nVerifying a sample of fixed examples:")
    issues_found = 0
    for i in range(min(20, len(fixed_dataset))):
        issues = check_example(fixed_dataset[i], i)
        if issues:
            issues_found += 1
            print(f"  Example {i}: {issues}")
        else:
            gr = fixed_dataset[i]["golden_response"]
            ga_len = len(fixed_dataset[i]["golden_answer"])
            if len(gr) > 0:
                step_count = gr[0].count("Action:")
                print(f"  Example {i}: {step_count} actions in response, {ga_len} in golden_answer ✓")

    print(f"\nIssues found: {issues_found}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nSaving fixed dataset to {args.output}...")
    fixed_dataset.save_to_disk(args.output)
    print(f"Done. Saved {len(fixed_dataset)} examples.")

    merged_before = before_multi
    merged_after = sum(1 for ex in fixed_dataset if "\n\n" in ex["golden_response"][0] if len(ex["golden_response"]) > 0)
    print(f"Multi-step examples merged: {merged_before}")
    total_actions_before = sum(
        len(item["golden_answer"]) for item in dataset
    )
    total_actions_after = sum(
        len(item["golden_answer"]) for item in fixed_dataset
    )
    print(f"Total actions (golden_answer): {total_actions_before}")
    print(f"Total response entries: {before_multi} (before) → {after_multi} (after)")

if __name__ == "__main__":
    main()
