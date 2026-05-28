import json
import os
from datasets import load_from_disk

def load_sdft_dataset(dataset_name="science", split="train", dataset_path=None):
    """
    Loads SDFT datasets from the cloned repository.
    dataset_name can be "science", "tooluse", "medical", or "wiki".
    If dataset_path is provided, it overrides the default path for the train split.
    """
    if dataset_path and split == "train":
        path = dataset_path
    else:
        base_path = "sdft_repo/data"
        if dataset_name == "science":
            sub_dir = "train_data" if split == "train" else "eval_data"
            path = os.path.join(base_path, "science_data", sub_dir)
        elif dataset_name == "tooluse":
            sub_dir = "train_data" if split == "train" else "eval_data"
            path = os.path.join(base_path, "tooluse_data", sub_dir)
        elif dataset_name == "medical":
            path = os.path.join(base_path, "medical_data", f"{split}_data")
        elif dataset_name == "wiki":
            path = os.path.join(base_path, "wiki_data", f"{split}_data")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist.")
        return []

    dataset = load_from_disk(path)

    data = []
    for i, item in enumerate(dataset):
        if dataset_name == "science":
            if split == "train":
                input_text = item['messages'][1]['content']
                target = item['output_text']
            else:
                input_text = item['prompt'][1]['content']
                target = item['answer']

            data.append({
                'id': f"science_{split}_{i}",
                'input': input_text,
                'target': target,
            })
        elif dataset_name == "tooluse":
            input_text = item['prompt']
            if split == "train":
                target = item['golden_response'][0]
            else:
                target_parts = []
                for action_item in item['golden_answer']:
                    target_parts.append(f"Action: {action_item['Action']}\nAction Input: {action_item['Action_Input']}")
                target = "\n".join(target_parts)

            data.append({
                'id': f"tooluse_{split}_{i}",
                'input': input_text,
                'target': target
            })
    return data

def format_prompt(item):
    return item['input']

if __name__ == "__main__":
    try:
        train_data = load_sdft_dataset('science', 'train')
        print(f"Loaded {len(train_data)} Science training examples.")
        if train_data:
            print("Sample Prompt:")
            print(format_prompt(train_data[0]))
            print("\nSample Target:")
            print(train_data[0]['target'])
    except Exception as e:
        print(f"Error loading science data: {e}")
