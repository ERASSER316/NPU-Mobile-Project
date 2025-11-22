#!/usr/bin/env python3
"""
Merge multiple JSONL files for Llama 3.2 QAT fine-tuning.
"""

import json
import os
from pathlib import Path

def merge_jsonl_files(input_dir, output_file, shuffle=True):
    """
    Merge multiple JSONL files from a directory into one file.
    
    Args:
        input_dir: Directory containing JSONL files
        output_file: Output JSONL file path
        shuffle: Whether to shuffle the data (default: True)
    """
    import random
    
    all_data = []
    input_path = Path(input_dir)
    
    # Get all jsonl files
    jsonl_files = sorted(input_path.glob("*.jsonl"))
    
    print(f"\nProcessing {input_dir}:")
    print(f"Found {len(jsonl_files)} JSONL files:")
    
    # Read all files
    for jsonl_file in jsonl_files:
        print(f"  - Reading {jsonl_file.name}...", end=" ")
        count = 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                        count += 1
                    except json.JSONDecodeError as e:
                        print(f"\n    Warning: Failed to parse line in {jsonl_file.name}: {e}")
        print(f"{count} examples")
    
    print(f"\nTotal examples collected: {len(all_data)}")
    
    # Shuffle if requested
    if shuffle:
        print("Shuffling data...")
        random.seed(42)  # For reproducibility
        random.shuffle(all_data)
    
    # Write merged data
    print(f"Writing to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Successfully wrote {len(all_data)} examples to {output_file}")
    return len(all_data)

def main():
    # Define paths
    base_dir = Path("/home/wcp/torchao/datasets/pre_datasets")
    output_dir = Path("/home/wcp/torchao/datasets")
    
    # Merge test files
    test_input = base_dir / "test"
    test_output = output_dir / "test.jsonl"
    test_count = merge_jsonl_files(test_input, test_output, shuffle=True)
    
    # Merge train files
    train_input = base_dir / "train"
    train_output = output_dir / "train.jsonl"
    train_count = merge_jsonl_files(train_input, train_output, shuffle=True)
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Test set:  {test_count} examples → {test_output}")
    print(f"  Train set: {train_count} examples → {train_output}")
    print("="*60)

if __name__ == "__main__":
    main()
