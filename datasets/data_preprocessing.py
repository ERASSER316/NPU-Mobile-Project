"""
数据预处理脚本：将messages格式转换为QAT训练格式

使用方法:
    python data_preprocessing.py \
        --input_dir ../datasets/mergedata \
        --output_dir ../datasets/mergedata_preprocessed \
        --model_path ../../models/Llama-3.2-1B-Instruct \
        --max_length 1024\
        --batch_size 256\
        --num_proc 4
"""

import os
import argparse
import json
from typing import Dict, List
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path


def preprocess_function(examples: Dict, tokenizer, max_length: int = 4096) -> Dict:
    """
    预处理函数：将messages转换为tokenized格式
    
    Args:
        examples: Batch of examples with 'messages' field
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with input_ids, attention_mask, labels
    """
    # 检查数据格式
    if "messages" not in examples:
        raise ValueError("Data must contain 'messages' field!")
    
    # 处理messages格式
    texts = []
    
    for messages in examples["messages"]:
        # 使用tokenizer的apply_chat_template（如果支持）
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                # Llama-3.2使用标准的chat template
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                print(f"Warning: apply_chat_template failed, using manual format: {e}")
                raise ValueError(f"apply_chat_template failed for sample: {e}")
        else:
            raise RuntimeError(
                "Tokenizer has no valid chat_template. "
                "Please use an Instruct tokenizer (e.g. Llama-3.2-1B-Instruct)."
            )
        
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        add_special_tokens=False,  # 关键：避免重复加特符
        return_tensors=None,  # 返回list而不是tensor
    )
    
    # 创建labels：对于causal LM，labels就是input_ids的副本
    # 对于QAT训练，通常对所有token计算loss（包括system和user部分）
    # 如果需要只对assistant部分计算loss，可以将其他部分设为-100
    labels = []
    for input_ids in tokenized["input_ids"]:
        # 对于QAT，我们通常对所有token计算loss以保持训练稳定性
        # 如果需要mask某些部分，可以在这里处理
        label = input_ids.copy()
        
        # 可选：mask掉padding tokens（pad_token_id对应的位置）
        if tokenizer.pad_token_id is not None:
            label = [-100 if token_id == tokenizer.pad_token_id else token_id 
                    for token_id in label]
        
        labels.append(label)
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess messages format data for QAT training")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing messages format dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed dataset")
    parser.add_argument("--model_path", type=str,
                       default="/models/meta-llama/Llama-3.2-1B-Instruct",
                       help="Path to model for tokenizer")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for processing")
    parser.add_argument("--num_proc", type=int, default=4,
                       help="Number of processes for parallel processing")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Preprocessing for QAT Training")
    print("=" * 60)
    
    # 1. 加载tokenizer
    print(f"\n[Step 1] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  ✓ Set pad_token to eos_token")
    print(f"  ✓ Tokenizer loaded (vocab_size={len(tokenizer)})")
    
    # 2. 加载原始数据
    print(f"\n[Step 2] Loading dataset from {args.input_dir}...")
    
    input_path = Path(args.input_dir)
    
    # 检查是 JSONL 文件还是 datasets 目录
    if input_path.is_file() and input_path.suffix == '.jsonl':
        # 单个 JSONL 文件（需要指定 train 和 test）
        raise ValueError("Please provide a directory containing train.jsonl and test.jsonl, or use --input_dir with a datasets directory")
    elif input_path.is_dir():
        # 检查是否是 HuggingFace datasets 格式
        if (input_path / "dataset_dict.json").exists() or (input_path / "train").exists():
            # HuggingFace datasets 格式
            print("  Detected HuggingFace datasets format")
            dataset = load_from_disk(str(input_path))
        else:
            # 检查是否有 train.jsonl 和 test.jsonl
            train_file = input_path / "train.jsonl"
            test_file = input_path / "test.jsonl"
            
            if train_file.exists() and test_file.exists():
                # JSONL 文件格式
                print("  Detected JSONL format (train.jsonl and test.jsonl)")
                print(f"  Loading train.jsonl...")
                train_dataset = load_dataset("json", data_files=str(train_file), split="train")
                print(f"  Loading test.jsonl...")
                test_dataset = load_dataset("json", data_files=str(test_file), split="train")
                
                dataset = DatasetDict({
                    "train": train_dataset,
                    "test": test_dataset
                })
            else:
                raise FileNotFoundError(
                    f"Input directory must contain either:\n"
                    f"  - HuggingFace datasets format (with dataset_dict.json or train/ subdirectory), OR\n"
                    f"  - train.jsonl and test.jsonl files\n"
                    f"Found in {args.input_dir}: {list(input_path.iterdir())}"
                )
    else:
        raise FileNotFoundError(f"Input path not found: {args.input_dir}")
    
    print(f"  ✓ Dataset loaded")
    print(f"    - Train samples: {len(dataset['train'])}")
    print(f"    - Test samples: {len(dataset['test'])}")
    
    # 检查数据格式
    print(f"\n[Step 3] Checking data format...")
    sample = dataset["train"][0]
    print(f"  Sample keys: {sample.keys()}")
    if "messages" in sample:
        print(f"  ✓ Found 'messages' field")
        print(f"  Sample messages: {sample['messages'][:2]}...")  # 只显示前2条
    else:
        raise ValueError("Data must contain 'messages' field!")
    
    # 3. 预处理数据
    print(f"\n[Step 4] Preprocessing data (max_length={args.max_length})...")
    print("  This may take a while...")
    
    def process_batch(examples):
        return preprocess_function(examples, tokenizer, args.max_length)
    
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=args.batch_size,#先设置为256，如果报错再减小
        num_proc=args.num_proc,
        remove_columns=dataset["train"].column_names,  # 移除原始列
        desc="Tokenizing",
    )
    
    print("  ✓ Preprocessing completed")
    
    # 4. 验证处理结果
    print(f"\n[Step 5] Validating processed data...")
    train_sample = processed_dataset["train"][0]
    print(f"  Processed keys: {train_sample.keys()}")
    print(f"  Input IDs shape: {len(train_sample['input_ids'])}")
    print(f"  Attention mask shape: {len(train_sample['attention_mask'])}")
    print(f"  Labels shape: {len(train_sample['labels'])}")
    
    # 检查是否有-100（masked tokens）
    labels_sample = train_sample["labels"]
    masked_count = sum(1 for x in labels_sample if x == -100)
    print(f"  Masked tokens in sample: {masked_count}/{len(labels_sample)}")
    
    # 5. 保存处理后的数据
    print(f"\n[Step 6] Saving processed dataset to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    processed_dataset.save_to_disk(args.output_dir)
    print("  ✓ Dataset saved successfully")
    
    # 6. 统计信息
    print(f"\n[Step 7] Dataset statistics:")
    train_len = processed_dataset["train"]
    test_len = processed_dataset["test"]
    
    # 计算实际有效长度（排除padding）
    # 使用 attention_mask 统计有效token数量（1表示有效，0表示padding）
    def avg_length(examples):
        return {
            "avg_len": [
                sum(mask) for mask in examples["attention_mask"]
            ]
        }
    
    train_stats = train_len.map(avg_length, batched=True, remove_columns=train_len.column_names)
    test_stats = test_len.map(avg_length, batched=True, remove_columns=test_len.column_names)
    
    train_avg_len = sum(train_stats["avg_len"]) / len(train_stats["avg_len"])
    test_avg_len = sum(test_stats["avg_len"]) / len(test_stats["avg_len"])
    
    print(f"  Train:")
    print(f"    - Samples: {len(train_len)}")
    print(f"    - Avg effective length: {train_avg_len:.1f} tokens (excluding padding)")
    print(f"    - Max length: {args.max_length} tokens (with padding)")
    print(f"  Test:")
    print(f"    - Samples: {len(test_len)}")
    print(f"    - Avg effective length: {test_avg_len:.1f} tokens (excluding padding)")
    print(f"    - Max length: {args.max_length} tokens (with padding)")
    
    print("\n" + "=" * 60)
    print("✓ Data preprocessing completed successfully!")
    print("=" * 60)
    print(f"\nNext step: Use --data_dir={args.output_dir} in training script")


if __name__ == "__main__":
    main()

