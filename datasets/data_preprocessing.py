import os
import argparse
import copy
from typing import Dict
from datasets import load_from_disk, load_dataset, DatasetDict
from transformers import AutoTokenizer
from pathlib import Path

# ==============================================================================
# 核心预处理逻辑
# ==============================================================================
def preprocess_function(examples: Dict, tokenizer, max_length: int = 4096) -> Dict:
    """
    预处理函数：
    1. 使用 chat_template 转换文本
    2. 实现 Train-on-Assistant-Only (只对助手回答计算 Loss)
    3. 修复 Padding 可能会覆盖 EOS 的 Bug
    4. [Fix] 修复 apply_chat_template 空列表报错
    """
    if "messages" not in examples:
        raise ValueError("Data must contain 'messages' field!")

    input_ids_list = []
    labels_list = []
    attention_masks_list = []

    # 获取特殊的 token id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    for messages in examples["messages"]:
        # --- 步骤 1: 生成完整的 Input IDs ---
        try:
            full_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            tokenized_full = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
                return_tensors=None,
            )
        except Exception as e:
            print(f"Template Error, skipping: {e}")
            continue

        input_ids = tokenized_full["input_ids"]
        
        # --- 步骤 2: 构建 Labels (实现 Assistant Only Masking) ---
        # 默认全部 mask 掉 (-100)
        labels = [-100] * len(input_ids)
        
        current_context = []
        
        for msg in messages:
            # --- [Fix] 关键修改开始 ---
            # 如果当前上下文为空，不需要调用模板，直接设为空串
            if len(current_context) == 0:
                prev_context_text = ""
                # 注意：Llama-3 的 template 会在第一个 message 前加 <|begin_of_text|>
                # 但这里设为空串没问题，因为后续计算的是 diff (增量)
            else:
                prev_context_text = tokenizer.apply_chat_template(
                    current_context, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            # --- [Fix] 关键修改结束 ---

            # 获取包含当前消息的文本
            # 这里的深拷贝很重要，防止 modify current_context 影响逻辑（虽然这里 append 是安全的）
            current_context.append(msg)
            current_text = tokenizer.apply_chat_template(
                current_context, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 只处理 Assistant 的回复
            if msg["role"] == "assistant":
                # Tokenize 两次以获取边界
                # add_special_tokens=False 很重要，防止重复添加 BOS
                tokenized_prev = tokenizer(prev_context_text, add_special_tokens=False)["input_ids"]
                tokenized_curr = tokenizer(current_text, add_special_tokens=False)["input_ids"]
                
                start_idx = len(tokenized_prev)
                end_idx = len(tokenized_curr)
                
                # 填充 Labels
                if start_idx < len(input_ids):
                    real_end_idx = min(end_idx, len(input_ids))
                    labels[start_idx:real_end_idx] = input_ids[start_idx:real_end_idx]
        
        # --- 步骤 3: 确保 EOS Token ---
        if input_ids[-1] != eos_token_id and len(input_ids) < max_length:
            input_ids.append(eos_token_id)
            labels.append(eos_token_id)
        
        # --- 步骤 4: Padding 处理 ---
        seq_len = len(input_ids)
        pad_len = max_length - seq_len
        
        if pad_len > 0:
            input_ids = input_ids + [pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = [1] * seq_len + [0] * pad_len
        else:
            attention_mask = [1] * seq_len
            
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_masks_list,
        "labels": labels_list,
    }
# ==============================================================================
# 主程序
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Preprocess messages format data for QAT training")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="../../models/Llama-3.2-1B-Instruct")
    parser.add_argument("--max_length", type=int, default=2048) # 建议调小一点测试，QAT 1024/2048 较常见
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_proc", type=int, default=8)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Preprocessing for QAT (Assistant-Only Loss)")
    print("=" * 60)
    
    # 1. 加载 Tokenizer
    print(f"\n[Step 1] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    
    # Llama 3 必须设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  ✓ Set pad_token to eos_token")
    
    # 2. 加载数据
    print(f"\n[Step 2] Loading dataset from {args.input_dir}...")
    input_path = Path(args.input_dir)
    if (input_path / "dataset_dict.json").exists() or (input_path / "train").exists():
        dataset = load_from_disk(str(input_path))
    elif (input_path / "train.jsonl").exists():
        train_dataset = load_dataset("json", data_files=str(input_path / "train.jsonl"), split="train")
        test_dataset = load_dataset("json", data_files=str(input_path / "test.jsonl"), split="train")
        dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    else:
        raise FileNotFoundError("Could not find valid dataset (HF format or jsonl)")
        
    print(f"  ✓ Train samples: {len(dataset['train'])}")
    
    # 3. 处理数据
    print(f"\n[Step 3] Processing with {args.num_proc} processes...")
    
    def process_batch(examples):
        return preprocess_function(examples, tokenizer, args.max_length)
    
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing & Masking",
        load_from_cache_file=False 
    )
    
    # 4. 验证 Masking 效果 (非常重要!)
    print(f"\n[Step 4] Validating Masking Logic...")
    sample = processed_dataset["train"][0]
    labels = sample["labels"]
    input_ids = sample["input_ids"]
    
    # 解码看看哪些部分被保留了
    # 过滤掉 -100
    valid_labels = [l for l in labels if l != -100]
    decoded_target = tokenizer.decode(valid_labels)
    
    print(f"  --- Original Input Length: {len(input_ids)}")
    print(f"  --- Tokens calculating loss: {len(valid_labels)}")
    print(f"  --- Preview of Target (Loss Part):\n'{decoded_target[:200]}...'")
    
    if len(valid_labels) == 0:
        print("\n⚠️ WARNING: Sample has 0 labels! Check if 'assistant' role exists in data.")
    elif len(valid_labels) == len(input_ids):
        print("\n⚠️ WARNING: Full sequence is used for loss! Masking might have failed.")
    else:
        print("\n  ✓ Masking looks correct (Target < Input)")

    # 5. 保存
    print(f"\n[Step 5] Saving to {args.output_dir}...")
    processed_dataset.save_to_disk(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()