#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline FP/BF16 Finetune Script for Llama-3.2-1B (Train Only)

修改说明：
- 增加了随机种子固定 (set_seed)，确保与 QAT 实验的数据顺序一致。
- 增加了 LR Scheduler (Cosine/Linear)，避免固定 LR 导致的收敛差异。
- 增加了梯度累积 (Gradient Accumulation) 支持，以便对齐 batch size。
- 优化了训练循环和日志记录。

使用方法：
    CUDA_VISIBLE_DEVICES=2 python baseline_finetune_train.py \
    --model_path ../../../models/Llama-3.2-1B-Instruct \
    --data_dir ../../datasets/mergedata_prev2  \
    --save_dir ../../modelnew/llama3.2-1b-baseline-control \
    --train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 100 \
    --lr 2e-5 \
    --seed 42 \
    --device cuda
"""

import os
import time
import json
import random
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_scheduler,
    set_seed
)

# ======================
# Setup Logging
# ======================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ======================
# Utils
# ======================

def parse_args():
    parser = argparse.ArgumentParser("Baseline FP/BF16 Finetune (Train Only)")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--data_dir", type=str, required=True, help="Tokenized dataset path")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate before backward()")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for scheduler")
    
    parser.add_argument("--max_samples", type=int, default=None, help="Debug: subsample dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--meta_file", type=str, default="baseline_fp_finetune_meta.json")

    return parser.parse_args()


def collate_fn(examples):
    """
    与 QAT 脚本保持严格一致的数据处理
    使用数据自带的labels（保留预处理中的-100标记，用于mask padding tokens）
    """
    input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
    attention_mask = [torch.tensor(e["attention_mask"], dtype=torch.long) for e in examples]
    labels = [torch.tensor(e["labels"], dtype=torch.long) for e in examples]  # 使用数据自带的labels

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    labels = torch.stack(labels, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,  # 保留-100标记，避免对padding tokens计算loss
    }


def get_model_size_mb_via_save(model, save_dir):
    """
    通过临时保存测体积，确保统计口径一致
    """
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = os.path.join(save_dir, "tmp_baseline_fp_model.pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return size_mb


# ======================
# Main
# ======================

def main():
    args = parse_args()
    
    # 0. Set Seed (Crucial for Baseline vs QAT comparison)
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Baseline FP/BF16 Finetune - Train Only")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Steps: {args.max_steps}, Batch: {args.train_batch_size}, GradAccum: {args.gradient_accumulation_steps}")
    logger.info(f"Device: {args.device}, Seed: {args.seed}")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Data
    logger.info("Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    train_data = dataset.get("train", dataset) if isinstance(dataset, dict) else dataset
    
    if args.max_samples is not None:
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    
    logger.info(f"Train samples: {len(train_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True, # Seed ensures this shuffle is identical to QAT run if seed matches
        collate_fn=collate_fn,
        drop_last=True # Prevent shape mismatch in last batch
    )

    # 2. Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Model
    logger.info("Loading base model (FP/BF16)...")
    # 直接加载到目标设备通常更快，除非显存非常吃紧
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # 显存优化配置
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    # 4. Optimizer & Scheduler
    # 确保 Optimizer 与 QAT 设置一致 (通常 QAT 也是 AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler 是 Baseline 能否打过 QAT 的关键，建议使用 Cosine
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.max_steps * args.warmup_ratio),
        num_training_steps=args.max_steps,
    )

    # 5. Training Loop
    logger.info("Starting training...")
    global_step = 0      # 参数更新次数（真正的训练步数）
    batch_iterator = 0   # batch计数（用于梯度累积）
    total_loss = 0.0
    ema_loss = None
    
    progress_bar = tqdm(range(args.max_steps), desc="Training")

    # 转换为无限迭代器，通过 max_steps 控制退出
    train_iterator = iter(train_loader)

    while global_step < args.max_steps: 
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        batch_iterator += 1
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Forward
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps

        # Backward
        loss.backward()

        total_loss += loss.item() * args.gradient_accumulation_steps

        # Step (Gradient Accumulation Logic)
        # 累积满N个batch才更新一次参数
        if batch_iterator % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1  # 真正的参数更新次数
            
            # Update logs
            current_loss = total_loss
            ema_loss = current_loss if ema_loss is None else 0.9 * ema_loss + 0.1 * current_loss
            
            progress_bar.update(1) # 更新进度条
            progress_bar.set_postfix({"loss": f"{ema_loss:.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
            
            total_loss = 0.0
            
            if global_step >= args.max_steps:
                break

    progress_bar.close()
    logger.info("Training finished.")

    # 6. Save
    logger.info("Saving model & tokenizer...")
    # 先移回 CPU 节省显存，防止 OOM
    model = model.to("cpu").eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir, safe_serialization=False) # 使用 .bin 格式，与你原脚本一致
    tokenizer.save_pretrained(args.save_dir)

    model_size_mb = get_model_size_mb_via_save(model, args.save_dir)
    logger.info(f"Model Size: {model_size_mb:.2f} MB")

    # 7. Meta Info
    meta = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "train_batch_size": args.train_batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "seed": args.seed,
        "model_size_mb": model_size_mb,
        "note": "Baseline (FP16/BF16) with Seed & Scheduler."
    }

    meta_path = Path(args.save_dir) / args.meta_file
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"Meta saved to {meta_path}")


if __name__ == "__main__":
    main()