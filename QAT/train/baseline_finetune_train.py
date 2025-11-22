#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline FP/BF16 Finetune Script for Llama-3.2-1B (Train Only)

作用：
- 在原始 Llama-3.2-1B 上做纯 FP/BF16 的短轮次训练（例如 100 steps）
- 不做任何在线评测（不算 PPL、不测 throughput）
- 训练结束后，只负责：
    * 保存 finetune 后的 BF16 模型
    * 保存 tokenizer
    * 记录一些训练元信息（steps、模型大小等），供 offline eval 脚本使用

CUDA_VISIBLE_DEVICES=2 python baseline_finetune_train.py \
    --model_path ../../../models/Llama-3.2-1B-Instruct \
    --data_dir ../../datasets/mergedata_preprocessed \
    --save_dir ../../model_quantization/llama3.2-1b-fp-finetune-100steps \
    --train_batch_size 1 \
    --max_steps 100 \
    --lr 5e-5 \
    --device cuda

后续：
- 使用统一的 offline eval 脚本，对该 finetuned 模型做 PPL / throughput 评测
- 再在此 finetuned BF16 模型上做 W8A8 PTQ，导出 INT8 模型
"""

import os
import time
import json
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================
# Utils
# ======================

def parse_args():
    parser = argparse.ArgumentParser("Baseline FP/BF16 Finetune (Train Only)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base Llama-3.2-1B model path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Tokenized dataset path (load_from_disk)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Where to save finetuned baseline model")

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, subsample train split for quick debug")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--meta_file", type=str,
                        default="baseline_fp_finetune_meta.json",
                        help="Meta info JSON saved into save_dir")

    return parser.parse_args()


def collate_fn(examples):
    """
    与 QAT 脚本保持一致：
    - 假设 dataset 中每个样本含有 "input_ids", "attention_mask"
    - labels = input_ids.clone()
      （注意：offline eval 脚本有自己的 labels 处理逻辑，这里只用于训练）
    """
    input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
    attention_mask = [torch.tensor(e["attention_mask"], dtype=torch.long) for e in examples]

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def get_model_size_mb_via_save(model, save_dir):
    """
    与 QAT 脚本相同的“通过临时保存 state_dict 测模型体积”的方法。
    不属于精度评测，只是记录一个模型大小的 meta 信息。
    """
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = os.path.join(save_dir, "tmp_baseline_fp_model.pt")
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


# ======================
# Main
# ======================

def main():
    args = parse_args()

    print("=" * 60)
    print("Baseline FP/BF16 Finetune - Train Only")
    print("=" * 60)
    print(f"Base model path   : {args.model_path}")
    print(f"Data dir          : {args.data_dir}")
    print(f"Save dir          : {args.save_dir}")
    print(f"Train batch size  : {args.train_batch_size}")
    print(f"Max train steps   : {args.max_steps}")
    print(f"LR                : {args.lr}")
    print("=" * 60)

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. Data（只用 train split 做 finetune）
    print("\n[Step 1] Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    if isinstance(dataset, dict):
        train_data = dataset.get("train", None)
    else:
        train_data = dataset

    if train_data is None:
        raise ValueError("Train split is required.")

    print(f"  ✓ Train samples: {len(train_data)}")

    if args.max_samples is not None:
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        print(f"  ✓ Subsampled to {len(train_data)} train samples")

    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 2. Tokenizer
    print("\n[Step 2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("  ✓ Set pad_token_id = eos_token_id")

    # 3. Load base model on CPU (BF16)
    print("\n[Step 3] Loading base model on CPU (FP/BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 4. Baseline FP/BF16 training（无 fake quant）
    print("\n[Step 4] Baseline FP/BF16 finetune (no fake quant, no QAT)...")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    ema_loss = None

    for epoch in range(10**9):
        for batch in train_loader:
            global_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
            ):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()

            if global_step % 10 == 0:
                print(f"[step {global_step}] loss={loss.item():.4f}, ema_loss={ema_loss:.4f}")

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    print(f"\n  ✓ Baseline finetune finished. total_steps={global_step}")

    # 5. 保存 finetuned baseline 模型（FP/BF16）
    print("\n[Step 5] Saving finetuned baseline model (FP/BF16) to disk...")
    model = model.to("cpu").eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.save_dir)

    model_size_mb = get_model_size_mb_via_save(model, args.save_dir)
    print(f"  ✓ Finetuned baseline model size (FP/BF16) : {model_size_mb:.1f} MB")

    # 6. 保存 meta 信息（不含任何 PPL / throughput 指标）
    print("\n[Step 6] Saving baseline finetune meta info...")
    meta = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "save_dir": args.save_dir,
        "train_batch_size": args.train_batch_size,
        "max_steps": args.max_steps,
        "actual_train_steps": global_step,
        "lr": args.lr,
        "device": str(device),
        "model_size_mb_via_save": model_size_mb,
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "This is a pure FP/BF16 finetuned baseline (no fake quant, no QAT). "
            "Use this model as a control group to compare with W8A8 QAT-finetune, "
            "and as an initialization for subsequent PTQ W8A8. "
            "All PPL / throughput evaluation should be done via the unified offline eval script."
        ),
    }

    meta_path = Path(args.save_dir) / args.meta_file
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Baseline finetune meta saved to {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Baseline FP/BF16 Finetune Summary (Train Only)")
    print("=" * 60)
    print(f"Base model path       : {args.model_path}")
    print(f"Finetuned model path  : {args.save_dir}")
    print(f"Train steps           : {global_step}")
    print(f"Finetuned size (MB)   : {model_size_mb:.1f}")
    print("=" * 60)
    print("Done. Use offline eval script for PPL & throughput.")
    print("=" * 60)


if __name__ == "__main__":
    main()
