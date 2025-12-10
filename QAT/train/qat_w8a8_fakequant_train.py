#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QAT W8A8 Fake-Quant Script for Llama-3.2-1B + TorchAO on 3090

功能：
- 使用 IntxFakeQuantizeConfig 配置 W8A8 的假量化（QAT）
- 在训练过程中插入 fake-quant 节点进行带量化噪声的训练
- 训练结束后通过 QATConfig(step="convert") 去掉 fake-quant wrapper，
  得到一个“QAT 适配后”的实数权重 BF16 模型
- 不做任何在线评测（PPL / throughput），所有评测统一由 offline 脚本完成

用法示例：


CUDA_VISIBLE_DEVICES=2 python qat_w8a8_fakequant_train.py \
    --model_path ../../../models/Llama-3.2-1B-Instruct \
    --data_dir ../../datasets/mergedata_prev2  \
    --save_dir ../../modelnew/llama3.2-1b-w8a8-qat-experiment \
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
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_scheduler

from torchao.quantization import quantize_
from torchao.quantization.qat import (
    QATConfig,
    IntxFakeQuantizeConfig,
)


# ======================
# Utils
# ======================

def parse_args():
    parser = argparse.ArgumentParser("QAT W8A8 Fake-Quant Train Only")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base BF16 model path (e.g., Llama-3.2-1B-Instruct)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed dataset (load_from_disk)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Where to save the converted (de-fake) QAT model")

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of micro-batches to accumulate before each optimizer step")

    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional: limit training samples for quick experiments")

    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--meta_file", type=str, default="qat_w8a8_train_meta.json",
                        help="Optional meta info file saved in save_dir")
    # Scheduler 配置，与 Baseline 对齐（支持 warmup）
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["cosine", "linear"],
                        help="Scheduler type, same naming as transformers.get_scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio of total training steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def collate_fn(examples):
    """
    QAT 训练用简单 collate：
    - 假设 input_ids / attention_mask 已经是定长、padding 好的
    - 使用数据自带的labels（保留预处理中的-100标记，用于mask padding tokens）
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
    对 QAT-convert 后的模型测大小（即不带 fake quant wrapper 的实数权重模型）。
    """
    os.makedirs(save_dir, exist_ok=True)
    tmp_path = os.path.join(save_dir, "tmp_qat_w8a8_fp_model.pt")
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
    print("QAT W8A8 Fake-Quant - Train Only")
    print("=" * 60)
    print(f"Base model path   : {args.model_path}")
    print(f"Data dir          : {args.data_dir}")
    print(f"Save dir          : {args.save_dir}")
    print(f"Train batch size  : {args.train_batch_size}")
    print(f"Max QAT steps     : {args.max_steps}")
    print(f"Init_LR                : {args.lr}")
    print(f"Grad Accum Steps  : {args.gradient_accumulation_steps}")
    print(f"LR Scheduler      : {args.lr_scheduler_type} (warmup_ratio={args.warmup_ratio})")
    print(f"Seed                   : {args.seed}")
    print("=" * 60)

    # Fix random seed for reproducibility
    set_seed(args.seed)

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. Data (只用 train split 做 QAT 训练)
    print("\n[Step 1] Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    if isinstance(dataset, dict):
        train_data = dataset.get("train", None)
    else:
        train_data = dataset
    if train_data is None:
        raise ValueError("Train split is required for QAT.")

    print(f"  ✓ Train samples: {len(train_data)}")

    if args.max_samples is not None:
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        print(f"  ✓ Subsampled to {len(train_data)} train samples")

    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    # 2. Tokenizer（只是顺便一起保存，方便之后 from_pretrained）
    print("\n[Step 2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("  ✓ Set pad_token_id = eos_token_id")

    # 3. Load base model on CPU
    print("\n[Step 3] Loading base model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # 4. QAT prepare: 手动配置 W8A8 fake quant（对齐 Int8DynamicActivationInt8WeightConfig 思路）
    print("\n[Step 4] Preparing QAT (W8A8 fake quant) on CPU...")

    # 激活：int8，per-token，非对称（动态）
    act_qcfg = IntxFakeQuantizeConfig(
        torch.int8,
        "per_token",
        is_symmetric=False,  # 动态量化通常使用非对称
        is_dynamic=True,
    )

    # 权重：int8，per-group，对称，group_size=32
    w_qcfg = IntxFakeQuantizeConfig(
        torch.int8,
        granularity="per_channel",  # <--- 【新增】必须显式指定为 per_channel
        group_size=None,
        is_symmetric=True,
        is_dynamic=False,  # 权重一般静态
    )

    qat_prepare_cfg = QATConfig(
        activation_config=act_qcfg,
        weight_config=w_qcfg,
        step="prepare",
    )

    quantize_(model, qat_prepare_cfg)
    print("  ✓ Fake-quant modules inserted.")

    # # 5. QAT training on GPU (fake-quant)
    # print("\n[Step 5] QAT training (W8A8 fake quant)...")
    # model.to(device)
    # model.train()

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # global_step = 0
    # ema_loss = None

    # for epoch in range(10**9):
    #     for batch in train_loader:
    #         global_step += 1
    #         batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    #         optimizer.zero_grad(set_to_none=True)

    #         with torch.amp.autocast(
    #             device_type=device.type,
    #             dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    #         ):
    #             outputs = model(**batch)
    #             loss = outputs.loss

    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()

    #         ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()

    #         if global_step % 10 == 0:
    #             print(f"[step {global_step}] loss={loss.item():.4f}, ema_loss={ema_loss:.4f}")

    #         if global_step >= args.max_steps:
    #             break
    #     if global_step >= args.max_steps:
    #         break

    # print(f"\n  ✓ QAT training finished. total_steps={global_step}")

    # 5. QAT training on GPU (fake-quant)
    print("\n[Step 5] QAT training (W8A8 fake quant)...")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_update_steps = args.max_steps
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    
    # === 梯度累积配置（从 args 读取） ===
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    global_step = 0      # 记录参数“更新”次数
    batch_iterator = 0   # 记录读入的 micro-batch 数
    ema_loss = None

    print(f"Starting training with gradient_accumulation_steps={gradient_accumulation_steps}")
    print(f"Target total update steps: {args.max_steps}")

    for epoch in range(10**9):
        for batch in train_loader:
            batch_iterator += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
            ):
                outputs = model(**batch)
                # Loss 均分到每个累积步
                loss = outputs.loss / gradient_accumulation_steps

            # 反向传播，梯度累积
            loss.backward()

            # 累积满 N 步才更新一次参数 & scheduler
            if batch_iterator % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Scheduler 每次“参数更新”后 step 一次
                if scheduler is not None:
                    scheduler.step()
                
                global_step += 1
                
                # 恢复原始 loss 用于日志
                current_loss_val = loss.item() * gradient_accumulation_steps
                ema_loss = current_loss_val if ema_loss is None else 0.9 * ema_loss + 0.1 * current_loss_val

                # 打印当前学习率
                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]["lr"]

                if global_step % 10 == 0:
                    print(
                        f"[step {global_step}] loss={current_loss_val:.4f}, "
                        f"ema_loss={ema_loss:.4f}, lr={current_lr:.6e}"
                    )

                if global_step >= args.max_steps:
                    break
        
        if global_step >= args.max_steps:
            break

    print(f"\n  ✓ QAT training finished. total_update_steps={global_step}, total_batches_seen={batch_iterator}")

    # 6. 转回 CPU 做 convert（去掉 fake quant wrapper，得到 QAT 后实数权重模型）
    print("\n[Step 6] Converting QAT model (remove fake quant wrappers)...")
    model = model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 先convert，再eval
    qat_convert_cfg = QATConfig(step="convert")
    quantize_(model, qat_convert_cfg)
    model.eval()  # convert后再eval
    print("  ✓ Fake-quant wrappers removed (de-fake QAT model ready).")

    # 7. 保存这个“QAT 后权重模型”（BF16），作为后续 offline PTQ 的起点
    print(f"\n[Step 7] Saving converted (de-fake) QAT model to {args.save_dir} ...")
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.save_dir)

    model_size_mb = get_model_size_mb_via_save(model, args.save_dir)
    print(f"  ✓ Converted QAT model size (FP/BF16) : {model_size_mb:.1f} MB")

    # 8. 保存少量 meta 信息（不包含任何 test/throughput 结果）
    print("\n[Step 8] Saving QAT train meta info...")
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
            "This model is BF16 after QAT (fake quant converted). "
            "Use the unified offline eval script to measure PPL & throughput. "
            "Then optionally apply W8A8 PTQ (e.g., Int8DynamicActivationInt8WeightConfig) on top."
        ),
    }

    meta_path = Path(args.save_dir) / args.meta_file
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  ✓ QAT train meta saved to {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("QAT W8A8 Fake-Quant Training Summary")
    print("=" * 60)
    print(f"Base model path       : {args.model_path}")
    print(f"QAT de-fake save_dir  : {args.save_dir}")
    print(f"Train steps           : {global_step}")
    print(f"BF16 QAT model sizeMB : {model_size_mb:.1f}")
    print("=" * 60)
    print("Done. Use offline eval script for PPL & throughput.")
    print("=" * 60)


if __name__ == "__main__":
    main()
