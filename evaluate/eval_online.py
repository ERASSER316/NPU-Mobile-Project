"""
Eval Script for Llama-3.2-1B (Baseline & TorchAO Int8 Weight-Only)

用法示例：

1）Baseline（原始 BF16 模型）：
CUDA_VISIBLE_DEVICES=1 python eval_online.py \
    --model_path ../../models/Llama-3.2-1B-Instruct \
    --data_dir ../datasets/mergedata_preprocessed \
    --output_file baseline_results.json \
    --batch_size 8 \
    --device cuda \
    --quant_mode none

2）Int8 Weight-Only PTQ（在线量化）：
CUDA_VISIBLE_DEVICES=1 python eval_online.py \
    --model_path ../../models/Llama-3.2-1B-Instruct \
    --data_dir ../datasets/mergedata_preprocessed \
    --output_file PTQ_WeightINT8WOQ_results.json \
    --batch_size 8 \
    --device cuda \
    --quant_mode int8_woq

可选：
    --max_samples 2000       # 只评一部分样本
    --eval_train             # 同时评估 train split
    --use_compile            # 对模型使用 torch.compile（如果环境支持）
"""

import os
import argparse
import math
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TorchAoConfig,
)
from torchao.quantization import Int8WeightOnlyConfig
from torchao.utils import benchmark_model  # 官方测速工具


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model PPL & throughput")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to (base) pretrained model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed dataset directory (load_from_disk)")
    parser.add_argument("--output_file", type=str, default="eval_results.json",
                        help="Output JSON file to save results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per split (None = all)")
    parser.add_argument("--eval_train", action="store_true",
                        help="Also evaluate on training split if present")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: cuda or cpu")
    parser.add_argument("--quant_mode", type=str, default="none",
                        choices=["none", "int8_woq"],
                        help="Quantization mode: none (baseline) | int8_woq (TorchAO Int8 weight-only)")
    parser.add_argument("--use_compile", action="store_true",
                        help="Use torch.compile for model")
    parser.add_argument("--throughput_runs", type=int, default=50,
                        help="Num runs for benchmark_model throughput test")
    return parser.parse_args()


def collate_fn(examples):
    """将样本列表组装成 batch Tensor（假设已 padding 为等长）"""
    batch = {}
    keys = examples[0].keys()
    for key in keys:
        if key in ["input_ids", "attention_mask", "labels"]:
            values = [ex[key] for ex in examples]
            if isinstance(values[0], list):
                batch[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = torch.tensor(values, dtype=torch.long)
    return batch


def evaluate_model(model, data_loader, device, split_name="test", max_samples=None):
    """
    计算 PPL / Avg Loss / Loss 分布
    （保持原有 PPL 计算方法，不在这里算吞吐量）
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    all_losses = []
    samples_evaluated = 0

    print(f"\nEvaluating on {split_name} set...")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            if max_samples is not None and samples_evaluated >= max_samples:
                break

            # 移动到 device
            for k in ["input_ids", "attention_mask", "labels"]:
                if k in batch:
                    batch[k] = batch[k].to(device)

            outputs = model(**batch)
            loss = outputs.loss

            # 只统计 labels != -100 的 token
            n_tokens = (batch["labels"] != -100).sum().item()

            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens
            all_losses.append(loss.item())
            samples_evaluated += batch["input_ids"].size(0)

            if max_samples is not None and samples_evaluated >= max_samples:
                break

    if total_tokens == 0:
        return {
            "ppl": float("inf"),
            "avg_loss": 0.0,
            "loss_std": 0.0,
            "loss_min": 0.0,
            "loss_max": 0.0,
            "total_tokens": 0,
            "samples_evaluated": samples_evaluated,
        }

    avg_loss = total_nll / total_tokens
    ppl = math.exp(avg_loss)

    loss_tensor = torch.tensor(all_losses)
    loss_std = loss_tensor.std().item()
    loss_min = loss_tensor.min().item()
    loss_max = loss_tensor.max().item()

    return {
        "ppl": float(ppl),
        "avg_loss": float(avg_loss),
        "loss_std": float(loss_std),
        "loss_min": float(loss_min),
        "loss_max": float(loss_max),
        "total_tokens": int(total_tokens),
        "samples_evaluated": int(samples_evaluated),
    }


def get_model_size_mb_via_save(model, tmp_path="__tmp_model_size__.pt"):
    """
    使用官方 quick_start 风格：
    通过 torch.save(model) 的实际文件大小度量整体模型大小。
    """
    torch.save(model, tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


class CausalLMWrapper(nn.Module):
    """
    用于 benchmark_model 的包装，接受 (input_ids, attention_mask) 位置参数。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def make_example_inputs_from_dataset(dataset, batch_size, device):
    """
    根据真实数据推断 seq_len，用于构造 benchmark_model 的 example_inputs。
    假设预处理后样本已等长（或接近等长）。
    """
    sample = dataset[0]
    seq_len = len(sample["input_ids"])
    input_ids = torch.randint(
        low=10,
        high=max(sample["input_ids"]) if len(sample["input_ids"]) > 0 else 32000,
        size=(batch_size, seq_len),
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, device=device)
    return (input_ids, attention_mask), seq_len


def benchmark_throughput(model, dataset, batch_size, device, num_runs=50):
    """
    使用 torchao.utils.benchmark_model 风格测试吞吐：
    - 固定 batch_size、根据数据估计 seq_len
    - 多次前向，返回 mean_latency_ms 和 tokens/sec
    """
    if num_runs <= 0:
        return None

    model.eval()
    wrapper = CausalLMWrapper(model).to(device)

    example_inputs, seq_len = make_example_inputs_from_dataset(dataset, batch_size, device)

    # 清理 graph 缓存更干净（可选）
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    # benchmark_model 返回平均耗时（ms）
    mean_time_ms = benchmark_model(wrapper, num_runs, example_inputs)

    tokens_per_run = batch_size * seq_len
    tokens_per_sec = tokens_per_run / (mean_time_ms / 1000.0)

    return {
        "mean_latency_ms": float(mean_time_ms),
        "tokens_per_sec": float(tokens_per_sec),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "num_runs": int(num_runs),
    }


def load_model(args, device):
    """根据 quant_mode 加载 baseline / Int8WOQ 模型"""
    if args.quant_mode == "int8_woq":
        print("\n[Model] Loading TorchAO Int8 Weight-Only model (online PTQ)...")
        quant_config = TorchAoConfig(
            quant_type=Int8WeightOnlyConfig(group_size=32)
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            quantization_config=quant_config,
        )
    else:
        print("\n[Model] Loading baseline BF16 model...")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    model.to(device)

    if args.use_compile and device.type == "cuda":
        try:
            print("[Model] Applying torch.compile(...)")
            model = torch.compile(model)
        except Exception as e:
            print(f"[Model] torch.compile failed, continue without it. Error: {e}")

    # Debug 信息：确认量化 & dtype 状态
    print(f"  - First param dtype: {next(model.parameters()).dtype}")
    print(f"  - Quantization config: {getattr(model.config, 'quantization_config', None)}")
    try:
        sample_layer = model.model.layers[0].self_attn.q_proj
        w = sample_layer.weight
        print("  - Sample layer type:", type(sample_layer))
        print("  - Weight class     :", w.__class__)
        print("  - Weight dtype     :", w.dtype)
    except Exception:
        pass

    return model


def main():
    args = parse_args()

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model path      : {args.model_path}")
    print(f"Quantization    : {args.quant_mode}")
    print(f"Data dir        : {args.data_dir}")
    print(f"Output file     : {args.output_file}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Max samples     : {args.max_samples}")
    print(f"Throughput runs : {args.throughput_runs}")
    print("=" * 60)

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Load dataset
    print("\n[Step 1] Loading dataset...")
    dataset = load_from_disk(args.data_dir)

    if isinstance(dataset, dict):
        test_data = dataset.get("test", None)
        train_data = dataset.get("train", None)
    else:
        test_data = dataset
        train_data = None

    if test_data is None:
        raise ValueError("Test dataset not found in provided data_dir.")

    print(f"  ✓ Test samples : {len(test_data)}")
    if train_data is not None:
        print(f"  ✓ Train samples: {len(train_data)}")

    # Subsample if needed
    if args.max_samples is not None:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))
        if train_data is not None:
            train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        print(f"  ✓ Limited to {args.max_samples} samples per split")

    # Dataloaders
    test_dl = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    train_dl = None
    """
    if args.eval_train and train_data is not None:
        train_dl = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    """


    # Load tokenizer
    print("\n[Step 2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  ✓ Set pad_token to eos_token")

    # Load model
    print("\n[Step 3] Loading model...")
    model = load_model(args, device)

    # Model size via torch.save (官方 quick_start 风格)
    print("\n[Step 4] Measuring model size via torch.save(...)")
    model_size_mb = get_model_size_mb_via_save(model)
    print(f"  ✓ Model size (via save): {model_size_mb:.1f} MB")

    # Evaluate PPL
    print("\n[Step 5] Evaluating on test set (PPL)...")
    test_results = evaluate_model(model, test_dl, device, "test", args.max_samples)

    print(f"\nTest Set Results:")
    print(f"  PPL            : {test_results['ppl']:.4f}")
    print(f"  Avg Loss       : {test_results['avg_loss']:.4f}")
    print(f"  Loss Std       : {test_results['loss_std']:.4f}")
    print(f"  Loss Range     : [{test_results['loss_min']:.4f}, {test_results['loss_max']:.4f}]")
    print(f"  Samples        : {test_results['samples_evaluated']}")
    print(f"  Total tokens   : {test_results['total_tokens']}")

    train_results = None
    if train_dl is not None:
        print("\n[Step 6] Evaluating on train set (PPL)...")
        train_results = evaluate_model(model, train_dl, device, "train", args.max_samples)
        print(f"\nTrain Set Results:")
        print(f"  PPL            : {train_results['ppl']:.4f}")
        print(f"  Avg Loss       : {train_results['avg_loss']:.4f}")
        print(f"  Samples        : {train_results['samples_evaluated']}")
        print(f"  Total tokens   : {train_results['total_tokens']}")

    # Throughput Benchmark（官方 benchmark_model 风格）
    print("\n[Step 7] Benchmarking throughput with torchao.utils.benchmark_model...")
    throughput_results = benchmark_throughput(
        model=model,
        dataset=test_data,
        batch_size=args.batch_size,
        device=device,
        num_runs=args.throughput_runs,
    )
    if throughput_results is not None:
        print(f"  ✓ Mean latency : {throughput_results['mean_latency_ms']:.3f} ms")
        print(f"  ✓ Tokens/sec   : {throughput_results['tokens_per_sec']:.2f}")
        print(f"  ✓ Config       : "
              f"bs={throughput_results['batch_size']}, "
              f"seq_len={throughput_results['seq_len']}, "
              f"runs={throughput_results['num_runs']}")
    else:
        print("  - Throughput benchmark skipped (num_runs <= 0)")

    # Save results
    print("\n[Step 8] Saving results...")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "model_path": args.model_path,
        "quant_mode": args.quant_mode,
        "data_dir": args.data_dir,
        "model_size_mb_via_save": model_size_mb,
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test": test_results,
        "throughput": throughput_results,
    }
    if train_results is not None:
        results["train"] = train_results

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Model path        : {args.model_path}")
    print(f"Quantization      : {args.quant_mode}")
    print(f"Model size (MB)   : {model_size_mb:.1f}")
    print(f"Test PPL          : {test_results['ppl']:.4f}")
    if throughput_results is not None:
        print(f"Throughput tokens/s: {throughput_results['tokens_per_sec']:.2f}")
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
