#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offline Eval for Baseline & Offline-Quantized Models (e.g., Int8-WOQ)

- 不做在线量化：只对已经量化并保存好的模型目录做评测
- 指标：
    * Test PPL (与之前一致的基于 NLL / 有效 token 的计算)
    * Throughput: 固定 (batch_size, seq_len) 的前向，tokens/sec
    * Model Size: torch.save(model) 文件大小 (MB)

CUDA_VISIBLE_DEVICES=2 python eval_offline.py \
  --model_path ../modelnew_quantization/llama3.2-1b-fp-finetune-100steps-w8a8-int8 \
  --data_dir ../datasets/mergedata_prev2 \
  --output_file ./resultnew/resules_llama3.2-1b-baseline_int8.json \
  --batch_size 8 \
  --device cuda \
  --quant_mode ptq_w8a8

CUDA_VISIBLE_DEVICES=2 python eval_offline.py \
  --model_path ../modelnew_quantization/llama3.2-1b-w8a8-qat-int8 \
  --data_dir ../datasets/mergedata_prev2  \
  --output_file ./resultnew/resules_llama3.2-1b-qat_int8.json \
  --batch_size 8 \
  --device cuda \
  --quant_mode qat_w8a8

CUDA_VISIBLE_DEVICES=2 python eval_offline.py \
  --model_path ../modelnew/llama3.2-1b-baseline-control \
  --data_dir ../datasets/mergedata_prev2  \
  --output_file ./resultnew/resules_llama3.2-1b-baseline-fp.json \
  --batch_size 8 \
  --device cuda \
  --quant_mode none

CUDA_VISIBLE_DEVICES=2 python eval_offline.py \
  --model_path ../modelnew/llama3.2-1b-w8a8-qat-experiment \
  --data_dir ../datasets/mergedata_prev2  \
  --output_file ./resultnew/resules_llama3.2-1b-fp-qat.json \
  --batch_size 8 \
  --device cuda \
  --quant_mode none

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

import torchao.quantization  # 确保量化类型注册
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# ========= 参数解析 =========

def parse_args():
    parser = argparse.ArgumentParser(description="Offline eval for baseline & offline PTQ models")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model directory (baseline or offline-quantized)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed dataset (load_from_disk)")
    parser.add_argument("--output_file", type=str, default="eval_offline_results.json",
                        help="JSON file to save evaluation results")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for PPL evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for test split (None = use all)")
    parser.add_argument("--eval_train", action="store_true",
                        help="Also evaluate on train split if present")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")

    parser.add_argument("--quant_mode", type=str, default="none",
                        #choices=["none", "offline_int8woq"],
                        help=(
                            "none: treat model_path as is (baseline or offline ckpt)\n"
                            "offline_int8woq: for logging only; loading逻辑与none相同，不再做在线量化"
                        ))

    parser.add_argument("--throughput_runs", type=int, default=50,
                        help="Number of runs for throughput benchmark")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


# ========= Data Collate =========

def collate_fn(examples):
    """
    假设数据集中已经是定长 or 预先 padding 好的 input_ids / attention_mask / labels。
    """
    batch = {}
    keys = examples[0].keys()
    for key in keys:
        if key in ["input_ids", "attention_mask", "labels"]:
            vals = [ex[key] for ex in examples]
            if isinstance(vals[0], list):
                batch[key] = torch.tensor(vals, dtype=torch.long)
            elif isinstance(vals[0], torch.Tensor):
                batch[key] = torch.stack(vals)
            else:
                batch[key] = torch.tensor(vals, dtype=torch.long)
    return batch


# ========= PPL Evaluation =========

def evaluate_model(model, dataloader, device, split_name="test", max_samples=None):
    """
    与之前框架一致：
    - 对每个 batch 取 loss
    - 用 labels != -100 的 token 个数加权累加 NLL
    - ppl = exp(total_nll / total_tokens)
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    all_losses = []
    samples_evaluated = 0

    print(f"\n[Eval] Evaluating on {split_name} set...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
            if max_samples is not None and samples_evaluated >= max_samples:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

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
    loss_std = float(loss_tensor.std().item())
    loss_min = float(loss_tensor.min().item())
    loss_max = float(loss_tensor.max().item())

    return {
        "ppl": float(ppl),
        "avg_loss": float(avg_loss),
        "loss_std": loss_std,
        "loss_min": loss_min,
        "loss_max": loss_max,
        "total_tokens": int(total_tokens),
        "samples_evaluated": int(samples_evaluated),
    }


# ========= Model Size (via save) =========

def get_model_size_mb_via_save(model, tmp_path="__tmp_offline_eval_model__.pt"):
    """
    官方 quick_start 风格：直接 torch.save(model)，测真实落盘体积。
    对于离线量化模型，这会包含量化权重 + 封装结构。
    """
    torch.save(model, tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


# ========= Throughput Benchmark =========

class CausalLMWrapper(nn.Module):
    """
    用于 throughput：接受 (input_ids, attention_mask) 作为位置参数。
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
    用 test 集第一条样本的长度作为 seq_len，构造随机输入：
    - 确保 baseline / quantized 在同一 (bs, seq_len) 下对比
    """
    sample = dataset[0]
    seq_len = len(sample["input_ids"])
    # vocab上界粗取一个较大值，不影响速度比较
    vocab_max = max(sample["input_ids"]) if len(sample["input_ids"]) > 0 else 32000
    vocab_max = max(vocab_max, 1000)

    input_ids = torch.randint(
        low=10,
        high=vocab_max,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)
    return (input_ids, attention_mask), seq_len


def benchmark_throughput(model, dataset, batch_size, device, num_runs=50):
    """
    使用固定形状的随机输入，评测 mean_latency_ms 和 tokens/sec。

    tokens/sec = (batch_size * seq_len) / (mean_latency_sec)
    """
    if num_runs <= 0:
        return None

    model.eval()
    wrapper = CausalLMWrapper(model).to(device)

    example_inputs, seq_len = make_example_inputs_from_dataset(dataset, batch_size, device)

    # warmup
    with torch.no_grad():
        for _ in range(5):
            _ = wrapper(*example_inputs)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = wrapper(*example_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.time() - start) * 1000.0)

    mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
    tokens_per_sec = (batch_size * seq_len) / (mean_latency_ms / 1000.0)

    return {
        "mean_latency_ms": float(mean_latency_ms),
        "tokens_per_sec": float(tokens_per_sec),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "num_runs": int(num_runs),
    }


# ========= 模型加载（离线） =========

def load_model(args, device):
    """
    离线评测逻辑：
    - 对 baseline: 直接从原始目录加载（bf16 / fp32）
    - 对 offline_int8woq: 从量化后目录加载；不再传 TorchAoConfig，不再调用 quantize_
      量化信息已包含在权重 + config 中，由 transformers + torchao 反序列化。
    """
    print("\n[Model] Loading model from:", args.model_path)

    # if args.quant_mode == "none":
    #     # baseline 或直接当作普通 ckpt（也可以用于离线量化模型，只是标签不同）
    #     #dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model_path,
    #         torch_dtype="auto",
    #         low_cpu_mem_usage=True,
    #     )
    # elif args.quant_mode == "offline_int8woq":
    #     # 离线 PTQ int8-woq：量化信息已内嵌
    #     # torch_dtype="auto" 让它按存储权重/模块来
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model_path,
    #         torch_dtype="auto",
    #         low_cpu_mem_usage=True,
    #     )
    # else:
    #     raise ValueError(f"Unsupported quant_mode: {args.quant_mode}")
    model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
    model.to(device)
    model.eval()

    # Debug 信息
    print("  - First param dtype:", next(model.parameters()).dtype)
    qconf = getattr(model.config, "quantization_config", None)
    print("  - Quantization config in config:", qconf)

    return model

def compile_model(model):
    compiled_model = torch.compile(
        model,
        mode="max-autotune",
        fullgraph=False,
    )
    return compiled_model

# ========= 主流程 =========

def main():
    args = parse_args()

    print("=" * 60)
    print("Offline Model Evaluation")
    print("=" * 60)
    print(f"Model path      : {args.model_path}")
    print(f"Quantization    : {args.quant_mode}")
    print(f"Data dir        : {args.data_dir}")
    print(f"Output file     : {args.output_file}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Max samples     : {args.max_samples}")
    print(f"Throughput runs : {args.throughput_runs}")
    print(f"Seed            : {args.seed}")
    print("=" * 60)

    # Seed everything for reproducibility across runs
    set_seed(args.seed)

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True

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
        raise ValueError("Test dataset not found in data_dir.")

    print(f"  ✓ Test samples : {len(test_data)}")
    if train_data is not None:
        print(f"  ✓ Train samples: {len(train_data)}")

    # Subsample
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
    if args.eval_train and train_data is not None:
        train_dl = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # Tokenizer
    print("\n[Step 2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("  ✓ Set pad_token_id = eos_token_id")

    # Model
    print("\n[Step 3] Loading model...")
    model = load_model(args, device)

    # Model size via save
    print("\n[Step 4] Measuring model size via torch.save(...)")
    model_size_mb = get_model_size_mb_via_save(model)
    print(f"  ✓ Model size (via save): {model_size_mb:.1f} MB")

    # Test PPL
    print("\n[Step 5] Evaluating on test set (PPL)...")
    test_results = evaluate_model(
        model,
        test_dl,
        device,
        split_name="test",
        max_samples=args.max_samples,
    )

    print("\n[Test Set Results]")
    print(f"  PPL          : {test_results['ppl']:.4f}")
    print(f"  Avg Loss     : {test_results['avg_loss']:.4f}")
    print(f"  Loss Std     : {test_results['loss_std']:.4f}")
    print(f"  Loss Range   : [{test_results['loss_min']:.4f}, {test_results['loss_max']:.4f}]")
    print(f"  Total Tokens : {test_results['total_tokens']}")
    print(f"  Samples      : {test_results['samples_evaluated']}")

    #compiled model
    print("\n Compiling model...")
    compiled_model = compile_model(model)
    print("  ✓ Model compiled")
    
    # Train PPL (optional)
    train_results = None
    if train_dl is not None:
        print("\n[Step 6] Evaluating on train set (PPL)...")
        train_results = evaluate_model(
            model,
            train_dl,
            device,
            split_name="train",
            max_samples=args.max_samples,
        )
        print("\n[Train Set Results]")
        print(f"  PPL          : {train_results['ppl']:.4f}")
        print(f"  Avg Loss     : {train_results['avg_loss']:.4f}")
        print(f"  Total Tokens : {train_results['total_tokens']}")
        print(f"  Samples      : {train_results['samples_evaluated']}")

    # Throughput benchmark
    print("\n[Step 7] Benchmarking eager model throughput with fixed shape inputs...")
    throughput_results = benchmark_throughput(
        model=model,
        dataset=test_data,
        batch_size=args.batch_size,
        device=device,
        num_runs=args.throughput_runs,
    )

    print("\n[Step 8] Benchmarking compiled model throughput with fixed shape inputs...")
    throughput_results_compiled = benchmark_throughput(
        model=compiled_model,
        dataset=test_data,
        batch_size=args.batch_size,
        device=device,
        num_runs=args.throughput_runs,
    )
    
    if throughput_results is not None:
        print("  - Eager model throughput:")
        print(f"  ✓ Mean latency (ms): {throughput_results['mean_latency_ms']:.3f}")
        print(f"  ✓ Tokens/sec       : {throughput_results['tokens_per_sec']:.2f}")
        print(f"  ✓ Config           : "
              f"bs={throughput_results['batch_size']}, "
              f"seq_len={throughput_results['seq_len']}, "
              f"runs={throughput_results['num_runs']}")
    else:
        print("  - Throughput benchmark skipped")

    if throughput_results_compiled is not None:
        print("  - Compiled model throughput:")
        print(f"  ✓ Mean latency (ms): {throughput_results_compiled['mean_latency_ms']:.3f}")
        print(f"  ✓ Tokens/sec       : {throughput_results_compiled['tokens_per_sec']:.2f}")
        print(f"  ✓ Config           : "
              f"bs={throughput_results_compiled['batch_size']}, "
              f"seq_len={throughput_results_compiled['seq_len']}, "
              f"runs={throughput_results_compiled['num_runs']}")
    else:
        print("  - Throughput benchmark skipped")
    # Save JSON
    print("\n[Step 9] Saving results JSON...")
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
        "throughput_compiled": throughput_results_compiled,
    }
    if train_results is not None:
        results["train"] = train_results

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Offline Evaluation Summary")
    print("=" * 60)
    print(f"Model path        : {args.model_path}")
    print(f"Quantization mode : {args.quant_mode}")
    print(f"Model size (MB)   : {model_size_mb:.1f}")
    print(f"Test PPL          : {test_results['ppl']:.4f}")
    if throughput_results is not None:
        print(f"Throughput tok/s  : {throughput_results['tokens_per_sec']:.2f}")
    if throughput_results_compiled is not None:
        print(f"Compiled throughput tok/s  : {throughput_results_compiled['tokens_per_sec']:.2f}")
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
