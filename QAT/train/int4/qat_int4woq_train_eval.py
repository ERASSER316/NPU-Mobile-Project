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

from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.qat import QATConfig
from torchao.utils import benchmark_model
"""
由于3090显卡不支持Int4，所以这个脚本在后面的eval阶段报错了
这个版本的代码目前为半成品，但是可以未来迁移到A800显卡上进行评估。
"""

# =====================
# 配置 & 参数解析
# =====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="QAT (Int4 Weight-Only) training + eval for Llama-3.2-1B"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base pretrained model (FP/BF16)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to preprocessed dataset directory (load_from_disk)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Where to save QAT-converted Int4WOQ model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training/eval device: cuda or cpu")

    # QAT 训练相关
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size for QAT training")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="QAT training steps")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for QAT optimizer")

    # Eval 相关
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Batch size for evaluation (PPL & throughput)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Max samples for eval splits")
    parser.add_argument("--throughput_runs", type=int, default=50,
                        help="Num runs for benchmark_model throughput test")
    parser.add_argument("--output_file", type=str, default="qat_int4woq_eval_results.json",
                        help="JSON to save eval results")

    return parser.parse_args()


# =====================
# 数据组装
# =====================

def collate_fn(examples):
    input_ids = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
    attention_mask = [torch.tensor(e["attention_mask"], dtype=torch.long) for e in examples]
    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


# =====================
# PPL 评测（沿用你原来的逻辑）
# =====================

def evaluate_model(model, data_loader, device, split_name="test", max_samples=None):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    all_losses = []
    samples_evaluated = 0

    print(f"\n[Eval] Evaluating on {split_name} set...")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            if max_samples is not None and samples_evaluated >= max_samples:
                break

            for k in ["input_ids", "attention_mask", "labels"]:
                if k in batch:
                    batch[k] = batch[k].to(device)

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
    return {
        "ppl": float(ppl),
        "avg_loss": float(avg_loss),
        "loss_std": float(loss_tensor.std().item()),
        "loss_min": float(loss_tensor.min().item()),
        "loss_max": float(loss_tensor.max().item()),
        "total_tokens": int(total_tokens),
        "samples_evaluated": int(samples_evaluated),
    }


# =====================
# 模型大小（官方 quick_start 风格）
# =====================

def get_model_size_mb_via_save(model, tmp_path="__tmp_qat_int4_model__.pt"):
    torch.save(model, tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb


# =====================
# 吞吐测速（torchao.utils.benchmark_model 风格）
# =====================

class CausalLMWrapper(nn.Module):
    """
    benchmark_model 要求 model(*example_inputs)，这里包装成 (input_ids, attention_mask) -> logits
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
    用数据集第一条的长度作为 seq_len，构造随机输入，保证：
    - baseline / QAT / PTQ 都用同一个 (bs, seq_len) 形状，公平对比。
    """
    sample = dataset[0]
    seq_len = len(sample["input_ids"])
    # 简单用 vocab 较大值近似；真实 vocab_size 对测速不敏感
    vocab_max = max(sample["input_ids"]) if len(sample["input_ids"]) > 0 else 32000
    input_ids = torch.randint(
        low=10,
        high=max(vocab_max, 1000),
        size=(batch_size, seq_len),
        device=device,
    )
    attention_mask = torch.ones_like(input_ids, device=device)
    return (input_ids, attention_mask), seq_len


def benchmark_throughput(model, dataset, batch_size, device, num_runs=50):
    if num_runs <= 0:
        return None

    model.eval()
    wrapper = CausalLMWrapper(model).to(device)

    example_inputs, seq_len = make_example_inputs_from_dataset(dataset, batch_size, device)

    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

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


# =====================
# 主流程：QAT + 转换 + 保存 + 评测
# =====================

def main():
    args = parse_args()

    print("=" * 60)
    print("QAT Int4 Weight-Only - Train + Eval")
    print("=" * 60)
    print(f"Base model path   : {args.model_path}")
    print(f"Data dir          : {args.data_dir}")
    print(f"Save dir          : {args.save_dir}")
    print(f"Train batch size  : {args.train_batch_size}")
    print(f"Max QAT steps     : {args.max_steps}")
    print(f"Eval batch size   : {args.eval_batch_size}")
    print(f"Throughput runs   : {args.throughput_runs}")
    print("=" * 60)

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Backend configs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. 数据加载
    print("\n[Step 1] Loading dataset...")
    dataset = load_from_disk(args.data_dir)
    if isinstance(dataset, dict):
        train_data = dataset.get("train", None)
        test_data = dataset.get("test", None)
    else:
        train_data = dataset
        test_data = None

    if train_data is None:
        raise ValueError("Train split is required for QAT but not found.")
    if test_data is None:
        # 没有 test 就用 train 当 eval，不过更推荐有单独 test
        print("Warning: test split not found, using train split as eval set.")
        test_data = train_data

    print(f"  ✓ Train samples: {len(train_data)}")
    print(f"  ✓ Test  samples: {len(test_data)}")

    # Eval 子集裁剪
    if args.max_eval_samples is not None:
        test_data = test_data.select(range(min(args.max_eval_samples, len(test_data))))
        print(f"  ✓ Eval limited to {len(test_data)} samples")

    # DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 2. tokenizer
    print("\n[Step 2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("  ✓ Set pad_token_id = eos_token_id")

    # 3. 加载基线模型到 CPU
    print("\n[Step 3] Loading base model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    # 4. QAT prepare
    print("\n[Step 4] Preparing QAT (Int4 weight-only) on CPU...")
    base_config = Int4WeightOnlyConfig(group_size=32)
    qat_prepare_cfg = QATConfig(base_config=base_config, step="prepare")
    quantize_(base_model, qat_prepare_cfg)

    # 5. QAT 训练
    print("\n[Step 5] QAT training...")
    model = base_model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    moving_loss = None

    for epoch in range(10**9):
        for batch in train_loader:
            global_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if moving_loss is None:
                moving_loss = loss.item()
            else:
                moving_loss = 0.9 * moving_loss + 0.1 * loss.item()

            if global_step % 10 == 0:
                print(f"[step {global_step}] loss={loss.item():.4f}, ema_loss={moving_loss:.4f}")

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    print(f"  ✓ QAT training finished. total_steps={global_step}")

    # 6. QAT convert + 保存
    print("\n[Step 6] Converting to real Int4 weight-only model on CPU...")
    model = model.to("cpu").eval()
    torch.cuda.empty_cache()

    qat_convert_cfg = QATConfig(base_config=base_config, step="convert")
    quantize_(model, qat_convert_cfg)

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.save_dir)
    print(f"  ✓ QAT Int4WOQ model saved to {args.save_dir}")

    # 7. 模型大小（via save）
    print("\n[Step 7] Measuring QAT model size via torch.save(...)")
    model_size_mb = get_model_size_mb_via_save(model)
    print(f"  ✓ QAT Int4WOQ model size: {model_size_mb:.1f} MB")

    # 8. 使用同一模型做 PPL 评测
    # （不重新 from_pretrained，直接用内存中的已 convert 模型，避免反序列化坑）
    print("\n[Step 8] Evaluating QAT model on test set (PPL)...")
    eval_device = device if device.type == "cuda" else torch.device("cpu")
    model.to(eval_device)
    qat_test_results = evaluate_model(
        model, test_loader, eval_device,
        split_name="test", max_samples=args.max_eval_samples
    )

    print(f"\n[Test Results - QAT Int4WOQ]")
    print(f"  PPL          : {qat_test_results['ppl']:.4f}")
    print(f"  Avg Loss     : {qat_test_results['avg_loss']:.4f}")
    print(f"  Loss Std     : {qat_test_results['loss_std']:.4f}")
    print(f"  Loss Range   : [{qat_test_results['loss_min']:.4f}, {qat_test_results['loss_max']:.4f}]")
    print(f"  Samples      : {qat_test_results['samples_evaluated']}")
    print(f"  Total tokens : {qat_test_results['total_tokens']}")

    # 9. 吞吐测速（官方 benchmark_model 风格）
    print("\n[Step 9] Benchmarking throughput with torchao.utils.benchmark_model...")
    throughput_results = benchmark_throughput(
        model=model,
        dataset=test_data,
        batch_size=args.eval_batch_size,
        device=eval_device,
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

    # 10. 保存结果到 JSON
    print("\n[Step 10] Saving results JSON...")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "base_model_path": args.model_path,
        "qat_model_path": args.save_dir,
        "quant_mode": "qat_int4_weight_only",
        "data_dir": args.data_dir,
        "train_steps": args.max_steps,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "model_size_mb_via_save": model_size_mb,
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test": qat_test_results,
        "throughput": throughput_results,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Results saved to {output_path}")

    print("\n" + "=" * 60)
    print("QAT Int4WOQ Train + Eval Summary")
    print("=" * 60)
    print(f"QAT model path     : {args.save_dir}")
    print(f"Model size (MB)    : {model_size_mb:.1f}")
    print(f"Test PPL           : {qat_test_results['ppl']:.4f}")
    if throughput_results is not None:
        print(f"Throughput tokens/s: {throughput_results['tokens_per_sec']:.2f}")
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
