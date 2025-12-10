# QAT for Llama‑3.2‑1B‑Instruct with TorchAO — Starter Kit

> 适配最新版 **torchao**（PyTorch AO）QAT 工作流；面向本地已有权重与自备 5 个微调数据集的场景；可直接运行。

---

## 0. 环境与依赖

**基础**
- Python ≥ 3.10，PyTorch ≥ 2.4（建议 2.5+）
- CUDA 12.x / ROCm（视硬件）

**核心库**
```bash
pip install -U torchao transformers>=4.44 accelerate datasets peft bitsandbytes
pip install -U torchtune  # 如需使用 torchtune 集成的分布式 QAT 配方
```
> 参考：TorchAO 文档 Quick Start 与 QAT 教程。

可选：
```bash
pip install flash-attn --no-build-isolation  # GPU 支持时
pip install evaluate tqdm tensorboard
```

---

## 1. 将 Llama‑3.2‑1B‑Instruct 与 TorchAO 整合

### 1.1 加载本地权重

假设你的模型在：`/models/meta-llama/Llama-3.2-1B-Instruct/`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "/models/meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,  # 建议 bf16 训练
    low_cpu_mem_usage=True,
    device_map="auto",
)
model.config.use_cache = False  # 训练需关闭
```

### 1.2 注入 TorchAO QAT 量化（伪量化）

为线性层应用 **激活 int8（动态/每 token）+ 权重 int4（分组/每通道）** 的 QAT：

```python
import torch
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
from torchao.quantization.qat import prepare_qat

# 典型设置：组大小 128~256；scales/zp 使用 fp32 以稳定训练
qat_quantizer = Int8DynActInt4WeightQATQuantizer(
    groupsize=256,
    padding_allowed=False,
    precision=torch.float32,
    scales_precision=torch.float32,
)

# 过滤：仅对 Linear 层与注意力/MLP 路径施加 QAT；保留 LayerNorm、Embeddings 为浮点
def include_fn(module_name, module):
    name = module.__class__.__name__.lower()
    if "linear" in name:
        return True
    return False

qmodel = prepare_qat(model, qat_quantizer, include_function=include_fn)
```
> 上述 `prepare_qat` 将在训练中对目标层插入 **fake quant**（训练期间仍以浮点计算），从而逼近部署时的 int8/int4 行为。

---

## 2. 量化策略与训练配置（1B 模型推荐）

**推荐策略 A（通用、稳定）**
- **权重 int4 / 激活 int8（动态/每 token）**：`Int8DynActInt4WeightQATQuantizer`
- 组大小（group size）：128 或 256（1B 建议 256，显存更省；128 精度略好）
- AMP：bf16（或 fp16）
- 学习率：1e-5 ~ 2e-5（SFT）
- 有效 batch size（全局）：256~1024 tokens/step（视显存与梯度累积）
- 训练步数：5k~20k（根据 5 个数据集规模调整）
- Warmup：3% 步数；Cosine decay
- 优化器：AdamW(betas=(0.9, 0.95), weight_decay=0.1)
- 正则：LoRA（若需低显存+稳健收敛），r=8~16, alpha=16~32, dropout=0.05
- 梯度检查点：开启（显存友好）
- Flash-Attn：有则开

**策略 B（更省显存/更激进）**
- **权重 int4-only QAT**（先做权重 QAT，激活保留 fp16/bf16），收敛更稳、推理侧再切换激活动态量化。

**Mixed Precision 与编译**
- `torch.compile`（dynamic=True）可选；需确保与 torchao 伪量化模块兼容。

---

## 3. 数据集准备与加载（5 个数据集合并）

假设你有 5 个数据源，统一转为 **instruction-tuning** 格式：
```json
{"instruction": "...", "input": "...", "output": "..."}
```
或对话 SFT：
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

**合并与清洗建议**
- 统一字符集与空白符；去除重复与超长样本
- 最终拼接为一个 `train.jsonl` / `val.jsonl`，或使用 `datasets.DatasetDict`
- 控制样本最大长度（e.g., 4096 tokens），避免长尾拖垮吞吐

**模板化与分词**

```python
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# 假设五个 jsonl 路径
paths = ["/data/ds1.jsonl","/data/ds2.jsonl","/data/ds3.jsonl","/data/ds4.jsonl","/data/ds5.jsonl"]
raw_ds = [load_dataset("json", data_files=p, split="train") for p in paths]
train_raw = Dataset.from_dict({k: sum([ds[k] for ds in raw_ds], [])})  # 简易合并

# 切分验证集（或使用你自带 dev 集）
raw = train_raw.train_test_split(test_size=0.02, seed=42)

SYS_PROMPT = "You are a helpful, multilingual assistant."

def format_example(ex):
    if "messages" in ex:  # chat 格式
        msgs = ex["messages"]
    else:  # 强制转 chat 格式
        msgs = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": (ex.get("instruction","")) + ("\n"+ex.get("input",""))},
            {"role": "assistant", "content": ex.get("output","")},
        ]
    ex["messages"] = msgs
    return ex

raw = raw.map(format_example, num_proc=4)

# 使用 transformers 的 chat 模板生成 labels（忽略系统与用户 token 的损失）
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

IGNORE_INDEX = -100

def tokenize_chat(batch):
    inputs = []
    labels = []
    for msgs in batch["messages"]:
        # 使用新式 chat 模板
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(text, truncation=True, max_length=4096)
        # 构造监督标签：仅对 assistant 段落计算 loss
        label_ids = [IGNORE_INDEX] * len(tokenized["input_ids"])
        # 粗略：按照最后一条 assistant 回复做监督
        # 更精细可按 message 边界标注
        if msgs[-1]["role"] == "assistant":
            resp = tokenizer(msgs[-1]["content"], truncation=True, max_length=4096)
            L = len(resp["input_ids"]) - 1  # 去掉 eos 重复
            label_ids[-L:] = resp["input_ids"][1:]
        inputs.append(tokenized)
        labels.append(label_ids)
    out = {
        "input_ids": [x["input_ids"] for x in inputs],
        "attention_mask": [x["attention_mask"] for x in inputs],
        "labels": labels,
    }
    return out

proc = raw.map(tokenize_chat, batched=True, remove_columns=raw["train"].column_names)
proc = proc.with_format("torch")
```

---

## 4. 完整训练脚本（单机多卡/单卡均可）

保存为 `train_qat_llama32_1b.py`：

```python
import os, math, json, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup)
from accelerate import Accelerator
from datasets import load_from_disk
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer, prepare_qat

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/meta-llama/Llama-3.2-1B-Instruct")
DATA_DIR = os.environ.get("DATA_DIR", "/data/merged_ds")  # 可换成上节 map 后的 dataset.save_to_disk 目录
OUTPUT = os.environ.get("OUTPUT", "./qat_runs/llama32_1b_int4w_int8a")

BF16 = True
LR = 1e-5
WD = 0.1
BETAS = (0.9, 0.95)
EPOCHS = 2
WARMUP = 0.03
MAX_LEN = 4096
BATCH_SIZE = 1  # per-device micro-batch
GRAD_ACCUM = 8  # 累计到合适的 tokens/step
MAX_NORM = 1.0
SEED = 42

accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, log_with=["tensorboard"])
accelerator.init_trackers("qat_llama32_1b")

torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if BF16 else torch.float16,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False

# --- TorchAO QAT 注入 ---
qat = Int8DynActInt4WeightQATQuantizer(groupsize=256, precision=torch.float32, scales_precision=torch.float32)

def include_fn(name, module):
    return isinstance(module, nn.Linear)

model = prepare_qat(model, qat, include_function=include_fn)

# --- 数据加载 ---
from datasets import load_from_disk

if os.path.isdir(DATA_DIR):
    ds = load_from_disk(DATA_DIR)
else:
    raise FileNotFoundError("DATA_DIR not found; please save your processed dataset via .save_to_disk")

train_dl = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(ds["test"], batch_size=1, shuffle=False, num_workers=2)

# --- 优化器与调度 ---
from torch.optim import AdamW

no_decay = ["bias", "LayerNorm.weight"]
param_groups = [
    {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WD},
    {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = AdamW(param_groups, lr=LR, betas=BETAS)

num_steps_per_epoch = math.ceil(len(train_dl) / GRAD_ACCUM)
max_steps = int(EPOCHS * num_steps_per_epoch)
warmup_steps = int(WARMUP * max_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

# --- 准备分布式 ---
model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)

# --- 训练 ---
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

os.makedirs(OUTPUT, exist_ok=True)

def evaluate_ppl():
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in val_dl:
            outputs = model(**{k: v.to(model.device) for k,v in batch.items() if k in ["input_ids","attention_mask","labels"]})
            loss = outputs.loss
            n_tokens = (batch["labels"] != -100).sum().item()
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens
    ppl = math.exp(total_nll / max(1,total_tokens))
    return ppl

best_ppl = float("inf")

for step, batch in enumerate(tqdm(train_dl, total=len(train_dl))):
    model.train()
    with accelerator.accumulate(model):
        batch = {k: v.to(model.device) for k,v in batch.items()}
        out = model(**batch)
        loss = out.loss
        accelerator.backward(loss)
        clip_grad_norm_(model.parameters(), MAX_NORM)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    
    if accelerator.is_main_process and (accelerator.step % 100 == 0):
        accelerator.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=accelerator.step)
    
    if accelerator.is_main_process and (accelerator.step % 1000 == 0):
        ppl = evaluate_ppl()
        accelerator.log({"eval/ppl": ppl}, step=accelerator.step)
        if ppl < best_ppl:
            best_ppl = ppl
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(os.path.join(OUTPUT, "ckpt_best"), safe_serialization=True)
            tokenizer.save_pretrained(os.path.join(OUTPUT, "ckpt_best"))

# 最终保存
if accelerator.is_main_process:
    accelerator.print(f"Training done. Best PPL: {best_ppl:.2f}")
    accelerator.unwrap_model(model).save_pretrained(os.path.join(OUTPUT, "ckpt_last"), safe_serialization=True)
    tokenizer.save_pretrained(os.path.join(OUTPUT, "ckpt_last"))
```

> 说明：该脚本直接在 **fake‑quant** 条件下训练；训练结束后进行导出/离线量化以得到推理时的真正 int8/int4 计算图。

---

## 5. 训练后导出与推理部署

### 5.1 将 QAT 模型导出为 PT2E 量化图（静态图）

```python
import torch
from torchao.quantization import convert

# 继续在同进程内：qmodel = prepare_qat(...) 训练收敛后
qmodel.eval()
pt2e_model = torch.export.export(qmodel, (torch.randint(1,100,(1,128)),))  # 伪输入视你的前向签名而定
quantized = convert(pt2e_model)  # 根据 QAT 注释生成真正的量化算子

# 保存
quantized_module = torch.export.load(quantized)
torch.save(quantized_module, "./qat_runs/llama32_1b_int4w_int8a/pt2e_quantized.pt")
```

> 如需导出到特定后端（如 XNNPACK、ONNX/TensorRT/Qualcomm SDK），需要相应的后端量化器与转换链路。

### 5.2 推理性能/精度对比脚本

```python
import time, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 浮点基线
fp_model = AutoModelForCausalLM.from_pretrained("/models/meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16, device_map="auto")

# QAT-导出量化（伪代码，依据你的后端加载方式）
# quant_model = torch.load("./qat_runs/.../pt2e_quantized.pt").to("cuda")

prompt = "你是谁？请用中文简要回答。"

@torch.inference_mode()
def measure_latency(m, tokenizer, n_warm=5, n_run=20):
    input_ids = tokenizer(prompt, return_tensors="pt").to(m.device)
    # 预热
    for _ in range(n_warm):
        _ = m.generate(**input_ids, max_new_tokens=32)
    # 计时
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_run):
        _ = m.generate(**input_ids, max_new_tokens=32)
    torch.cuda.synchronize()
    dt = (time.time()-t0)/n_run
    return dt

# dt_fp = measure_latency(fp_model, tokenizer)
# dt_q = measure_latency(quant_model, tokenizer)
# print({"fp16": dt_fp, "qat_int4w_int8a": dt_q, "speedup": dt_fp/dt_q})
```

### 5.3 验证精度
- 语言建模：WikiText-2/3 验证困惑度（PPL）
- 常识：HellaSwag / PIQA / Winogrande（使用 `lm-eval-harness`）
- 代码/数学：基于你 5 个数据集的 held-out 验证集合或 `HumanEval`/`GSM8K`（小样评估）

---

## 6. 关键参数解释与调优

- **groupsize（权重量化组大小）**：越小越精细，精度更好但开销更大。1B 推荐 128~256。
- **precision / scales_precision**：训练时常用 fp32 以降低量化噪声的抖动，提升收敛稳定性。
- **动态激活（int8 per‑token）**：更贴合自回归生成的分布漂移；与静态激活相比更稳健。
- **LoRA + QAT**：在低显存或小样本时强烈建议结合；LoRA 仅训练 Adapter 权重，而 QAT 让全网络在 fake‑quant 分布下收敛。
- **学习率**：量化后可适当 **降低** 学习率；先 warmup 再 cosine，有助于稳定。
- **梯度裁剪**：`max_norm=1.0` 常见；QAT 噪声下可防止梯度爆炸。

**显存优化**
- 梯度检查点（`model.gradient_checkpointing_enable()`）
- 降低 `groupsize`、增大 `precision`（训练态仍是浮点，不影响显存太多）
- 降低 `max_length` 或启用 `flash-attn`

---

## 7. 常见问题与解法（FAQ）

1. **训练发散/损失不降**
   - 调低 LR（×0.5），增大 warmup；开启梯度裁剪；确保 `precision=scales_precision=torch.float32`。
2. **精度回收不足**
   - 缩小 `groupsize`（256→128）；延长微调步数；引入 LoRA；数据增强（提升覆盖与你目标推理域匹配）。
3. **导出失败 / 不支持的算子**
   - 改为先导出 `torch.export` 静态图；检查 TorchAO 版本是否支持相应后端；必要时只做权重 QAT，再在部署时做激活动态量化。
4. **吞吐下降**
   - 启用 `torch.compile`（如兼容）；打开张量并行/流水并行；减少生成阶段 max_new_tokens；核对后端 kernel 支持（int4 pack 格式）。
5. **Tokenizer 报错**
   - Llama‑3.2 使用兼容的分词文件；`pad_token` 置为 `eos_token`。

---

## 8.（可选）使用 Torchtune 的一行式 QAT 配方

```bash
# 单机多卡范例（按你的 GPU 数量调整）
TUNE_TP_ZERO=false \
    tune run --nproc_per_node 4 full_finetune_distributed \
      --config llama3_2/1B_full \
      qat.enable=true \
      qat.quantizer=int8_dynact_int4_weight \
      train.batch_size=16 train.max_steps=8000 optim.lr=1e-5
```
> Torchtune 会在其 recipe 中自动调用 TorchAO 的 QAT 注入与导出流程。

---

## 9. 参考（官方/权威）
- TorchAO 官方 QAT 博客与教程（Llama3 示例）
- TorchAO API：`Int8DynActInt4WeightQATQuantizer`、`Int8DynActInt4WeightQATLinear`、PT2E QAT 导出/转换
- Llama‑3.2‑1B‑Instruct 模型卡（本地权重同结构）

---

> 完成度：本套件已包含 **整合 → 数据 → 训练 → 导出 → 评估** 的最小可行链路。你可直接在本地替换数据路径后运行。

