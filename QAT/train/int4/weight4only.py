import os
import torch
from math import inf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.qat import QATConfig

model_path = "../../../models/Llama-3.2-1B-Instruct"
data_dir = "../../datasets/mergedata_preprocessed"
save_dir = "../../model_quantization/llama3.2-1b-int4woq-qat"
device = "cuda"

"""
3090用不了int4，暂时搁置此代码的使用
"""

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 1. tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. dataset
print("\n[Step 1] Loading dataset...")
dataset = load_from_disk(data_dir)
train_data = dataset["train"] if isinstance(dataset, dict) else dataset
print(f"  ✓ Train samples: {len(train_data)}")

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

train_loader = DataLoader(
    train_data,
    batch_size=1,          # 已经是1，继续保持
    shuffle=True,
    collate_fn=collate_fn,
)

# 3. 加载模型到 CPU
print("\n[Step 2] Loading base model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
)

# 关键省显存设置
model.config.use_cache = False
model.gradient_checkpointing_enable()

# 4. QAT prepare on CPU
print("\n[Step 3] Preparing QAT (Int4 weight-only) on CPU...")
base_config = Int4WeightOnlyConfig(group_size=32)
qat_prepare_cfg = QATConfig(base_config, step="prepare")
quantize_(model, qat_prepare_cfg)

# 5. move to GPU for training
model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)

max_steps = 100          # 先跑小步数验证能不能稳
log_interval = 10
global_step = 0
moving_loss = inf

print("\n[Step 4] QAT training...")
for epoch in range(999999):
    for batch in train_loader:
        global_step += 1
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        # 用 bf16 autocast，避免算子乱升 fp32
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        moving_loss = loss.item() if moving_loss is inf else 0.9 * moving_loss + 0.1 * loss.item()
        if global_step % log_interval == 0:
            print(f"[step {global_step}] loss={loss.item():.4f}, ema_loss={moving_loss:.4f}")

        if global_step >= max_steps:
            break
    if global_step >= max_steps:
        break

# 6. 转换和保存在 CPU 上完成
print("\n[Step 5] Converting to real Int4 weight-only model on CPU...")
model = model.to("cpu").eval()
torch.cuda.empty_cache()

qat_convert_cfg = QATConfig(base_config, step="convert")
quantize_(model, qat_convert_cfg)

os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)

print(f"QAT model saved to {save_dir}")

