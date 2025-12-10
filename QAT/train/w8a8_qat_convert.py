import torch
import torchao.quantization  # 确保注册quant类型
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
import os

# 1. QAT 后的模型作为量化起点（不是原始BF16）
qat_model_path = "../../modelnew/llama3.2-1b-w8a8-qat-experiment"  # 换成你实际QAT保存路径
save_dir = "../../modelnew_quantization/llama3.2-1b-w8a8-qat-int8"        # 新目录，避免覆盖
device = "cuda"

os.makedirs(save_dir, exist_ok=True)

# 2. 加载 tokenizer（可以用原始或QAT目录，一般一致）
tokenizer = AutoTokenizer.from_pretrained(qat_model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 3. 从 QAT checkpoint 加载模型（QAT后实数权重）
print(f"Loading QAT model from {qat_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    qat_model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)

# 4. 基于 QAT 权重做 W8A8 PTQ（真正的推理模型）
print("Applying Int8DynamicActivationInt8WeightConfig (W8A8) on QAT weights...")
w8a8_cfg = Int8DynamicActivationInt8WeightConfig()
quantize_(model, w8a8_cfg)

# 5. 标记 quantization_config，便于 from_pretrained 正确恢复
model.config.quantization_config = TorchAoConfig(quant_type=w8a8_cfg)

# 6. 保存最终的 QAT+W8A8 模型
print(f"Saving QAT+W8A8 model to {save_dir}")
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)

print("Done.")
