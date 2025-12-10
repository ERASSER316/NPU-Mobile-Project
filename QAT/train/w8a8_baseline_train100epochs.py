import torch
import torchao.quantization  # 确保注册 quant 类型
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
import os

###############################################
# 1. Baseline finetune 后的模型作为 PTQ 起点
###############################################
# 改成你 baseline_finetune_train_eval.py 保存出来的目录
fp_finetune_model_path = "../../modelnew/llama3.2-1b-baseline-control"

# 导出的 W8A8 int8 模型保存目录（不要和上面混在一起）
save_dir = "../../modelnew_quantization/llama3.2-1b-fp-finetune-100steps-w8a8-int8"

os.makedirs(save_dir, exist_ok=True)

###############################################
# 2. 加载 tokenizer（用原始或 finetune 后目录都可以，一般一致）
###############################################
tokenizer = AutoTokenizer.from_pretrained(fp_finetune_model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

###############################################
# 3. 从 baseline finetune checkpoint 加载模型（BF16 浮点权重）
###############################################
print(f"Loading FP-finetuned model from {fp_finetune_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    fp_finetune_model_path,
    torch_dtype=torch.bfloat16,   # 起点是 BF16 浮点权重
    device_map={"": "cpu"},       # 在 CPU 上做 PTQ
    low_cpu_mem_usage=True,
)

###############################################
# 4. 基于 FP-finetune 权重做 W8A8 PTQ（真正的 int8 推理模型）
###############################################
print("Applying Int8DynamicActivationInt8WeightConfig (W8A8) on FP-finetuned weights...")
# Int8DynamicActivationInt8WeightConfig:
#   - 权重：静态 int8 per-channel
#   - 激活：动态 int8 per-token（runtime 计算 scale/zp）
w8a8_cfg = Int8DynamicActivationInt8WeightConfig()

# 这一步会把 Linear 等模块替换为 TorchAO 的 W8A8 量化实现，
# 权重真正变成 int8 quantized tensor，激活在推理时做动态 int8 量化。
quantize_(model, w8a8_cfg)

###############################################
# 5. 在 config 里标记 quantization_config，方便 from_pretrained 识别
###############################################
model.config.quantization_config = TorchAoConfig(quant_type=w8a8_cfg)

###############################################
# 6. 保存最终的 “FP-finetune + W8A8 PTQ” 模型
###############################################
print(f"Saving FP-finetune + W8A8 PTQ model to {save_dir}")
# 注意：TorchAO 量化模型目前需要 safe_serialization=False
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)

print("Done.")
print("You can now eval this model as an offline int8 W8A8 model.")
