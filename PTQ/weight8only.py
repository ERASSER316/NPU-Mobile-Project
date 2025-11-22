import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
model_path = "../../models/Llama-3.2-1B-Instruct"
device = "cuda"
save_dir = "../model_quantization/llama-3.2-1b-int8woq-ptq"
#加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
#我的数据集处理时候一个样本自动填充为了1024个token大小
ptq_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=None,
)
#复制一个新的模型用于PTQ，防止改变原始模型的权重，根本不需要这么做，冗余操作
#ptq_model = copy.deepcopy(model)

from torchao.quantization import Int8WeightOnlyConfig, quantize_
woq_config = Int8WeightOnlyConfig(
    group_size=32,     # 推荐：32/64，先用32做 baseline
    # 其他用默认即可：uint4、per-group、非对称、tensor_core_tiled 布局等
)
quantize_(ptq_model, woq_config)#得到权重INT8量化后的模型
ptq_model.to(device)
#保存量化信息到config文件中
from transformers import TorchAoConfig

ptq_model.config.quantization_config = TorchAoConfig(quant_type=woq_config) 
#save
#safe_serialization=False，否则会因为 torchao 张量不兼容 safetensors 挂掉;具体原因不了解
ptq_model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)
