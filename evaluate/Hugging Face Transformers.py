import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="/workspace/models/Llama-3.2-1B-Instruct",
    device="cuda"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "用一句话解释深度学习中的量化感知训练(Quantization-Aware Training, QAT)。"},
]

out = pipe(messages, max_new_tokens=128)
print(out[0]["generated_text"][-1])
