import json
from pathlib import Path
from itertools import chain

SYS_PROMPT = (
    "You are a helpful assistant that can solve math word problems, write Python functions, "
    "and answer reading comprehension questions."
)

root = Path('/home/wcp/torchao/datasets')
out_dir = root / 'processed'
out_dir.mkdir(exist_ok=True)

def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')

# GSM8K

def convert_gsm8k(src_path, dst_path):
    records = []
    with src_path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            question = ex['question'].strip()
            answer = ex['answer'].strip()
            if '####' in answer:
                reasoning, final = answer.rsplit('####', 1)
                reasoning = reasoning.strip()
                final = final.strip()
                assistant = f"{reasoning}\nFinal answer: {final}"
            else:
                assistant = answer
            user_prompt = (
                "Solve the following math word problem step by step:\n\n"
                f"{question}"
            )
            records.append({
                'messages': [
                    {'role': 'system', 'content': SYS_PROMPT},
                    {'role': 'user', 'content': user_prompt},
                    {'role': 'assistant', 'content': assistant},
                ]
            })
    write_jsonl(dst_path, records)
    return len(records)

# MBPP

def convert_mbpp(src_path, dst_path):
    records = []
    with src_path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            text = ex['text'].strip()
            code = ex['code'].strip().replace('\r\n', '\n')
            tests = ex.get('test_list') or []
            test_section = ''
            if tests:
                test_section = '\n\nUse these tests for validation:\n' + '\n'.join(tests)
            user_prompt = (
                "Write a Python function that meets this requirement."
                "\n\nRequirement:\n"
                f"{text}{test_section}"
            )
            records.append({
                'messages': [
                    {'role': 'system', 'content': SYS_PROMPT},
                    {'role': 'user', 'content': user_prompt},
                    {'role': 'assistant', 'content': code},
                ]
            })
    write_jsonl(dst_path, records)
    return len(records)

# SQuAD

def convert_squad(src_path, dst_path):
    with src_path.open(encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for entry in data['data']:
        for para in entry['paragraphs']:
            context = para['context'].strip()
            for qa in para['qas']:
                if qa.get('is_impossible'):
                    continue
                answers = qa.get('answers') or []
                if not answers:
                    continue
                answer_text = answers[0]['text'].strip()
                question = qa['question'].strip()
                user_prompt = (
                    "Answer the question using the provided context."
                    "\n\nContext:\n"
                    f"{context}\n\nQuestion:\n{question}"
                )
                records.append({
                    'messages': [
                        {'role': 'system', 'content': SYS_PROMPT},
                        {'role': 'user', 'content': user_prompt},
                        {'role': 'assistant', 'content': answer_text},
                    ]
                })
    write_jsonl(dst_path, records)
    return len(records)

counts = {}
counts['gsm8k_train'] = convert_gsm8k(root / 'gsm8K/train.jsonl', out_dir / 'gsm8k_train_llama.jsonl')
counts['gsm8k_test'] = convert_gsm8k(root / 'gsm8K/test.jsonl', out_dir / 'gsm8k_test_llama.jsonl')
#counts['mbpp'] = convert_mbpp(root / 'mbpp/mbpp.jsonl', out_dir / 'mbpp_llama.jsonl')
counts['squad_dev'] = convert_squad(root / 'SQuAD/raw/dev-v2.0.json', out_dir / 'squad_dev_llama.jsonl')

for k, v in counts.items():
    print(f"{k}: {v} samples")

#采样6000个样本训练集，平衡gsm8K和mbpp
import json
import random
from pathlib import Path

SYS_PROMPT = (
    "You are a helpful assistant that can solve math word problems, write Python functions, "
    "and answer reading comprehension questions."
)

def normalize_entries(data_iter):
    for entry in data_iter:
        for para in entry['paragraphs']:
            context = para['context'].strip()
            for qa in para['qas']:
                if qa.get('is_impossible'):
                    continue
                answers = qa.get('answers') or []
                if not answers:
                    continue
                answer_text = answers[0]['text'].strip()
                question = qa['question'].strip()
                user_prompt = (
                    "Answer the question using the provided context."
                    "\n\nContext:\n"
                    f"{context}\n\nQuestion:\n{question}"
                )
                yield {
                    'messages': [
                        {'role': 'system', 'content': SYS_PROMPT},
                        {'role': 'user', 'content': user_prompt},
                        {'role': 'assistant', 'content': answer_text},
                    ]
                }

def reservoir_sample(iterable, k, seed=42):
    rnd = random.Random(seed)
    reservoir = []
    for i, item in enumerate(iterable, start=1):
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            j = rnd.randrange(i)
            if j < k:
                reservoir[j] = item
    return reservoir

src_path = Path('/home/wcp/torchao/datasets/SQuAD/raw/train-v2.0.json')
out_path = Path('/home/wcp/torchao/datasets/processed/squad_train_sample_llama.jsonl')
target_samples = 6000  # balance vs gsm8K (~7.4k) and mbpp (~1k)

with src_path.open('r', encoding='utf-8') as f:
    data = json.load(f)

sampled_records = reservoir_sample(normalize_entries(data['data']), target_samples)

with out_path.open('w', encoding='utf-8') as f:
    for rec in sampled_records:
        json.dump(rec, f, ensure_ascii=False)
        f.write('\n')

print(f"Wrote {len(sampled_records)} samples to {out_path}")