import json
from pathlib import Path
from itertools import chain

SYS_PROMPT = (
    "You are a helpful assistant that can solve math word problems, write Python functions, "
    "and answer reading comprehension questions."
)

root = Path('/workspace/torchao/datasets')
out_dir = root / 'processed'
out_dir.mkdir(exist_ok=True)
dst_path_train = out_dir / 'mbpp_train_sample_llama.jsonl'
dst_path_test = out_dir / 'mbpp_test_sample_llama.jsonl'
def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')

# MBPP

def convert_mbpp(src_path, dst_path):
    records_train = []
    records_test = []
    with src_path.open(encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            if ex['task_id'] > 510 and ex['task_id'] < 975:
                if not line.strip():
                    continue
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
                records_train.append({
                    'messages': [
                        {'role': 'system', 'content': SYS_PROMPT},
                        {'role': 'user', 'content': user_prompt},
                        {'role': 'assistant', 'content': code},
                    ]
                })
            else:   
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
                records_test.append({
                    'messages': [
                        {'role': 'system', 'content': SYS_PROMPT},
                        {'role': 'user', 'content': user_prompt},
                        {'role': 'assistant', 'content': code},
                    ]
                })
    write_jsonl(dst_path_train, records_train)
    write_jsonl(dst_path_test, records_test)
    return len(records_train), len(records_test)
counts = {}
counts['mbpp_train'] = convert_mbpp(root / 'mbpp/mbpp.jsonl', out_dir / 'mbpp_train_sample_llama.jsonl')
counts['mbpp_test'] = convert_mbpp(root / 'mbpp/mbpp.jsonl', out_dir / 'mbpp_test_sample_llama.jsonl')
for k, v in counts.items():
    print(f"{k}: {v} samples")