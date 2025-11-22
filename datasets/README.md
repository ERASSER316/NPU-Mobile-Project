# Datasets Directory

This directory contains datasets for Llama 3.2 QAT (Quantization-Aware Training) fine-tuning.

## Directory Structure

```
datasets/
├── README.md                          # This file
│
├── raw/                               # Raw datasets (original format)
│   ├── gsm8K/
│   │   ├── train.jsonl               # GSM8K math problems (training)
│   │   └── test.jsonl                # GSM8K math problems (testing)
│   ├── mbpp/
│   │   └── mbpp.jsonl                # MBPP Python programming problems
│   └── SQuAD/
│       └── ...                       # SQuAD reading comprehension data
│
├── pre_datasets/                      # Preprocessed datasets (Llama format)
│   ├── predata.py                    # Preprocessing script
│   ├── mbpp_seperate.py              # MBPP separation script
│   ├── mbpp_llama.jsonl              # Full MBPP in Llama format
│   │
│   ├── train/                        # Training data (preprocessed)
│   │   ├── gsm8k_train_llama.jsonl   # 7,473 math problems
│   │   ├── mbpp_train_sample_llama.jsonl  # 464 coding problems
│   │   └── squad_train_sample_llama.jsonl # 6,000 reading comprehension
│   │
│   └── test/                         # Testing data (preprocessed)
│       ├── gsm8k_test_llama.jsonl    # 1,319 math problems
│       ├── mbpp_test_sample_llama.jsonl   # 510 coding problems
│       └── squad_dev_llama.jsonl     # 5,928 reading comprehension
│
└── mergedata/                         # Final merged datasets (ready for training)
    ├── merge_jsonl.py                # Merging script
    ├── train.jsonl                   # 13,937 examples (all training data merged)
    └── test.jsonl                    # 7,757 examples (all testing data merged)
```

## Dataset Statistics

### Raw Datasets
- **GSM8K**: Math word problems requiring multi-step reasoning
- **MBPP**: Python programming problems with function requirements and test cases
- **SQuAD**: Reading comprehension questions with context passages

### Preprocessed Datasets (Llama Format)
All preprocessed files follow the Llama 3.2 chat format with structured messages:

```json
{
  "messages": [
    {"role": "system", "content": "System prompt..."},
    {"role": "user", "content": "Question or task..."},
    {"role": "assistant", "content": "Answer or solution..."}
  ]
}
```

#### Training Set (pre_datasets/train/)
| Dataset | File | Examples | Task Type |
|---------|------|----------|-----------|
| GSM8K | gsm8k_train_llama.jsonl | 7,473 | Math word problems |
| MBPP | mbpp_train_sample_llama.jsonl | 464 | Python function writing |
| SQuAD | squad_train_sample_llama.jsonl | 6,000 | Reading comprehension |
| **Total** | - | **13,937** | - |

#### Test Set (pre_datasets/test/)
| Dataset | File | Examples | Task Type |
|---------|------|----------|-----------|
| GSM8K | gsm8k_test_llama.jsonl | 1,319 | Math word problems |
| MBPP | mbpp_test_sample_llama.jsonl | 510 | Python function writing |
| SQuAD | squad_dev_llama.jsonl | 5,928 | Reading comprehension |
| **Total** | - | **7,757** | - |

### Final Merged Datasets (mergedata/)
These are the ready-to-use files for Llama 3.2 QAT fine-tuning:

- **train.jsonl**: 13,937 examples (shuffled mix of all training data)
- **test.jsonl**: 7,757 examples (shuffled mix of all testing data)

Both files are shuffled with `random.seed(42)` for reproducibility.

## Data Processing Pipeline

### 1. Raw Data Collection
Original datasets are stored in `raw/` directory in their native formats.

### 2. Preprocessing (raw → pre_datasets)
The `predata.py` script converts raw data into Llama 3.2 chat format:
- Adds system prompts
- Structures conversations with role-based messages
- Separates into train/test splits
- Formats task-specific content (math, code, QA)

### 3. Merging (pre_datasets → mergedata)
The `merge_jsonl.py` script:
- Combines all three datasets (GSM8K, MBPP, SQuAD)
- Shuffles data for better training diversity
- Creates final train.jsonl and test.jsonl files

## Usage

### For Llama 3.2 QAT Fine-tuning
Use the final merged files:
```bash
# Training data
/home/wcp/torchao/datasets/mergedata/train.jsonl

# Testing data
/home/wcp/torchao/datasets/mergedata/test.jsonl
```

### For Task-Specific Training
Use individual preprocessed files from `pre_datasets/train/` or `pre_datasets/test/`.

## Data Format

All preprocessed and merged data files use JSONL format (one JSON object per line):

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Each line is a complete training example with:
- **System message**: Task instructions and context
- **User message**: The question or task
- **Assistant message**: The expected response

## Notes

- All data is encoded in UTF-8
- Data is shuffled with seed=42 for reproducibility
- The merged datasets combine three different task types to create a multi-task fine-tuning dataset
- System prompt is shared across all tasks: "You are a helpful assistant that can solve math word problems, write Python functions, and answer reading comprehension questions."

