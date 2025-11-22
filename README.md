# NPU-Mobile Quantization Project

This repository consolidates datasets, quantization scripts, and evaluation tooling for experimenting with post-training quantization (PTQ) and quantization-aware training (QAT) on the Llama-3.2-1B family. The workflow spans data preprocessing, QAT fine-tuning, PTQ export, and offline/online evaluation so you can track accuracy and throughput across model variants.

## Repository Layout
- **datasets/** – Raw/processed corpora plus documentation of sample counts and merge scripts for building train/test splits used across all experiments.【F:datasets/README.md†L1-L137】  
- **datasets/data_preprocessing.py** – Converts chat-style `messages` examples into tokenized tensors (`input_ids`, `attention_mask`, `labels`) saved with `Dataset.save_to_disk` for QAT/PTQ scripts.【F:datasets/data_preprocessing.py†L1-L256】  
- **QAT/train/** – Training utilities for baseline finetuning and W8A8 fake-quant QAT; scripts load the preprocessed dataset, run short training loops, and export checkpoints for later PTQ/eval.【F:QAT/train/qat_w8a8_fakequant_train.py†L4-L299】【F:QAT/train/baseline_finetune_train.py†L1-L170】  
- **PTQ/** – Post-training quantizers that turn BF16 or QAT-adapted weights into weight-only Int8/Int4 exports; saved models carry a `TorchAoConfig` for automatic deserialization.【F:PTQ/weight8only.py†L1-L34】  
- **evaluate/** – Offline/online evaluation scripts that measure perplexity, loss statistics, model size, and throughput for both baseline and quantized checkpoints.【F:evaluate/eval_online.py†L1-L200】  
- **README_quantization.md** – Existing deep-dive on how PTQ, QAT, and stored `model_quantization` artifacts relate; complements this top-level overview.【F:README_quantization.md†L1-L112】

## Data Workflow
1. **Assemble datasets** following the structure in `datasets/README.md`, which merges GSM8K, MBPP, and SQuAD into chat-formatted JSONL for training and testing.【F:datasets/README.md†L1-L137】
2. **Preprocess for training** with `datasets/data_preprocessing.py`, which validates `messages` format, applies the tokenizer chat template, tokenizes with fixed length, masks padding tokens in labels, and saves a Hugging Face dataset to disk for reuse.【F:datasets/data_preprocessing.py†L24-L256】
3. **Reuse the saved dataset** path (`--data_dir`) across QAT, PTQ evaluation, and baseline finetune scripts to keep splits consistent.【F:datasets/data_preprocessing.py†L214-L253】

## Training and Quantization Pipelines
### Baseline Finetune (FP/BF16)
- `QAT/train/baseline_finetune_train.py` loads the base model on CPU, trains with autocast and gradient clipping for a fixed number of steps, then saves the finetuned BF16 checkpoint plus tokenizer and meta info (model size, steps, hyperparameters).【F:QAT/train/baseline_finetune_train.py†L65-L170】

### QAT: W8A8 Fake-Quant
- `QAT/train/qat_w8a8_fakequant_train.py` inserts W8A8 fake-quant observers (`IntxFakeQuantizeConfig`), trains with quantization noise, converts the model back to BF16 weights via `QATConfig(step="convert")`, and records meta data for downstream PTQ or evaluation.【F:QAT/train/qat_w8a8_fakequant_train.py†L135-L299】

### Post-Training Quantization (Weight-Only)
- `PTQ/weight8only.py` (and the analogous Int4 variant) load the BF16 checkpoint, apply `Int8WeightOnlyConfig` via `quantize_`, attach a `TorchAoConfig`, and save deployable weight-only exports under `model_quantization/`.【F:PTQ/weight8only.py†L1-L34】

## Evaluation
- `evaluate/eval_online.py` benchmarks baseline vs. online PTQ (quantize-on-load) models: builds dataloaders from the preprocessed dataset, computes perplexity over non-masked tokens, measures serialized model size, and profiles throughput with synthetic inputs.【F:evaluate/eval_online.py†L52-L200】
- `evaluate/eval_offline.py` (not shown above) mirrors the workflow for already-quantized checkpoints, including optional compiled-model throughput comparisons.

## Example End-to-End Flow
1. Prepare merged JSONL data as described in `datasets/README.md`, then run `datasets/data_preprocessing.py` to generate a saved dataset directory.【F:datasets/README.md†L5-L103】【F:datasets/data_preprocessing.py†L97-L218】
2. Optionally finetune the BF16 model with `baseline_finetune_train.py` to establish a stronger starting point.【F:QAT/train/baseline_finetune_train.py†L65-L170】
3. Run `qat_w8a8_fakequant_train.py` to obtain a QAT-adapted BF16 checkpoint ready for PTQ or direct evaluation.【F:QAT/train/qat_w8a8_fakequant_train.py†L135-L299】
4. Export weight-only PTQ variants (e.g., Int8) using `PTQ/weight8only.py`; the saved directory can be loaded by Hugging Face with quantization config embedded.【F:PTQ/weight8only.py†L1-L34】
5. Evaluate accuracy and throughput with `evaluate/eval_online.py` for on-the-fly PTQ or `evaluate/eval_offline.py` for offline exports, keeping `--data_dir` pointed to the preprocessed dataset for consistent metrics.【F:evaluate/eval_online.py†L52-L200】
