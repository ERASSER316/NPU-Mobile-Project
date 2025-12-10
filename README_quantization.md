# TorchAO Quantization Project Overview

This README summarizes the structure and responsibilities of the three key directories that drive quantization experiments in this workspace: `model_quantization`, `PTQ`, and `QAT`. It also outlines how the assets in each directory connect to form a full post-training and quantization-aware training workflow.

## Directory Maps

### `model_quantization/`

```
model_quantization/
├─ llama-3.2-1b-int4woq-ptq/
├─ llama-3.2-1b-int8woq-ptq/
├─ llama-3.2-1b-w8a8-qat-fake/
├─ llama-3.2-1b-w8a8-qat-int8/
└─ llama-3.2-1b-int4woq-qat/
```

- Each subdirectory is a self-contained Hugging Face model export (`config.json`, `tokenizer*.json`, `pytorch_model.bin`, `chat_template.jinja`, etc.).
- Naming indicates provenance and quantization scheme:
  - `*-ptq` → produced by post-training quantization scripts in `PTQ/`.
  - `*-qat` → produced by QAT scripts under `QAT/train/`, optionally followed by a PTQ conversion pass.
- Special cases worth highlighting:
  - `llama-3.2-1b-w8a8-qat-fake/`: QAT-converted FP/BF16 checkpoint (fake-quant wrappers removed) that already adapted weights to W8A8 constraints; acts as the teacher for the subsequent PTQ step, not a deployable quantized model.
  - `llama-3.2-1b-w8a8-qat-int8/`: Final W8A8 inference model obtained by applying `Int8DynamicActivationInt8WeightConfig` to the QAT teacher checkpoint—this is the deployable target.
  - `llama-3.2-1b-int4woq-qat/`: Int4 weight-only export from QAT. The files are structurally correct, but current 3090 GPUs lack the required kernels for runtime benchmarking; treat results as experimental until run on int4-capable hardware.
- Artifacts are consumed by the evaluation scripts (`eval_offline.py`, downstream inference, or deployment).

### `PTQ/`

```
PTQ/
├─ eval_online.py
├─ eval_offline.py
├─ weight4only.py
├─ weight8only.py
├─ w8a8_qat_fake.py
├─ baseline_results.json
├─ baseline_offline_results.json
├─ PTQ_WeightINT8WOQ_results.json
├─ PTQ_offline_int8woq_results.json
└─ w8a8_qat_int8_results.json
```

- **Creation scripts**
  - `weight4only.py`: PTQ pipeline that clones the BF16 base model, applies `Int4WeightOnlyConfig`, tags config with `TorchAoConfig`, and saves into `model_quantization/llama-3.2-1b-int4woq-ptq`.
  - `weight8only.py`: Same flow for `Int8WeightOnlyConfig`, exporting to `llama-3.2-1b-int8woq-ptq`.
  - `w8a8_qat_fake.py`: Converts a fake-quant QAT checkpoint into a real W8A8 model using `Int8DynamicActivationInt8WeightConfig`, exporting to `llama-3.2-1b-w8a8-qat-int8`. _Note: despite the filename, this script produces the deployable int8 model._
- **Evaluation scripts**
  - `eval_online.py`: Runs baseline vs. online PTQ (quantize-on-load) evaluations—perplexity, throughput, and serialized size—directly from the original BF16 checkpoint.
  - `eval_offline.py`: Evaluates already quantized model directories (baseline, PTQ, or QAT outputs) for perplexity and throughput without re-quantization.
- **Result snapshots** (`*.json`): Persist benchmark outputs for quick comparisons across runs and quantization modes.

### `QAT/`

```
QAT/
├─ QAT_TRAIN.py
├─ data_preprocessing.py
├─ baseline/
│  ├─ baseline_eval.py
│  └─ baseline_results.json
└─ train/
   ├─ qat_w8a8_fakequant_train_eval.py
   ├─ qat_int4woq_train_eval.py
   ├─ weight4only.py
   └─ qat_w8a8_qat_fake_results.json
```

- **Data preparation**
  - `data_preprocessing.py`: Converts JSON/chat-style datasets into tokenized HF `Dataset` objects (with `input_ids`, `attention_mask`, `labels`) suitable for QAT scripts.
- **Baseline evaluation**
  - `baseline/baseline_eval.py`: Captures BF16 reference perplexity, loss statistics, and throughput; baseline metrics stored in `baseline_results.json`.
- **Training flows**
  - `QAT_TRAIN.py`: Prototype accelerator-backed runner using `Int8DynActInt4WeightQATQuantizer` with cosine scheduling and periodic checkpointing; currently experimental and not used for final reported numbers.
  - `train/qat_w8a8_fakequant_train_eval.py`: Modular CLI that inserts fake-quant W8A8 observers, trains, evaluates in-place, converts (step=`convert`) to a clean FP/BF16 checkpoint, and records metrics.
  - `train/qat_int4woq_train_eval.py`: Similar scaffold for Int4 weight-only QAT; conversions succeed, but benchmarking on the 3090 is blocked by missing int4-compatible kernels (treat as experimental).
  - `train/weight4only.py`: Minimal repro script showing manual Int4 WOQ QAT prepare/convert flow.
- **Outputs**
  - **W8A8 path**
    - During training, the model carries fake-quant modules purely for supervision and diagnostics.
    - After `step=convert`, weights (now FP/BF16) are saved to `model_quantization/llama-3.2-1b-w8a8-qat-fake`, providing the QAT-informed teacher for a follow-up PTQ pass.
    - The deployable W8A8 inference model is produced by `PTQ/w8a8_qat_fake.py`, landing in `model_quantization/llama-3.2-1b-w8a8-qat-int8`.
  - **Int4 path**
    - `train/qat_int4woq_train_eval.py` exports to `model_quantization/llama-3.2-1b-int4woq-qat`. Files load correctly, but runtime benchmarking requires hardware with native Int4 support.

## Cross-Directory Workflow

1. **Data Preparation**  
   - Use `QAT/data_preprocessing.py` to turn conversational datasets into fixed-length tensors stored via `Dataset.save_to_disk`.

2. **Baseline Measurement**  
   - Run `QAT/baseline/baseline_eval.py` for reference metrics; results guide expected fidelity after quantization.

3. **QAT Training (optional)**  
   - Train with `QAT/QAT_TRAIN.py` or the task-specific scripts under `QAT/train/`.  
   - Export QAT-informed checkpoints (e.g., `llama-3.2-1b-w8a8-qat-fake`) after `step=convert`; these act as improved teachers / initialization for subsequent PTQ (e.g., `w8a8_qat_fake.py`).


4. **Model Conversion / PTQ**  
   - For weight-only PTQ: execute `PTQ/weight4only.py` or `PTQ/weight8only.py`.  
   - For QAT-to-PTQ handoff: run `PTQ/w8a8_qat_fake.py` against QAT outputs to produce deployable W8A8 checkpoints.

5. **Evaluation & Benchmarking**  
   - All runs reuse the same preprocessed Hugging Face dataset splits for consistency.  
   - Perplexity is computed from aggregated NLL over tokens where `labels != -100`, ensuring comparable masking across experiments.  
   - Throughput benchmarks rely on fixed `(batch_size, seq_len)` synthetic inputs to isolate kernel/model effects from dataset variance.  
   - Use `PTQ/eval_online.py` mainly for API-style or quick smoke validation of online PTQ.  
   - Use `PTQ/eval_offline.py` as the source of truth when comparing baseline, PTQ, and QAT exports.  
   - Store findings in the JSON reports to track regressions and improvements over time.

This structure keeps quantization experiments modular: `QAT/` handles data prep and training, `PTQ/` transforms checkpoints and measures performance, and `model_quantization/` stores the resulting models for reuse or deployment.


