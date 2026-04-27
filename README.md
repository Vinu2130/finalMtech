# SmallLM Reasoning Baseline

This repository contains a minimal, reproducible pipeline for:

- data preparation (coin flip, LLC, optional GSM-Symbolic),
- train/valid/test split creation,
- baseline finetuning with `t5-base`.

## Quick start (local or Kaggle)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare data:

```bash
python scripts/01_prepare_data.py --output_dir data/processed --coin_n 8000 --llc_n 8000
```

Optional GSM-Symbolic ingestion:

```bash
python scripts/01_prepare_data.py --output_dir data/processed --coin_n 8000 --llc_n 8000 --gsm_symbolic_file data/raw/gsm_symbolic.jsonl --gsm_symbolic_n 4000
```

3. Build splits:

```bash
python scripts/02_build_splits.py --input_dir data/processed --output_dir data/splits
```

4. Train direct-answer baseline:

```bash
python scripts/03_train_baseline.py --task coin_flip --method direct --train_file data/splits/coin_flip/train.jsonl --valid_file data/splits/coin_flip/valid_iid.jsonl --output_dir outputs/coin_flip_direct
```

PowerShell helper script:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts/run_day1.ps1"
```

## Data schema

Every JSONL line contains:

- `id`
- `task`
- `question`
- `answer`
- `rationale_free`
- `rationale_structured`
- `meta`

## Methods

- `direct`: target is `Answer: <answer>`
- `free_cot`: target is `rationale_free + "\nAnswer: <answer>"`
- `structured_cot`: target is `rationale_structured`

