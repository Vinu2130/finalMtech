import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def parse_gsm8k_answer(raw_answer: str):
    text = str(raw_answer).strip()
    m = re.search(r"####\s*([^\n]+)", text)
    final = m.group(1).strip() if m else text.splitlines()[-1].strip()
    rationale = text[: m.start()].strip() if m else text.strip()
    return rationale, final


def extract_answer(text: str) -> str:
    m = re.search(r"Answer:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    return text.strip().split("\n")[-1].strip().lower()


def build_target(row, method: str):
    if method == "direct":
        return f"Answer: {row['answer']}"
    if method == "free_cot":
        return f"{row['rationale_free']}\nAnswer: {row['answer']}"
    if method == "structured_cot":
        return row["rationale_structured"]
    raise ValueError(f"Unsupported method: {method}")


def make_dataset(rows, method):
    items = [{"input_text": r["question"], "target_text": build_target(r, method)} for r in rows]
    return Dataset.from_list(items)


def load_gsm8k_rows(seed: int, valid_ratio: float, max_train: int, max_valid: int, max_test: int):
    ds = load_dataset("gsm8k", "main")
    train_full = ds["train"].shuffle(seed=seed)
    test_full = ds["test"]

    valid_size = int(len(train_full) * valid_ratio)
    valid_split = train_full.select(range(valid_size))
    train_split = train_full.select(range(valid_size, len(train_full)))

    def to_rows(split, prefix):
        rows = []
        for i, ex in enumerate(split):
            rationale_free, answer = parse_gsm8k_answer(ex["answer"])
            rationale_structured = f"Step1: {rationale_free}\nAnswer: {answer}" if rationale_free else f"Answer: {answer}"
            rows.append(
                {
                    "id": f"{prefix}_{i:06d}",
                    "task": "gsm8k",
                    "question": ex["question"],
                    "answer": answer,
                    "rationale_free": rationale_free,
                    "rationale_structured": rationale_structured,
                    "meta": {"source": "gsm8k_main"},
                }
            )
        return rows

    train_rows = to_rows(train_split, "gsm8k_train")
    valid_rows = to_rows(valid_split, "gsm8k_valid")
    test_rows = to_rows(test_full, "gsm8k_test")

    if max_train > 0:
        train_rows = train_rows[:max_train]
    if max_valid > 0:
        valid_rows = valid_rows[:max_valid]
    if max_test > 0:
        test_rows = test_rows[:max_test]
    return train_rows, valid_rows, test_rows


def first_non_empty(row, keys, default=""):
    for key in keys:
        if key in row and row[key] is not None and str(row[key]).strip():
            return row[key]
    return default


def format_options(value):
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value) if value is not None else ""


def load_hf_split(dataset_name, dataset_config, split_name):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split_name)
    return load_dataset(dataset_name, split=split_name)


def try_load_dataset(specs):
    last_err = None
    for dataset_name, dataset_config in specs:
        try:
            ds = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)
            return ds, dataset_name, dataset_config
        except Exception as exc:  # noqa: BLE001 - we want to try fallbacks
            last_err = exc
            continue
    raise RuntimeError(f"Could not load any dataset from specs={specs}. Last error: {last_err}") from last_err


def split_train_valid(train_split, seed: int, valid_ratio: float):
    train_split = train_split.shuffle(seed=seed)
    valid_size = max(1, int(len(train_split) * valid_ratio))
    valid_split = train_split.select(range(valid_size))
    train_split = train_split.select(range(valid_size, len(train_split)))
    return train_split, valid_split


def normalize_rows(split, task_name: str, prefix: str):
    rows = []
    for i, ex in enumerate(split):
        if task_name == "gsm8k":
            rationale_free, answer = parse_gsm8k_answer(ex["answer"])
            question = ex["question"]
        elif task_name == "svamp":
            body = first_non_empty(ex, ["Body", "body", "context", "question"])
            question_part = first_non_empty(ex, ["Question", "question"])
            question = f"{body}\n{question_part}".strip()
            answer = str(first_non_empty(ex, ["Answer", "answer"])).strip()
            rationale_free = ""
        elif task_name == "aqua":
            question = str(first_non_empty(ex, ["question", "Question"])).strip()
            options = format_options(first_non_empty(ex, ["options", "Options"], default=""))
            if options:
                question = f"{question}\nOptions:\n{options}"
            answer = str(first_non_empty(ex, ["correct", "answer", "correct_option"])).strip()
            rationale_free = str(first_non_empty(ex, ["rationale", "Rationale"], default="")).strip()
        elif task_name == "mathqa":
            problem = first_non_empty(ex, ["Problem", "problem", "question"])
            options = format_options(first_non_empty(ex, ["options", "Options"], default=""))
            question = f"{problem}\nOptions:\n{options}".strip() if options else str(problem).strip()
            answer = str(first_non_empty(ex, ["correct", "answer", "correct_option"])).strip()
            rationale_free = str(first_non_empty(ex, ["Rationale", "rationale"], default="")).strip()
        else:
            raise ValueError(f"Unsupported benchmark task: {task_name}")

        rationale_structured = f"Step1: {rationale_free}\nAnswer: {answer}" if rationale_free else f"Answer: {answer}"
        rows.append(
            {
                "id": f"{prefix}_{i:06d}",
                "task": task_name,
                "question": str(question).strip(),
                "answer": str(answer).strip(),
                "rationale_free": str(rationale_free).strip(),
                "rationale_structured": str(rationale_structured).strip(),
                "meta": {"source": task_name},
            }
        )
    return rows


def cap_rows(rows, limit):
    if limit > 0:
        return rows[:limit]
    return rows


def load_benchmark_rows(task: str, seed: int, valid_ratio: float, max_train: int, max_valid: int, max_test: int):
    if task == "gsm8k":
        return load_gsm8k_rows(seed, valid_ratio, max_train, max_valid, max_test)

    specs_by_task = {
        "svamp": [("ChilleD/SVAMP", None), ("svamp", None)],
        "aqua": [("aqua_rat", "raw"), ("aqua_rat", None)],
        "mathqa": [("math_qa", None)],
    }
    if task not in specs_by_task:
        raise ValueError(f"Unsupported --task '{task}'. Supported benchmark tasks: gsm8k, svamp, aqua, mathqa.")

    ds, _, _ = try_load_dataset(specs_by_task[task])
    if "train" in ds:
        train_split = ds["train"]
    else:
        first_key = list(ds.keys())[0]
        train_split = ds[first_key]

    if "test" in ds:
        test_split = ds["test"]
    else:
        test_split = train_split

    train_split, valid_split = split_train_valid(train_split, seed=seed, valid_ratio=valid_ratio)

    train_rows = cap_rows(normalize_rows(train_split, task, f"{task}_train"), max_train)
    valid_rows = cap_rows(normalize_rows(valid_split, task, f"{task}_valid"), max_valid)
    test_rows = cap_rows(normalize_rows(test_split, task, f"{task}_test"), max_test)
    return train_rows, valid_rows, test_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["direct", "free_cot", "structured_cot"], required=True)
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--valid_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base_t5.yaml")
    parser.add_argument("--gsm8k_valid_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config.get("seed", 42))

    if args.task in {"gsm8k", "svamp", "aqua", "mathqa"}:
        train_rows, valid_rows, test_rows = load_benchmark_rows(
            task=args.task,
            seed=int(config.get("seed", 42)),
            valid_ratio=float(args.gsm8k_valid_ratio),
            max_train=int(args.max_train_samples),
            max_valid=int(args.max_valid_samples),
            max_test=int(args.max_test_samples),
        )
        print(f"{args.task} runtime load: train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)}")
    else:
        if not args.train_file or not args.valid_file:
            raise ValueError("For non-gsm8k tasks, --train_file and --valid_file are required.")
        train_rows = read_jsonl(Path(args.train_file))
        valid_rows = read_jsonl(Path(args.valid_file))
        test_rows = []

    train_ds = make_dataset(train_rows, args.method)
    valid_ds = make_dataset(valid_rows, args.method)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    max_in = int(config["max_input_length"])
    max_out = int(config["max_target_length"])

    def tokenize_fn(examples):
        model_inputs = tokenizer(examples["input_text"], truncation=True, max_length=max_in)
        labels = tokenizer(examples["target_text"], truncation=True, max_length=max_out)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    valid_tok = valid_ds.map(tokenize_fn, batched=True, remove_columns=valid_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def sanitize_token_ids(arr):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = np.argmax(arr, axis=-1)
        arr = arr.astype(np.int64, copy=False)
        # Replace invalid token ids that may appear in some generation/eval edge cases.
        arr = np.where(arr < 0, tokenizer.pad_token_id, arr)
        arr = np.where(arr >= tokenizer.vocab_size, tokenizer.pad_token_id, arr)
        return arr

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = sanitize_token_ids(preds)
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = sanitize_token_ids(np.where(labels != -100, labels, tokenizer.pad_token_id))
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        correct = 0
        for p, g in zip(pred_texts, label_texts):
            if extract_answer(p) == extract_answer(g):
                correct += 1
        return {"em": correct / len(pred_texts) if pred_texts else 0.0}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        weight_decay=float(config["weight_decay"]),
        num_train_epochs=float(config["num_train_epochs"]),
        warmup_ratio=float(config["warmup_ratio"]),
        predict_with_generate=True,
        generation_max_length=max_out,
        generation_num_beams=1,
        logging_steps=20,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        disable_tqdm=False,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(max_length=max_out)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.output_dir) / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if args.task in {"gsm8k", "gsm8k_teacher"} and test_rows:
        test_ds = make_dataset(test_rows, args.method)
        test_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=test_ds.column_names)
        test_metrics = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test", max_length=max_out)
        with (Path(args.output_dir) / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
