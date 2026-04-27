import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["direct", "free_cot", "structured_cot"], required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base_t5.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config.get("seed", 42))

    train_rows = read_jsonl(Path(args.train_file))
    valid_rows = read_jsonl(Path(args.valid_file))

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

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
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
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        weight_decay=float(config["weight_decay"]),
        num_train_epochs=float(config["num_train_epochs"]),
        warmup_ratio=float(config["warmup_ratio"]),
        predict_with_generate=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
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
    print(metrics)


if __name__ == "__main__":
    main()
