import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_gsm8k_answer(raw_answer: str):
    text = str(raw_answer).strip()
    m = re.search(r"####\s*([^\n]+)", text)
    final = m.group(1).strip() if m else text.splitlines()[-1].strip()
    rationale = text[: m.start()].strip() if m else text.strip()
    return rationale, final


def cot_to_structured(cot: str, answer: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(cot)).strip()
    if not cleaned:
        return f"Answer: {answer}"

    # Build compact stepwise format from short-CoT sentences.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        sentences = [cleaned]

    step_lines = [f"Step{i + 1}: {sent}" for i, sent in enumerate(sentences)]
    step_lines.append(f"Answer: {answer}")
    return "\n".join(step_lines)


def convert_teacher_rows(raw_rows):
    rows = []
    for i, ex in enumerate(raw_rows):
        question = str(ex.get("question", "")).strip()
        cot = str(ex.get("cot", "")).strip()
        final_answer = str(ex.get("final_answer", "")).strip()
        if not question or not final_answer:
            continue

        rows.append(
            {
                "id": f"gsm8k_teacher_train_{i:06d}",
                "task": "gsm8k_teacher",
                "question": question,
                "answer": final_answer,
                "rationale_free": cot,
                "rationale_structured": cot_to_structured(cot, final_answer),
                "meta": {"source": "gsm8k_cot_120b"},
            }
        )
    return rows


def build_official_test_rows():
    ds = load_dataset("gsm8k", "main")
    rows = []
    for i, ex in enumerate(ds["test"]):
        rationale_free, answer = parse_gsm8k_answer(ex["answer"])
        rows.append(
            {
                "id": f"gsm8k_official_test_{i:06d}",
                "task": "gsm8k_teacher",
                "question": str(ex["question"]).strip(),
                "answer": str(answer).strip(),
                "rationale_free": str(rationale_free).strip(),
                "rationale_structured": cot_to_structured(rationale_free, answer),
                "meta": {"source": "gsm8k_main_test"},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_jsonl", type=str, required=True, help="Path to gsm8k_teacher_cot.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output dir, e.g. data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir)

    raw_rows = read_jsonl(Path(args.teacher_jsonl))
    teacher_rows = convert_teacher_rows(raw_rows)
    rng.shuffle(teacher_rows)

    n_valid = max(1, int(len(teacher_rows) * args.valid_ratio))
    valid_rows = teacher_rows[:n_valid]
    train_rows = teacher_rows[n_valid:]

    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_valid_samples > 0:
        valid_rows = valid_rows[: args.max_valid_samples]

    test_rows = build_official_test_rows()
    if args.max_test_samples > 0:
        test_rows = test_rows[: args.max_test_samples]

    # Save processed full set and model-ready splits.
    write_jsonl(out / "processed" / "gsm8k_teacher_all.jsonl", teacher_rows)
    split_dir = out / "splits" / "gsm8k_teacher"
    write_jsonl(split_dir / "train.jsonl", train_rows)
    write_jsonl(split_dir / "valid_iid.jsonl", valid_rows)
    write_jsonl(split_dir / "test_ood.jsonl", test_rows)

    print(
        "Prepared gsm8k_teacher dataset: "
        f"train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)} "
        f"(teacher_total={len(teacher_rows)})"
    )


if __name__ == "__main__":
    main()
