import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def split_rows(rows, seed=42):
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    return rows[:n_train], rows[n_train : n_train + n_valid], rows[n_train + n_valid :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    tasks = ["coin_flip", "llc", "gsm_symbolic"]
    for task in tasks:
        task_file = inp / f"{task}_all.jsonl"
        if not task_file.exists():
            continue
        rows = read_jsonl(task_file)
        train, valid, test = split_rows(rows, seed=args.seed)
        write_jsonl(out / task / "train.jsonl", train)
        write_jsonl(out / task / "valid_iid.jsonl", valid)
        write_jsonl(out / task / "test_ood.jsonl", test)
        print(f"{task}: train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
