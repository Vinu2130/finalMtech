import argparse
import json
from pathlib import Path


def safe_load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_float(x):
    if x is None:
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:  # noqa: BLE001
        return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--prefix", type=str, default="gsm8k_teacher_")
    parser.add_argument("--out_csv", type=str, default="outputs/gsm8k_teacher_results.csv")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    runs = sorted([p for p in outputs_dir.glob(f"{args.prefix}*") if p.is_dir()])

    rows = []
    for run in runs:
        eval_metrics = safe_load_json(run / "eval_metrics.json")
        test_metrics = safe_load_json(run / "test_metrics.json")
        rows.append(
            {
                "run": run.name,
                "eval_em": eval_metrics.get("eval_em"),
                "eval_loss": eval_metrics.get("eval_loss"),
                "test_em": test_metrics.get("test_em"),
                "test_loss": test_metrics.get("test_loss"),
                "train_runtime_sec": eval_metrics.get("train_runtime"),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = ["run", "eval_em", "eval_loss", "test_em", "test_loss", "train_runtime_sec"]
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(
                ",".join(
                    [
                        str(row["run"]),
                        fmt_float(row["eval_em"]),
                        fmt_float(row["eval_loss"]),
                        fmt_float(row["test_em"]),
                        fmt_float(row["test_loss"]),
                        fmt_float(row["train_runtime_sec"]),
                    ]
                )
                + "\n"
            )

    print(f"Wrote {len(rows)} rows to {out_csv}")
    for row in rows:
        print(
            f"{row['run']}: eval_em={fmt_float(row['eval_em'])}, "
            f"test_em={fmt_float(row['test_em'])}, eval_loss={fmt_float(row['eval_loss'])}"
        )


if __name__ == "__main__":
    main()
