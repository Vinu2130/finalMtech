import argparse
import json
import random
import string
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_coin_flip(n: int):
    rows = []
    for i in range(n):
        flips = random.randint(1, 15)
        start = random.choice(["heads", "tails"])
        answer = start if flips % 2 == 0 else ("tails" if start == "heads" else "heads")
        q = f"A coin starts {start}. It is flipped {flips} times. What is the final side?"
        rf = f"If flips are odd, side changes. {flips} is {'odd' if flips % 2 else 'even'}."
        rs = (
            f"Step1: Start side is {start}.\n"
            f"Step2: {flips} is {'odd' if flips % 2 else 'even'}, so side is "
            f"{'changed' if flips % 2 else 'unchanged'}.\n"
            f"Answer: {answer}"
        )
        rows.append(
            {
                "id": f"coin_flip_{i:06d}",
                "task": "coin_flip",
                "question": q,
                "answer": answer,
                "rationale_free": rf,
                "rationale_structured": rs,
                "meta": {"flips": flips, "parity": "odd" if flips % 2 else "even"},
            }
        )
    return rows


def random_word(min_len=3, max_len=8):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_lowercase, k=length))


def make_llc(n: int):
    rows = []
    for i in range(n):
        count = random.randint(2, 6)
        words = [random_word() for _ in range(count)]
        answer = "".join(w[-1] for w in words)
        q = f"Take the last letter of each word and concatenate: {', '.join(words)}."
        rf = f"Last letters are {' '.join([w[-1] for w in words])}."
        rs = (
            f"Step1: Words are {', '.join(words)}.\n"
            f"Step2: Last letters are {', '.join([w[-1] for w in words])}.\n"
            f"Answer: {answer}"
        )
        rows.append(
            {
                "id": f"llc_{i:06d}",
                "task": "llc",
                "question": q,
                "answer": answer,
                "rationale_free": rf,
                "rationale_structured": rs,
                "meta": {"word_count": count, "max_word_len": max(len(w) for w in words)},
            }
        )
    return rows


def ingest_gsm_symbolic(path: Path, limit: int = 0):
    raw_rows = read_jsonl(path)
    rows = []
    for i, row in enumerate(raw_rows):
        question = row.get("question") or row.get("input") or row.get("prompt")
        answer = row.get("answer") or row.get("target") or row.get("output")
        if question is None or answer is None:
            continue
        answer = str(answer).strip()
        rationale_free = row.get("rationale_free") or row.get("rationale") or ""
        rationale_structured = row.get("rationale_structured")
        if not rationale_structured:
            if rationale_free:
                rationale_structured = f"Step1: {rationale_free}\nAnswer: {answer}"
            else:
                rationale_structured = f"Step1: Solve the problem carefully.\nAnswer: {answer}"
        rows.append(
            {
                "id": f"gsm_symbolic_{i:06d}",
                "task": "gsm_symbolic",
                "question": str(question).strip(),
                "answer": answer,
                "rationale_free": str(rationale_free).strip(),
                "rationale_structured": str(rationale_structured).strip(),
                "meta": {"source_file": str(path.name)},
            }
        )
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--coin_n", type=int, default=8000)
    parser.add_argument("--llc_n", type=int, default=8000)
    parser.add_argument("--gsm_symbolic_file", type=str, default="")
    parser.add_argument("--gsm_symbolic_n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    ensure_dir(out)

    coin_rows = make_coin_flip(args.coin_n)
    llc_rows = make_llc(args.llc_n)

    write_jsonl(out / "coin_flip_all.jsonl", coin_rows)
    write_jsonl(out / "llc_all.jsonl", llc_rows)
    print(f"Wrote {len(coin_rows)} coin_flip rows and {len(llc_rows)} llc rows to {out}")

    if args.gsm_symbolic_file:
        gsm_path = Path(args.gsm_symbolic_file)
        gsm_rows = ingest_gsm_symbolic(gsm_path, limit=args.gsm_symbolic_n)
        write_jsonl(out / "gsm_symbolic_all.jsonl", gsm_rows)
        print(f"Wrote {len(gsm_rows)} gsm_symbolic rows from {gsm_path} to {out}")


if __name__ == "__main__":
    main()
