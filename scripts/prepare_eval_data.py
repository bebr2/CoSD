import argparse
import os

import requests

ALPACA = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/"
ARENA = ("https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
         "data/arena-hard-v2.0/")

FILES = [
    (ALPACA + "alpaca_eval.json", "alpaca_eval.json"),
    (ALPACA + "alpaca_eval_gpt4_baseline.json", "alpaca_eval_gpt4_baseline.json"),
    (ALPACA + "instruction_difficulty.csv", "instruction_difficulty.csv"),
    (ARENA + "question.jsonl", "question.jsonl"),
    (ARENA + "model_answer/o3-mini-2025-01-31.jsonl", "o3-mini-2025-01-31.jsonl"),
    (ARENA + "model_answer/gemini-2.0-flash-001.jsonl", "gemini-2.0-flash-001.jsonl"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/eval_ref")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for url, name in FILES:
        path = os.path.join(args.out_dir, name)
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"{len(response.content):>9} bytes  {path}")

if __name__ == "__main__":
    main()
