import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_api

SYSTEM = (
    "You are a highly efficient assistant, who evaluates and selects the best "
    "large language model (LLMs) based on the quality of their responses to a "
    "given instruction. This process will be used to create a leaderboard "
    "reflecting the most accurate and human-preferred answers."
)

TEMPLATE = """I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": \"\"\"{instruction}\"\"\",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "m",
        "output": \"\"\"{out_m}\"\"\"
    }},
    {{
        "model_identifier": "M",
        "output": \"\"\"{out_M}\"\"\"
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier
"""

_lock = threading.Lock()
_stats = {"calls": 0, "http_429": 0, "unparsed": 0, "errors": 0}
_limiter = judge_api.RateLimiter(55)
_cfg = None

def annotate(instruction, out_m, out_M, retries=8):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TEMPLATE.format(
            instruction=instruction, out_m=out_m, out_M=out_M)},
    ]
    for attempt in range(retries):
        _limiter.acquire()
        status, text, _ = judge_api.call(_cfg, messages, 2048, timeout=300)
        if status == 429:
            with _lock:
                _stats["http_429"] += 1
            time.sleep(min(60, 5 * (2 ** attempt)) * (0.75 + 0.5 * random.random()))
            continue
        if status != 200:
            if attempt == retries - 1:
                with _lock:
                    _stats["errors"] += 1
                return None
            time.sleep(2 * (attempt + 1))
            continue
        with _lock:
            _stats["calls"] += 1
        if not text:
            if attempt == retries - 1:
                with _lock:
                    _stats["unparsed"] += 1
                return None
            time.sleep(2 * (attempt + 1))
            continue
        if text in ("m", "M"):
            return text
        stripped = text.replace("*", "").replace("`", "").strip()
        for token in ("M", "m"):
            if stripped.endswith(token):
                return token
        with _lock:
            _stats["unparsed"] += 1
        return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_dir", default="data/eval_ref")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=55)
    args = ap.parse_args()

    global _limiter, _cfg
    _limiter = judge_api.RateLimiter(args.rpm)
    _cfg = judge_api.config()
    status, text = judge_api.probe(_cfg)
    assert status == 200 and text, f"judge probe failed: status={status} text={text!r}"
    print(f"[alpaca] judge {_cfg['model']} probe OK", flush=True)

    baseline = json.load(open(os.path.join(args.ref_dir, "alpaca_eval_gpt4_baseline.json")))
    ours = {d["instruction"]: d["output"] for d in json.load(open(args.answers))}
    items = [b for b in baseline if b["instruction"] in ours]

    games = [(i, ours_is_m) for i in range(len(items)) for ours_is_m in (True, False)]
    print(f"[alpaca] {args.label}: {len(items)} instructions, {len(games)} games", flush=True)

    t0 = time.time()
    done = [0]

    def work(game):
        idx, ours_is_m = game
        item = items[idx]
        mine, theirs = ours[item["instruction"]], item["output"]
        out_m, out_M = (mine, theirs) if ours_is_m else (theirs, mine)
        pick = annotate(item["instruction"], out_m, out_M)
        win = None if pick is None else float((pick == "m") == ours_is_m)
        with _lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  [{done[0]}/{len(games)}] {time.time() - t0:.0f}s "
                      f"429s={_stats['http_429']} unparsed={_stats['unparsed']}", flush=True)
        return {"instruction": item["instruction"], "dataset": item.get("dataset"),
                "ours_is_m": ours_is_m, "pick": pick, "ours_win": win,
                "ours_len": len(mine), "baseline_len": len(theirs)}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(work, games))

    scored = [r["ours_win"] for r in results if r["ours_win"] is not None]
    by_instruction = {}
    for r in results:
        if r["ours_win"] is not None:
            by_instruction.setdefault(r["instruction"], []).append(r["ours_win"])
    per_instruction = {k: sum(v) / len(v) for k, v in by_instruction.items()}

    position = {}
    for flag, name in ((True, "ours_as_m"), (False, "ours_as_M")):
        w = [r["ours_win"] for r in results
             if r["ours_is_m"] is flag and r["ours_win"] is not None]
        position[name] = round(100.0 * sum(w) / len(w), 2) if w else None

    agreed = sum(1 for v in per_instruction.values() if v in (0.0, 1.0))
    summary = {
        "label": args.label,
        "answers_file": args.answers,
        "judge_model": _cfg["model"],
        "metric": "raw winrate over position-swapped games; not length-controlled",
        "raw_winrate": round(100.0 * sum(scored) / len(scored), 2) if scored else None,
        "balanced_winrate_per_instruction":
            round(100.0 * statistics.mean(per_instruction.values()), 2) if per_instruction else None,
        "n_instructions": len(items),
        "n_games_scored": len(scored),
        "n_games_attempted": len(games),
        "both_orders_agreed": agreed,
        "both_orders_disagreed": len(per_instruction) - agreed,
        "position_bias": position,
        "mean_len_ours": round(statistics.mean([r["ours_len"] for r in results]), 1),
        "mean_len_baseline": round(statistics.mean([r["baseline_len"] for r in results]), 1),
        "failures": dict(_stats),
        "wall_seconds": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"summary": summary, "games": results}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
