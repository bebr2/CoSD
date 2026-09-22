import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_api

JUDGE_SYSTEM = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information than giving an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]."."""

JUDGE_USER = """<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""

VERDICTS = ["[[A>>B]]", "[[A>B]]", "[[A=B]]", "[[B>A]]", "[[B>>A]]"]
A_CREDIT = {"[[A>>B]]": 1.0, "[[A>B]]": 1.0, "[[A=B]]": 0.5,
            "[[B>A]]": 0.0, "[[B>>A]]": 0.0}

BASELINES = {"hard_prompt": "o3-mini-2025-01-31.jsonl",
             "creative_writing": "gemini-2.0-flash-001.jsonl"}

_lock = threading.Lock()
_stats = {"calls": 0, "http_429": 0, "no_verdict": 0, "errors": 0}
_limiter = judge_api.RateLimiter(55)
_cfg = None

def load_answers(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        content = d["messages"][1]["content"]
        if isinstance(content, dict):
            content = content.get("content") or content.get("answer") or ""
        out[d["uid"]] = content
    return out

def judge_once(question, answer_a, answer_b, retries=8):
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(
            question=question, answer_a=answer_a, answer_b=answer_b)},
    ]
    for attempt in range(retries):
        _limiter.acquire()
        status, content, _ = judge_api.call(_cfg, messages, 16000, timeout=600)
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
        if not content:
            if attempt == retries - 1:
                with _lock:
                    _stats["no_verdict"] += 1
                return None
            time.sleep(2 * (attempt + 1))
            continue
        found = [v for v in VERDICTS if v in content]
        if not found:
            with _lock:
                _stats["no_verdict"] += 1
            return None
        return found[-1]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_dir", default="data/eval_ref")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--rpm", type=int, default=55)
    args = ap.parse_args()

    global _limiter, _cfg
    _limiter = judge_api.RateLimiter(args.rpm)
    _cfg = judge_api.config()
    status, text = judge_api.probe(_cfg)
    assert status == 200 and text, f"judge probe failed: status={status} text={text!r}"
    print(f"[arena] judge {_cfg['model']} probe OK", flush=True)

    questions = {}
    for line in open(os.path.join(args.ref_dir, "question.jsonl")):
        d = json.loads(line)
        questions[d["uid"]] = d
    ours = load_answers(args.answers)
    baselines = {cat: load_answers(os.path.join(args.ref_dir, name))
                 for cat, name in BASELINES.items()}

    games = []
    for uid in sorted(questions):
        category = questions[uid]["category"]
        base = baselines.get(category)
        if base is None or uid not in ours or uid not in base:
            continue
        for ours_is_a in (False, True):
            games.append((uid, category, ours_is_a))
    print(f"[arena] {args.label}: {len(games)} games "
          f"({len(games) // 2} questions x 2 swaps)", flush=True)

    t0 = time.time()
    done = [0]

    def work(i):
        uid, category, ours_is_a = games[i]
        mine, theirs = ours[uid], baselines[category][uid]
        a, b = (mine, theirs) if ours_is_a else (theirs, mine)
        verdict = judge_once(questions[uid]["prompt"], a, b)
        score = None
        if verdict is not None:
            credit = A_CREDIT[verdict]
            score = credit if ours_is_a else 1.0 - credit
        with _lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  [{done[0]}/{len(games)}] {time.time() - t0:.0f}s "
                      f"429s={_stats['http_429']} no_verdict={_stats['no_verdict']}",
                      flush=True)
        return {"uid": uid, "category": category, "ours_is_a": ours_is_a,
                "verdict": verdict, "score_for_ours": score}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(work, range(len(games))))

    scored = [r["score_for_ours"] for r in results if r["score_for_ours"] is not None]
    summary = {
        "label": args.label,
        "answers_file": args.answers,
        "judge_model": _cfg["model"],
        "scoring": "unweighted game winrate; scored with the official weighting by score_arena.py",
        "overall_winrate": round(100.0 * sum(scored) / len(scored), 2) if scored else None,
        "n_games_scored": len(scored),
        "n_games_attempted": len(games),
        "failures": dict(_stats),
        "wall_seconds": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"summary": summary, "games": results}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
