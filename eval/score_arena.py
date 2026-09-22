import argparse
import json
import random
from collections import Counter

WEIGHT = {"[[A>>B]]": 3, "[[A>B]]": 1, "[[A=B]]": 1, "[[B>A]]": 1, "[[B>>A]]": 3}
A_CREDIT = {"[[A>>B]]": 1.0, "[[A>B]]": 1.0, "[[A=B]]": 0.5,
            "[[B>A]]": 0.0, "[[B>>A]]": 0.0}

def credit(verdict, ours_is_a):
    a = A_CREDIT[verdict]
    return a if ours_is_a else 1.0 - a

def weighted_winrate(games):
    num = den = 0.0
    for g in games:
        v = g.get("verdict")
        if not v:
            continue
        w = WEIGHT[v]
        num += w * credit(v, g["ours_is_a"])
        den += w
    return (100.0 * num / den) if den else None

def bootstrap_ci(games, rounds=1000, seed=0):
    by_question = {}
    for g in games:
        if g.get("verdict"):
            by_question.setdefault(g["uid"], []).append(g)
    uids = list(by_question)
    if len(uids) < 5:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        sample = [g for u in uids for g in by_question[rng.choice(uids)]]
        v = weighted_winrate(sample)
        if v is not None:
            values.append(v)
    if not values:
        return None
    values.sort()
    return [round(values[int(0.05 * len(values))], 2),
            round(values[int(0.95 * len(values))], 2)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = json.load(open(args.judged))
    games = blob["games"]

    report = {"label": blob.get("summary", {}).get("label"),
              "judge_model": blob.get("summary", {}).get("judge_model"),
              "weighting": "significantly-better verdicts count 3x, slight wins 1x, ties 0.5",
              "categories": {}}

    for category in ("hard_prompt", "creative_writing"):
        subset = [g for g in games if g["category"] == category]
        scored = [g for g in subset if g.get("verdict")]
        report["categories"][category] = {
            "n_games_attempted": len(subset),
            "n_games_scored": len(scored),
            "coverage_pct": round(100.0 * len(scored) / len(subset), 1) if subset else None,
            "weighted_winrate": round(weighted_winrate(subset), 2) if scored else None,
            "weighted_ci90": bootstrap_ci(subset),
            "verdict_distribution": dict(Counter(g["verdict"] for g in scored)),
        }

    overall = weighted_winrate(games)
    report["overall_weighted_winrate"] = round(overall, 2) if overall is not None else None

    for name, flag in (("ours_as_A", True), ("ours_as_B", False)):
        v = weighted_winrate([g for g in games if g["ours_is_a"] is flag])
        report[f"position_check_{name}"] = round(v, 2) if v is not None else None

    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
