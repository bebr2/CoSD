import argparse
import json
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

_L1 = ({"l1_ratio": 1.0}
       if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
       else {"penalty": "l1"})

def build_frame(games):
    by_instruction = {}
    for g in games:
        if g.get("ours_win") is None:
            continue
        by_instruction.setdefault(g["instruction"], []).append(g)

    rows = []
    for instruction, group in by_instruction.items():
        mean_win = float(np.mean([g["ours_win"] for g in group]))
        rows.append({"instruction": instruction,
                     "preference": 1.0 + mean_win,
                     "ours_len": group[0]["ours_len"],
                     "baseline_len": group[0]["baseline_len"]})
    return pd.DataFrame(rows)

def design(frame, zero_len=False):
    length_term = np.tanh(0.0 * frame["std_delta_len"] if zero_len
                          else frame["std_delta_len"])
    return np.column_stack([
        length_term.to_numpy(dtype=float),
        frame["instruction_difficulty"].to_numpy(dtype=float),
        np.ones(len(frame)),
    ])

def fit_predict(frame):
    X = design(frame)
    p = frame["y"].to_numpy(dtype=float)
    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(X)), np.zeros(len(X))])
    w2 = np.concatenate([p, 1.0 - p])
    keep = w2 > 0

    model = LogisticRegression(solver="liblinear", C=100, fit_intercept=False,
                               random_state=123, **_L1)
    model.fit(X2[keep], y2[keep], sample_weight=w2[keep])
    predictions = model.predict_proba(design(frame, zero_len=True))[:, 1]
    coefficients = dict(zip(["tanh_std_delta_len", "instruction_difficulty", "intercept"],
                            model.coef_[0].tolist()))
    return predictions, coefficients

def lc_winrate(df, difficulty, seed=0, n_boot=1000):
    d = df.copy()
    delta = d["baseline_len"].astype(float) - d["ours_len"].astype(float)
    spread = delta.std()
    d["std_delta_len"] = delta / (spread if spread else 1.0)
    d["y"] = d["preference"].astype(float).replace({0.0: 1.5}) - 1.0
    d["instruction_difficulty"] = d["instruction"].map(difficulty)
    d = d.dropna(subset=["instruction_difficulty"])

    predictions, coefficients = fit_predict(d)
    lc = 100.0 * float(np.mean(predictions))
    raw = 100.0 * float(d["y"].mean())

    rng = np.random.default_rng(seed)
    index = np.arange(len(d))
    boots = []
    for _ in range(n_boot):
        sample = d.iloc[rng.choice(index, size=len(index), replace=True)].reset_index(drop=True)
        if sample["y"].nunique() < 2:
            continue
        boots.append(100.0 * float(np.mean(fit_predict(sample)[0])))
    boots.sort()
    ci = ([round(boots[int(0.025 * len(boots))], 2),
           round(boots[int(0.975 * len(boots))], 2)] if len(boots) > 50 else None)

    return {
        "length_controlled_winrate": round(lc, 2),
        "raw_winrate": round(raw, 2),
        "lc_minus_raw": round(lc - raw, 2),
        "lc_bootstrap_ci95": ci,
        "glm_coefficients": {k: round(v, 4) for k, v in coefficients.items()},
        "n_instructions": int(len(d)),
        "mean_len_ours": round(float(d["ours_len"].mean()), 1),
        "mean_len_baseline": round(float(d["baseline_len"].mean()), 1),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_dir", default="data/eval_ref")
    args = ap.parse_args()

    blob = json.load(open(args.judged))
    df = build_frame(blob["games"])

    difficulty_by_position = (
        pd.read_csv(os.path.join(args.ref_dir, "instruction_difficulty.csv"))
        .set_index("index")["instruction_difficulty"].to_dict())
    reference = json.load(open(os.path.join(args.ref_dir, "alpaca_eval_gpt4_baseline.json")))
    difficulty = {r["instruction"]: difficulty_by_position[i]
                  for i, r in enumerate(reference) if i in difficulty_by_position}

    result = lc_winrate(df, difficulty)
    result["label"] = blob["summary"].get("label")
    result["judge_model"] = blob["summary"].get("judge_model")
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
