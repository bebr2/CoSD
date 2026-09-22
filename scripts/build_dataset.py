import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

WHITESPACE = re.compile(r"\s+")

CANDIDATE_WINDOW = 64
HISTORY_MAX_SESSIONS = 5
HISTORY_MAX_TOKENS = 100
SENTINEL = {"ref_id": "", "cosine": -1.0, "o": "", "y": ""}

def norm(s):
    return WHITESPACE.sub(" ", (s or "").strip()).lower()

def token_len(s):
    return max(1, len((s or "").split()))

def pad(items, k):
    out = [dict(x) for x in items[:k]]
    while len(out) < k:
        out.append(dict(SENTINEL))
    return out

def build_references(rows, k, seed):
    follow_ups = [((r.get("user_response") or {}).get("value") or "").strip() for r in rows]
    responses = [((r.get("completion") or {}).get("value") or "").strip() for r in rows]

    by_session = defaultdict(set)
    for r, o in zip(rows, follow_ups):
        if o:
            by_session[r["session_id"]].add(norm(o))

    pool = [i for i, o in enumerate(follow_ups) if o]
    rng = random.Random(seed)
    rejected = Counter()
    picks_per_row, length_ratios = [], []

    for i, r in enumerate(rows):
        target = token_len(follow_ups[i])
        own = norm(follow_ups[i])
        session = r["session_id"]

        candidates, taken = [], set()
        for _ in range(CANDIDATE_WINDOW):
            j = pool[rng.randrange(len(pool))]
            if j == i or rows[j]["id"] == r["id"]:
                rejected["self"] += 1
                continue
            text = norm(follow_ups[j])
            if text == own:
                rejected["identical_to_own_follow_up"] += 1
                continue
            if text in by_session.get(session, ()):
                rejected["same_session"] += 1
                continue
            if text in taken:
                rejected["duplicate_slot"] += 1
                continue
            taken.add(text)
            candidates.append((abs(math.log(token_len(follow_ups[j]) / target)), j))

        candidates.sort(key=lambda t: (t[0], rows[t[1]]["id"]))
        picks = []
        for _, j in candidates[:k]:
            picks.append({"ref_id": rows[j]["id"], "cosine": 1.0,
                          "o": follow_ups[j], "y": responses[j]})
            length_ratios.append(abs(math.log(token_len(follow_ups[j]) / target)))
        picks_per_row.append(pad(picks, k))

    full = sum(1 for p in picks_per_row
               if sum(1 for s in p if s["cosine"] >= 0.0) == k)
    beyond_2x = (sum(1 for x in length_ratios if x > math.log(2)) / max(len(length_ratios), 1))
    print(f"references: {full}/{len(rows)} rows have all {k} slots filled; "
          f"{100.0 * beyond_2x:.2f}% of pairs differ by more than 2x in length", flush=True)
    print(f"rejected during the draw {dict(rejected)}", flush=True)
    return picks_per_row

def build_history(rows, tokenizer):
    by_user = defaultdict(list)
    for r in rows:
        o = ((r.get("user_response") or {}).get("value") or "").strip()
        if o and r.get("user_id"):
            by_user[r["user_id"]].append((r["session_id"], o, r["id"]))

    out, n_with, n_leaked = [], 0, 0
    for r in rows:
        own_session = r["session_id"]
        own_follow_up = ((r.get("user_response") or {}).get("value") or "").strip()

        others = [(s, o) for s, o, rid in by_user.get(r.get("user_id") or "", [])
                  if s != own_session and rid != r["id"]]
        others.sort()

        seen, picked = set(), []
        for session, o in others:
            if session in seen:
                continue
            seen.add(session)
            picked.append(o)
            if len(picked) >= HISTORY_MAX_SESSIONS:
                break

        assert own_session not in seen, f"row {r['id']}: own session entered its history"
        text = " ".join(picked)
        if not text:
            out.append("")
            continue

        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > HISTORY_MAX_TOKENS:
            text = tokenizer.decode(ids[:HISTORY_MAX_TOKENS])
        out.append(text)
        n_with += 1
        if own_follow_up and own_follow_up in text:
            n_leaked += 1

    print(f"user history: {n_with}/{len(rows)} rows ({100.0 * n_with / len(rows):.1f}%); "
          f"{n_leaked} contain the row's own follow-up verbatim", flush=True)
    return out

def check(path, k):
    keysets, shapes = set(), set()
    n_self = n_identical = n_same_session = n_duplicate = 0
    n = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            n += 1
            keysets.add(tuple(sorted(d)))
            refs = d["references"]
            assert isinstance(refs, list) and len(refs) == k, (
                f"row {d['id']} holds {len(refs)} reference slots, expected {k}")
            own = norm((d.get("user_response") or {}).get("value"))
            real = []
            for s in refs:
                shapes.add(tuple(sorted(s)))
                assert isinstance(s["cosine"], float), f"row {d['id']}: cosine is not a float"
                if s["cosine"] < 0.0:
                    continue
                real.append(norm(s["o"]))
                if s["ref_id"] == d["id"]:
                    n_self += 1
                if norm(s["o"]) == own:
                    n_identical += 1
                if s["ref_id"].rsplit("_", 1)[0] == str(d["original_conv_id"]):
                    n_same_session += 1
            if len(real) != len(set(real)):
                n_duplicate += 1

    assert len(keysets) == 1, f"{len(keysets)} distinct key sets in {path}"
    assert len(shapes) == 1, f"inconsistent reference struct shape: {shapes}"
    assert n_self == 0, f"{n_self} references are the row itself"
    assert n_identical == 0, (
        f"{n_identical} references are identical to the row's own follow-up, which makes "
        f"its advantage exactly zero")
    assert n_same_session == 0, f"{n_same_session} references come from the row's own session"
    assert n_duplicate == 0, f"{n_duplicate} rows repeat a reference, so the effective k is below {k}"
    print(f"checked {n} rows: one key set, {k} slots per row, no self / identical / "
          f"same-session / duplicate references", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/interactions.with_user.jsonl")
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--tokenizer", required=True,
                    help="model path whose tokenizer caps the user history")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1214)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.src)]
    assert rows and "user_id" in rows[0], (
        "the source file has no `user_id`; run scripts/join_user_ids.py first")
    print(f"{len(rows)} source rows", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    references = build_references(rows, args.k, args.seed)
    histories = build_history(rows, tokenizer)

    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for r, refs, hist in zip(rows, references, histories):
            d = dict(r)
            d["references"] = refs
            d["user_history"] = hist
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out)

    check(args.out, args.k)
    print(f"wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
