import argparse
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

def digest(s):
    return hashlib.md5(s.encode("utf-8")).digest()

def conversation_prefix(messages):
    u1 = a1 = u2 = ""
    stage = 0
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if stage == 0 and role == "user":
            u1, stage = content, 1
        elif stage == 1 and role == "assistant":
            a1, stage = content, 2
        elif stage == 2 and role == "user":
            u2 = content
            break
    return u1, a1, u2

def scan_shard(path):
    import pyarrow.parquet as pq

    out = []
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(
            batch_size=2000,
            columns=["conversation", "hashed_ip", "conversation_hash"]):
        convs = batch.column("conversation").to_pylist()
        users = batch.column("hashed_ip").to_pylist()
        sessions = batch.column("conversation_hash").to_pylist()
        for messages, user, session in zip(convs, users, sessions):
            u1, a1, u2 = conversation_prefix(messages)
            out.append((digest(u1 + "\x00" + a1),
                        digest(u1 + "\x00" + a1 + "\x00" + u2) if u2 else None,
                        user, session))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", default="data/interactions.jsonl")
    ap.add_argument("--wildchat_glob", required=True,
                    help="glob for the WildChat parquet shards")
    ap.add_argument("--out", default="data/interactions.with_user.jsonl")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    first_turn = {}
    n_rows = 0
    with open(args.interactions) as f:
        for line in f:
            d = json.loads(line)
            n_rows += 1
            if d["turn_id"] == 1:
                first_turn[d["original_conv_id"]] = (
                    d["prompt"][0]["value"],
                    (d.get("completion") or {}).get("value") or "",
                    (d.get("user_response") or {}).get("value") or "",
                )
    print(f"{n_rows} rows over {len(first_turn)} conversations", flush=True)

    shards = sorted(glob.glob(args.wildchat_glob))
    assert shards, f"no parquet shards match {args.wildchat_glob!r}"
    print(f"scanning {len(shards)} shards with {args.workers} workers", flush=True)

    by_key1, by_key2 = defaultdict(set), defaultdict(set)
    with Pool(args.workers) as pool:
        for i, records in enumerate(pool.imap_unordered(scan_shard, shards)):
            for k1, k2, user, session in records:
                by_key1[k1].add((user, session))
                if k2 is not None:
                    by_key2[k2].add((user, session))
            if (i + 1) % 10 == 0 or i + 1 == len(shards):
                print(f"  {i + 1}/{len(shards)} shards", flush=True)

    user_of, session_of = {}, {}
    tiers = Counter()
    for cid, (u1, a1, u2) in first_turn.items():
        hits = by_key1.get(digest(u1 + "\x00" + a1))
        if not hits:
            tiers["no_match"] += 1
            continue
        if len({h[0] for h in hits}) == 1:
            user_of[cid], session_of[cid] = next(iter(hits))
            tiers["key1_unique"] += 1
            continue
        hits2 = by_key2.get(digest(u1 + "\x00" + a1 + "\x00" + u2)) if u2 else None
        if hits2 and len({h[0] for h in hits2}) == 1:
            user_of[cid], session_of[cid] = next(iter(hits2))
            tiers["key2_resolved"] += 1
        else:
            tiers["ambiguous"] += 1

    written = matched = 0
    tmp = args.out + ".tmp"
    with open(args.interactions) as f, open(tmp, "w") as out:
        for line in f:
            d = json.loads(line)
            cid = d["original_conv_id"]
            if cid in user_of:
                d["user_id"] = user_of[cid]
                d["session_id"] = session_of[cid]
                matched += 1
            else:
                d["user_id"] = ""
                d["session_id"] = str(cid)
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
            written += 1
    os.replace(tmp, args.out)

    print(f"resolution tiers {dict(tiers)}", flush=True)
    print(f"conversation coverage {len(user_of) / len(first_turn):.4f}  "
          f"row coverage {matched / written:.4f}  -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
