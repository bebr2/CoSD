import argparse
import json

from datasets import load_dataset

MAX_TOTAL_CONV_CHARS = 100000
MAX_COMPLETION_CHARS = 4096
MAX_HISTORY_MESSAGES = 5

def truncate_history(history, max_messages):
    cleaned = [m for m in history
               if (m.get("value") or "").strip() and m.get("from") in {"human", "gpt"}]
    if not cleaned:
        return None

    truncated = cleaned[-max_messages:]
    while truncated and truncated[0]["from"] != "human":
        truncated = truncated[1:]
    if not truncated:
        return None

    alternating = [truncated[0]]
    for m in truncated[1:]:
        if m["from"] != alternating[-1]["from"]:
            alternating.append(m)
    return alternating if alternating[0]["from"] == "human" else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/interactions.jsonl")
    args = ap.parse_args()

    dataset = load_dataset("microsoft/WildFeedback", "wildfeedback", split="train")

    rows, kept = [], 0
    for conv_idx, row in enumerate(dataset):
        conversation = row.get("conversations") or row.get("conversation")
        if not conversation or len(conversation) < 3:
            continue

        total_chars = sum(len(m.get("value") or "") for m in conversation)
        if total_chars > MAX_TOTAL_CONV_CHARS:
            continue
        if any(m.get("from") == "gpt" and len(m.get("value") or "") > MAX_COMPLETION_CHARS
               for m in conversation):
            continue
        kept += 1

        for i in range(len(conversation) - 1):
            response, follow_up = conversation[i], conversation[i + 1]
            if response.get("from") != "gpt" or follow_up.get("from") != "human":
                continue
            prompt = truncate_history(conversation[:i], MAX_HISTORY_MESSAGES)
            if prompt is None:
                continue
            rows.append({
                "id": f"{conv_idx}_{i}",
                "original_conv_id": conv_idx,
                "turn_id": i,
                "prompt": prompt,
                "completion": response,
                "user_response": follow_up,
            })

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"conversations kept {kept}  interaction rows {len(rows)}  -> {args.out}")

if __name__ == "__main__":
    main()
