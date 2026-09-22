import argparse
import json
import os
import time
import uuid

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model_name", required=True, help="label written into the outputs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ref_dir", default="data/eval_ref")
    p.add_argument("--which", default="both", choices=["both", "alpaca", "arena"])
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tp", type=int, default=2, help="tensor parallel size")
    p.add_argument("--gpu_frac", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=16384)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def render(user_msg):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    jobs = []
    if args.which in ("both", "alpaca"):
        data = json.load(open(os.path.join(args.ref_dir, "alpaca_eval.json")))
        jobs.append(("alpaca", [render(d["instruction"]) for d in data], data))
    if args.which in ("both", "arena"):
        questions = [json.loads(l) for l in
                     open(os.path.join(args.ref_dir, "question.jsonl"))]
        jobs.append(("arena", [render(q["prompt"]) for q in questions], questions))

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_frac, max_model_len=args.max_model_len,
              dtype="bfloat16", trust_remote_code=True)

    vocab = tokenizer.get_vocab() or {}
    stop_strings = [s for s in ("<end_of_turn>", "<|end|>", "<|eot_id|>", "<|im_end|>",
                                "</s>") if s in vocab]
    stop_ids = sorted({i for i in
                       ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
                       + [vocab[s] for s in stop_strings]})
    print(f"[generate] stop strings {stop_strings} ids {stop_ids}", flush=True)

    sampling = SamplingParams(temperature=args.temperature, top_p=1.0,
                              max_tokens=args.max_new_tokens,
                              stop=stop_strings or None, stop_token_ids=stop_ids or None)

    for tag, prompts, meta in jobs:
        budget = args.max_model_len - args.max_new_tokens
        clipped, safe = 0, []
        for text in prompts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > budget:
                text = tokenizer.decode(ids[-budget:], skip_special_tokens=False)
                clipped += 1
            safe.append(text)

        t0 = time.time()
        outputs = llm.generate(safe, sampling)
        texts = [o.outputs[0].text.strip() for o in outputs]
        empty = sum(1 for t in texts if not t)
        truncated = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
        print(f"[generate] {tag}: {len(texts)} answers in {time.time() - t0:.0f}s  "
              f"empty={empty} truncated={truncated} prompts_clipped={clipped}", flush=True)

        if tag == "alpaca":
            path = os.path.join(args.out_dir, "alpaca_answers.json")
            json.dump([{"dataset": m["dataset"], "instruction": m["instruction"],
                        "output": t, "generator": args.model_name}
                       for m, t in zip(meta, texts)],
                      open(path, "w"), ensure_ascii=False, indent=1)
        else:
            path = os.path.join(args.out_dir, "arena_answers.jsonl")
            with open(path, "w") as f:
                for m, t in zip(meta, texts):
                    f.write(json.dumps({
                        "uid": m["uid"],
                        "ans_id": uuid.uuid4().hex[:22],
                        "model": args.model_name,
                        "messages": [{"role": "user", "content": m["prompt"]},
                                     {"role": "assistant", "content": t}],
                        "tstamp": time.time(),
                    }, ensure_ascii=False) + "\n")
        print(f"[generate] wrote {path}", flush=True)

        json.dump({"model": args.model_name, "model_path": args.model, "n": len(texts),
                   "empty": empty, "truncated": truncated, "prompts_clipped": clipped,
                   "temperature": args.temperature,
                   "max_new_tokens": args.max_new_tokens,
                   "max_model_len": args.max_model_len,
                   "mean_chars": sum(len(t) for t in texts) / max(len(texts), 1)},
                  open(os.path.join(args.out_dir, f"{tag}_genstats.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
