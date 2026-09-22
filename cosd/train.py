import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from .sft import SFTCollator, SFTTrainer
from .trainer import MAX_COMPLETION_LEN, CoSDCollator, CoSDTrainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["cosd", "sdpo", "sft"], required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", default="checkpoints/run")
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--reference_count", type=int, default=3,
                   help="k, the number of reference messages per sample; ignored for "
                        "sdpo and sft")
    p.add_argument("--no_history", action="store_true",
                   help="drop the cross-session user history; the ablation of Table 5")
    p.add_argument("--score_chunk_size", type=int, default=4,
                   help="reference branches scored per forward pass")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--save_strategy", default="no", choices=["no", "steps"])
    p.add_argument("--save_steps", type=int, default=200)
    return p.parse_args()

def describe_dataset(ds, k, use_history):
    n = len(ds)
    n_ref = sum(1 for x in ds["references"]
                if x and any(float(r.get("cosine", -1.0)) >= 0.0 for r in x))
    n_hist = sum(1 for x in ds["user_history"] if (x or "").strip()) if use_history else 0
    print(f"rows {n}  with a reference message {n_ref} ({100.0 * n_ref / n:.1f}%)  "
          f"with user history {n_hist} ({100.0 * n_hist / n:.1f}%)  k={k}", flush=True)
    return n_ref

def main():
    args = parse_args()
    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    k = args.reference_count if args.method == "cosd" else 0
    use_history = args.method == "cosd" and not args.no_history

    if args.method in ("cosd", "sdpo"):
        assert "references" in ds.column_names, (
            "the training file has no `references` column; build it with "
            "scripts/build_dataset.py")
        n_ref = describe_dataset(ds, k, use_history)
        if args.method == "cosd":
            assert len(ds[0]["references"]) == k, (
                f"`references` holds {len(ds[0]['references'])} slots per row but "
                f"--reference_count is {k}")
            assert n_ref > 0, "no row carries a reference message; CoSD would reduce to SDPO"
            if use_history:
                assert any((x or "").strip() for x in ds["user_history"]), (
                    "`user_history` is empty on every row; run scripts/join_user_ids.py "
                    "or pass --no_history")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation)
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to=[],
        warmup_ratio=0.05,
        max_grad_norm=10.0,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    if args.method == "sft":
        trainer = SFTTrainer(
            model=model, args=training_args, train_dataset=ds,
            processing_class=tokenizer,
            data_collator=SFTCollator(tokenizer=tokenizer,
                                      max_completion_length=MAX_COMPLETION_LEN))
    else:
        trainer = CoSDTrainer(
            model=model, args=training_args, train_dataset=ds,
            processing_class=tokenizer,
            score_chunk_size=args.score_chunk_size,
            data_collator=CoSDCollator(tokenizer=tokenizer, reference_count=k,
                                       use_history=use_history,
                                       max_completion_length=MAX_COMPLETION_LEN))

    trainer.train()
    final = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"saved {final}", flush=True)

if __name__ == "__main__":
    main()
