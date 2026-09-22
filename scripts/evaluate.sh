set -euo pipefail

MODEL="${MODEL:?MODEL required}"
LABEL="${LABEL:?LABEL required}"
OUT="${OUT:-results/$LABEL}"
REF_DIR="${REF_DIR:-data/eval_ref}"
TP="${TP:-2}"
STAGE="${STAGE:-all}"
CONCURRENCY="${CONCURRENCY:-8}"
RPM="${RPM:-55}"

export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$OUT"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "generate" ]; then
  echo "=== generating answers ==="
  python -u eval/generate.py --model "$MODEL" --model_name "$LABEL" \
    --out_dir "$OUT" --ref_dir "$REF_DIR" --tp "$TP"

  echo "=== IFEval (0-shot) ==="
  python -u eval/run_lmeval.py --model vllm \
    --model_args "pretrained=$MODEL,tensor_parallel_size=$TP,gpu_memory_utilization=0.85,max_model_len=16384,dtype=bfloat16" \
    --tasks ifeval --batch_size auto --apply_chat_template \
    --output_path "$OUT/lmeval_ifeval" --log_samples

  echo "=== MMLU-Pro (5-shot chain of thought) ==="
  python -u eval/run_lmeval.py --model vllm \
    --model_args "pretrained=$MODEL,tensor_parallel_size=$TP,gpu_memory_utilization=0.85,max_model_len=16384,dtype=bfloat16" \
    --tasks mmlu_pro --num_fewshot 5 --batch_size auto --apply_chat_template \
    --gen_kwargs max_gen_toks=4096 \
    --output_path "$OUT/lmeval_mmlupro" --log_samples
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "judge" ]; then
  echo "=== Arena-Hard-v2 judging ==="
  python -u eval/judge_arena.py --answers "$OUT/arena_answers.jsonl" --label "$LABEL" \
    --out "$OUT/arena_judged.json" --ref_dir "$REF_DIR" \
    --concurrency "$CONCURRENCY" --rpm "$RPM"

  echo "=== AlpacaEval 2.0 annotation ==="
  python -u eval/judge_alpaca.py --answers "$OUT/alpaca_answers.json" --label "$LABEL" \
    --out "$OUT/alpaca_judged.json" --ref_dir "$REF_DIR" \
    --concurrency "$CONCURRENCY" --rpm "$RPM"

  echo "=== scoring ==="
  python eval/score_arena.py --judged "$OUT/arena_judged.json" \
    --out "$OUT/arena_scored.json"
  python eval/lc_winrate.py --judged "$OUT/alpaca_judged.json" \
    --out "$OUT/alpaca_lc.json" --ref_dir "$REF_DIR"
fi

echo "=== done: $LABEL -> $OUT ==="
