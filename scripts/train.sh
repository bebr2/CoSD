set -euo pipefail

METHOD="${METHOD:-cosd}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL required}"
TRAIN_JSONL="${TRAIN_JSONL:-data/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/$METHOD}"

WORLD_SIZE="${WORLD_SIZE:-4}"
BS="${BS:-4}"
GA="${GA:-8}"
LR="${LR:-2e-6}"
EPOCHS="${EPOCHS:-2}"
SEED="${SEED:-42}"
K="${K:-3}"

EFFECTIVE=$((BS * GA * WORLD_SIZE))
[ "$EFFECTIVE" = "128" ] || { echo "effective batch $EFFECTIVE != 128" >&2; exit 1; }
[ -f "$TRAIN_JSONL" ] || { echo "no training file at $TRAIN_JSONL" >&2; exit 1; }

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

echo "method=$METHOD  model=$BASE_MODEL  data=$TRAIN_JSONL"
echo "LR=$LR  BS $BS x GA $GA x world $WORLD_SIZE = $EFFECTIVE  epochs=$EPOCHS  seed=$SEED  k=$K"

python -m accelerate.commands.launch \
  --num_processes "$WORLD_SIZE" --mixed_precision bf16 --dynamo_backend no \
  -m cosd.train \
  --method "$METHOD" \
  --base_model "$BASE_MODEL" \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --learning_rate "$LR" \
  --batch_size "$BS" \
  --grad_accum "$GA" \
  --num_epochs "$EPOCHS" \
  --reference_count "$K" \
  --seed "$SEED"
