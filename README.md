# CoSD: Contrastive Self-Distillation

Code for the main real-log experiment: Qwen3-4B trained on WildFeedback with CoSD, and
the SDPO and SFT baselines it is compared against.

## Method

Self-distillation from user feedback scores a logged response `y` twice, with and
without the user's follow-up message `o` in the context, and uses the per-token
difference as an advantage:

```
A_t = log p(y_t | x, o) - log p(y_t | x)
```

Part of that difference does not come from the message: substituting an unrelated
message, or an empty one, still shifts the score, and the shift varies across responses.
CoSD replaces the baseline term with the average score under `k` reference messages
drawn from other interactions and matched to `|o|` in length:

```
A_t = log p(y_t | x, o) - (1/k) * sum_j log p(y_t | x, o_j)
```

Both terms now carry a follow-up message, so any message-independent contribution
cancels token by token. The loss is unchanged from SDPO:

```
L = - E[ (1/L) * sum_t sg[A_t] * log p_theta(y_t | x, y_<t) ]
```

Because the cancellation holds for anything shared across branches, the user's
cross-session history can be placed in all branches as well: it changes how the current
message is read without contributing a user-dependent offset to the update. Evaluation
never uses the history.

## Layout

```
cosd/templates.py   hindsight message formats, user-history block, chat rendering
cosd/trainer.py      the collator that builds the branches and the contrastive loss
cosd/sft.py          the SFT baseline
cosd/train.py        training entrypoint for cosd / sdpo / sft
scripts/             data preparation and the two driver scripts
eval/                answer generation, judging, scoring
```

## Setup

```bash
pip install -r requirements.txt
```

Training uses `transformers` with bf16, FlashAttention-2, gradient checkpointing and
8-bit AdamW. Evaluation generation uses vLLM, which pins its own torch build, so a
separate environment for the evaluation dependencies is convenient.

## Data

```bash
# 1. WildFeedback -> interaction rows (x, y, o)
python scripts/prepare_wildfeedback.py --out data/interactions.jsonl

# 2. recover a user identifier per conversation by matching back to WildChat
python scripts/join_user_ids.py \
    --interactions data/interactions.jsonl \
    --wildchat_glob '<wildchat dir>/train-*.parquet' \
    --out data/interactions.with_user.jsonl

# 3. add the k length-matched reference messages and the user history
python scripts/build_dataset.py \
    --src data/interactions.with_user.jsonl \
    --out data/train.jsonl \
    --tokenizer Qwen/Qwen3-4B
```

Step 2 needs a local copy of the WildChat parquet shards, which carry the anonymized
user identifier that WildFeedback drops. It is required only for the user history; with
`user_id` empty on every row the history column comes out empty and training still
runs, using the contrast alone.

The reference messages and the history are both built with leave-one-session-out: a
reference never comes from the row's own session and is never identical to the row's own
follow-up, and the history never contains the row's own session. `build_dataset.py`
asserts all of this against the file it wrote.

Evaluation reference files:

```bash
python scripts/prepare_eval_data.py --out_dir data/eval_ref
```

## Training

```bash
METHOD=cosd BASE_MODEL=Qwen/Qwen3-4B bash scripts/train.sh
METHOD=sdpo BASE_MODEL=Qwen/Qwen3-4B bash scripts/train.sh
METHOD=sft  BASE_MODEL=Qwen/Qwen3-4B bash scripts/train.sh
```

All three use the same recipe: effective batch 128, learning rate 2e-6, 2 epochs, seed
42, cosine schedule with 5% warmup. CoSD adds `k = 3` reference messages per sample.

`METHOD=sdpo` sets `k = 0`, which makes the advantage `log p(y|x,o) - log p(y|x)` and
drops the history, so it differs from CoSD in the advantage and nothing else.

Each CoSD sample costs `k + 1` scoring forward passes without gradients plus the one
forward-backward pass on the update branch, about 1.25x the training compute of SDPO.

## Evaluation

```bash
export JUDGE_API_KEY=...
export JUDGE_MODEL=gpt-5-mini

MODEL=checkpoints/cosd/final_model LABEL=cosd bash scripts/evaluate.sh
```

Five benchmarks:

| | source | output |
|---|---|---|
| AlpacaEval 2.0, length-controlled | 805 instructions, official baseline | `alpaca_lc.json` |
| Arena-Hard-v2, hard prompts | 500 questions vs o3-mini | `arena_scored.json` |
| Arena-Hard-v2, creative writing | 250 questions vs gemini-2.0-flash-001 | `arena_scored.json` |
| IFEval, prompt-level strict | lm-eval, 0-shot | `lmeval_ifeval/` |
| MMLU-Pro | lm-eval, 5-shot chain of thought | `lmeval_mmlupro/` |

Both judged benchmarks judge every question twice with the answers in swapped
positions, so the judge's position bias cancels; the per-slot winrates are reported
alongside the result so any residual bias is visible. Generation is greedy, non-thinking
mode, 4096 new tokens.

`JUDGE_BASE_URL` selects the endpoint (any OpenAI-compatible chat-completions service)
and defaults to the official OpenAI address. `JUDGE_BUDGET_KEY=max_tokens` suits
providers that do not accept `max_completion_tokens` or `reasoning_effort`.

## Hindsight format

`COSD_HINDSIGHT_FORMAT` selects the wrapper around the follow-up message: `format1`
(default, the one used in the main results), `format2` or `format3`. All three are the
same length in tokens and say the same thing in different words.
