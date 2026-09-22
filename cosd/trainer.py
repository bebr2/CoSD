import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, Trainer

from .templates import (build_hindsight_context, build_history_prefix, hindsight_format,
                        normalize_messages, render)

MAX_CONTEXT_LEN = 2048
MAX_COMPLETION_LEN = 2048
IGNORE_FIRST_K = 2

def real_references(row, k):
    return [r for r in (row.get("references") or [])
            if float(r.get("cosine", -1.0)) >= 0.0][:k]

@dataclass
class CoSDCollator:
    tokenizer: PreTrainedTokenizerBase
    reference_count: int = 3
    use_history: bool = True
    max_completion_length: int = MAX_COMPLETION_LEN

    def __post_init__(self):
        self._announced = False

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self._announced:
            print(f"[cosd] hindsight format = {hindsight_format()}  "
                  f"reference_count = {self.reference_count}  "
                  f"user_history = {self.use_history}", flush=True)
            self._announced = True

        prompt_texts, conditional_texts, completion_texts = [], [], []
        reference_texts, reference_meta = [], []

        for bi, ex in enumerate(examples):
            dialogue = normalize_messages(ex["prompt"])
            prefix = (build_history_prefix(ex.get("user_history"))
                      if self.use_history else [])

            follow_up = ex["user_response"]
            o_plus = (follow_up.get("value") or follow_up.get("content") or "").strip()

            prompt_texts.append(render(self.tokenizer, prefix + dialogue))
            conditional_texts.append(render(
                self.tokenizer, prefix + build_hindsight_context(dialogue, o_plus)))

            refs = real_references(ex, self.reference_count)
            if not refs:
                reference_meta.append(None)
            else:
                idxs = []
                for r in refs:
                    ctx = build_hindsight_context(dialogue, (r.get("o") or "").strip())
                    reference_texts.append(render(self.tokenizer, prefix + ctx))
                    idxs.append(len(reference_texts) - 1)
                reference_meta.append({"row": bi, "idxs": idxs})

            y = (ex["completion"].get("value") or ex["completion"].get("content")).rstrip()
            if self.tokenizer.eos_token is not None:
                y = y + self.tokenizer.eos_token
            completion_texts.append(y)

        completion = self.tokenizer(
            completion_texts, padding=True, truncation=True,
            max_length=self.max_completion_length, add_special_tokens=False,
            return_tensors="pt")

        return {
            "prompt_texts": prompt_texts,
            "conditional_texts": conditional_texts,
            "completion_ids": completion["input_ids"],
            "completion_mask": completion["attention_mask"],
            "reference_texts": reference_texts,
            "reference_meta": reference_meta,
        }

class CoSDTrainer(Trainer):
    def __init__(self, *args, score_chunk_size: int = 4, **kwargs):
        self.score_chunk_size = score_chunk_size
        self._metrics = defaultdict(list)
        super().__init__(*args, **kwargs)

    def _token_logps(self, context_texts, completion_ids, model):
        tok = self.processing_class
        device = model.device

        pad_side, trunc_side = tok.padding_side, tok.truncation_side
        tok.padding_side, tok.truncation_side = "left", "left"
        enc = tok(context_texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_CONTEXT_LEN, add_special_tokens=False).to(device)
        tok.padding_side, tok.truncation_side = pad_side, trunc_side

        y_ids = completion_ids.to(device)
        pad_id = tok.pad_token_id
        if pad_id is None:
            raise ValueError("pad_token_id must be set")
        y_mask = (y_ids != pad_id).long()

        input_ids = torch.cat([enc["input_ids"], y_ids], dim=1)
        attention_mask = torch.cat([enc["attention_mask"], y_mask], dim=1)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        labels = input_ids[:, 1:]

        n_y = y_ids.size(1)
        logits_y = logits[:, -n_y:, :]
        labels_y = labels[:, -n_y:].masked_fill(y_mask == 0, -100)

        B, C, V = logits_y.shape
        nll = F.cross_entropy(logits_y.reshape(B * C, V), labels_y.reshape(B * C),
                              reduction="none", ignore_index=-100).reshape(B, C)
        logps = -nll

        if IGNORE_FIRST_K > 0 and n_y > IGNORE_FIRST_K:
            logps = logps[:, IGNORE_FIRST_K:]
            y_ids = y_ids[:, IGNORE_FIRST_K:]
            y_mask = y_mask[:, IGNORE_FIRST_K:]
        return logps, y_ids, y_mask

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        reference_texts = inputs.pop("reference_texts", [])
        reference_meta = inputs.pop("reference_meta", [])
        completion_ids = inputs["completion_ids"]

        with torch.no_grad():
            rep_ids = [completion_ids[m["row"]]
                       for m in reference_meta if m for _ in m["idxs"]]
            rep_ids = torch.stack(rep_ids, 0) if rep_ids else None

            parts, masks = [], []
            n_ref = len(reference_texts) if rep_ids is not None else 0
            for i in range(0, n_ref, self.score_chunk_size):
                lp, _, mk = self._token_logps(
                    reference_texts[i:i + self.score_chunk_size],
                    rep_ids[i:i + self.score_chunk_size], model)
                parts.append(lp)
                masks.append(mk)

            width = max((p.shape[1] for p in parts), default=1)
            pad_to = lambda t: t if t.shape[1] == width else F.pad(t, (0, width - t.shape[1]))
            logps_ref = torch.cat([pad_to(p) for p in parts], 0) if parts else None

            logps_xo, _, _ = self._token_logps(
                inputs["conditional_texts"], completion_ids, model)

        logps_x, y_ids, token_mask = self._token_logps(
            inputs["prompt_texts"], completion_ids, model)
        token_mask_f = token_mask.float()

        advantage = (logps_xo - logps_x).detach()

        W = logps_x.shape[1]
        if logps_ref is not None and logps_ref.shape[1] != W:
            logps_ref = (F.pad(logps_ref, (0, W - logps_ref.shape[1]))
                         if logps_ref.shape[1] < W else logps_ref[:, :W])

        with torch.no_grad():
            for m in (reference_meta if logps_ref is not None else []):
                if not m:
                    continue
                acc = torch.zeros_like(logps_x[m["row"]])
                for gi in m["idxs"]:
                    acc = acc + logps_ref[gi]
                acc = acc / float(len(m["idxs"]))
                advantage[m["row"]] = (logps_xo[m["row"]] - acc).detach()

        token_lengths = token_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        per_token_loss = -(advantage * logps_x) * token_mask_f
        loss = (per_token_loss.sum(dim=1, keepdim=True) / token_lengths).mean()

        if model.training:
            with torch.no_grad():
                active = token_mask_f.sum().item()
                self._metrics["advantage_mean"].append(
                    (advantage * token_mask_f).sum().item() / active)
                self._metrics["advantage_std"].append(
                    advantage[token_mask.bool()].std().item())
                self._metrics["policy_logp"].append(
                    (logps_x * token_mask_f).sum().item() / active)
                self._metrics["loss"].append(loss.detach().float().item())
                self._metrics["rows_with_references"].append(
                    float(sum(1 for m in reference_meta if m)))

        return (loss, None) if return_outputs else loss

    def log(self, logs, start_time=None):
        for key, values in self._metrics.items():
            if values:
                logs[f"cosd/{key}"] = sum(values) / len(values)
        self._metrics.clear()
        super().log(logs, start_time)
