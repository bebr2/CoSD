from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, Trainer

from .templates import normalize_messages, render
from .trainer import IGNORE_FIRST_K, MAX_COMPLETION_LEN, MAX_CONTEXT_LEN

@dataclass
class SFTCollator:
    tokenizer: PreTrainedTokenizerBase
    max_completion_length: int = MAX_COMPLETION_LEN

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_texts, completion_texts = [], []
        for ex in examples:
            prompt_texts.append(render(self.tokenizer, normalize_messages(ex["prompt"])))
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
            "completion_ids": completion["input_ids"],
            "completion_mask": completion["attention_mask"],
        }

class SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        tok = self.processing_class
        device = model.device
        y_ids = inputs["completion_ids"].to(device)

        pad_side, trunc_side = tok.padding_side, tok.truncation_side
        tok.padding_side, tok.truncation_side = "left", "left"
        enc = tok(inputs["prompt_texts"], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_CONTEXT_LEN, add_special_tokens=False).to(device)
        tok.padding_side, tok.truncation_side = pad_side, trunc_side

        y_mask = (y_ids != tok.pad_token_id).long()
        input_ids = torch.cat([enc["input_ids"], y_ids], dim=1)
        attention_mask = torch.cat([enc["attention_mask"], y_mask], dim=1)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        labels = input_ids[:, 1:]

        n_y = y_ids.size(1)
        logits_y = logits[:, -n_y:, :]
        labels_y = labels[:, -n_y:].masked_fill(y_mask == 0, -100)

        if IGNORE_FIRST_K > 0 and n_y > IGNORE_FIRST_K:
            logits_y = logits_y[:, IGNORE_FIRST_K:, :]
            labels_y = labels_y[:, IGNORE_FIRST_K:]
            y_mask = y_mask[:, IGNORE_FIRST_K:]

        B, C, V = logits_y.shape
        nll = F.cross_entropy(logits_y.reshape(B * C, V), labels_y.reshape(B * C),
                              reduction="none", ignore_index=-100).reshape(B, C)

        mask_f = y_mask.float()
        lengths = mask_f.sum(dim=1).clamp(min=1.0)
        loss = ((nll * mask_f).sum(dim=1) / lengths).mean()
        return (loss, None) if return_outputs else loss
