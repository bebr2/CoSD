import sys

import vllm

_OriginalLLM = vllm.LLM

class _LLMWithoutDevice(_OriginalLLM):
    def __init__(self, *args, **kwargs):
        kwargs.pop("device", None)
        super().__init__(*args, **kwargs)

vllm.LLM = _LLMWithoutDevice

import lm_eval.models.vllm_causallms as _vllm_backend

if getattr(_vllm_backend, "LLM", None) is _OriginalLLM:
    _vllm_backend.LLM = _LLMWithoutDevice

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    sys.argv = ["lm_eval"] + sys.argv[1:]
    cli_evaluate()
