import os
import threading
import time

import requests

MIN_BUDGET = 4096

def config():
    key = os.environ.get("JUDGE_API_KEY")
    if not key:
        raise RuntimeError("JUDGE_API_KEY is not set")
    budget_key = os.environ.get("JUDGE_BUDGET_KEY", "max_completion_tokens")
    if budget_key not in ("max_completion_tokens", "max_tokens"):
        raise ValueError("JUDGE_BUDGET_KEY must be max_completion_tokens or max_tokens")
    return {
        "url": os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1").rstrip("/")
               + "/chat/completions",
        "key": key,
        "model": os.environ.get("JUDGE_MODEL", "gpt-5-mini"),
        "budget_key": budget_key,
        "effort": os.environ.get("JUDGE_EFFORT", "low"),
    }

class RateLimiter:

    def __init__(self, per_minute):
        self.interval = 60.0 / max(per_minute, 1)
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if wait > 0:
            time.sleep(wait)

def call(cfg, messages, max_tokens, timeout=300):
    body = {"model": cfg["model"], "messages": messages, "temperature": 0.0,
            "stream": False, cfg["budget_key"]: max(max_tokens, MIN_BUDGET)}
    if cfg["budget_key"] == "max_completion_tokens":
        body["reasoning_effort"] = cfg["effort"]
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {cfg['key']}"}
    try:
        r = requests.post(cfg["url"], headers=headers, json=body, timeout=timeout)
    except Exception:
        return 0, "", {}
    if r.status_code != 200:
        return r.status_code, "", {}
    d = r.json()
    choices = d.get("choices") or []
    if not choices:
        return 200, "", d.get("usage") or {}
    message = choices[0].get("message") or {}
    return 200, (message.get("content") or "").strip(), (d.get("usage") or {})

def probe(cfg, attempts=5):
    status, text = 0, ""
    for i in range(attempts):
        status, text, _ = call(cfg, [{"role": "user", "content": "Reply with exactly OK."}], 64)
        if status == 200 and text:
            return status, text
        time.sleep(min(30, 3 * (2 ** i)))
    return status, text
