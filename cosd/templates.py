import os

HINDSIGHT_FORMATS = {
    "format1": (
        "=== HINDSIGHT CONTEXT ===\n"
        "[The following is a future user message. Use this to guide your "
        "answer to the user prompt.]\n"
        "{o}"
    ),
    "format2": (
        "Advance notice of what happens next: written below are the very words that "
        "this person sends back afterwards. Let them shape how you address their "
        "request.\n"
        "{o}"
    ),
    "format3": (
        "<<<PREVIEW OF A LATER TURN>>>\n"
        "(Below sits what the human types later on; build your reply to their query "
        "around it.)\n"
        "{o}"
    ),
}

HISTORY_MARKER = (
    "=== USER PROFILE ===\n"
    "[Things this user has asked for in earlier, unrelated conversations. Use this to "
    "anticipate what they will want.]\n"
)

ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}

def hindsight_format():
    name = os.environ.get("COSD_HINDSIGHT_FORMAT", "format1")
    if name not in HINDSIGHT_FORMATS:
        raise ValueError(
            f"COSD_HINDSIGHT_FORMAT must be one of {sorted(HINDSIGHT_FORMATS)}, got {name!r}")
    return name

def normalize_messages(messages):
    out = []
    for m in messages:
        if "value" in m and "content" not in m:
            role = m.get("from", "user")
            out.append({"role": ROLE_MAP.get(role, role), "content": m["value"]})
        else:
            out.append(dict(m))
    return out

def build_hindsight_context(dialogue, o, fmt=None):
    template = HINDSIGHT_FORMATS[fmt or hindsight_format()]
    history = [dict(m) for m in dialogue]
    history.append({"role": "assistant", "content": template.format(o=o)})
    return history

def build_history_prefix(history_text):
    text = (history_text or "").strip()
    if not text:
        return []
    return [{"role": "system", "content": HISTORY_MARKER + text}]

def render(tokenizer, messages, add_generation_prompt=True):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
