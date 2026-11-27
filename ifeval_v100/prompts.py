"""The prompt path: item -> messages -> exact final string -> token ids.

We tokenize ourselves (add_special_tokens=False when a chat template is used,
since templates already contain BOS) and hand token ids to the backend, so the
string written to final_prompts.jsonl is exactly what the model saw.
"""
from __future__ import annotations


def build_messages(item: dict, variant: dict) -> list[dict]:
    msgs = []
    if variant.get("system"):
        msgs.append({"role": "system", "content": variant["system"]})
    msgs.append({"role": "user", "content": variant["user"].replace("{prompt}", item["prompt"])})
    return msgs


def build_final(tokenizer, item: dict, variant: dict, template_kwargs: dict) -> tuple[str, list[int]]:
    if variant.get("chat_template", True):
        msgs = build_messages(item, variant)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             **template_kwargs)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    else:
        text = variant["user"].replace("{prompt}", item["prompt"])
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        if tokenizer.bos_token_id is not None and ids and ids[0] == tokenizer.bos_token_id:
            text = tokenizer.bos_token + text  # make the logged string honest about BOS
    return text, ids


def stop_token_ids(tokenizer, model_id: str, revision: str) -> list[int]:
    """EOS ids = tokenizer eos plus generation_config eos (Llama 3 uses several)."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    try:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained(model_id, revision=revision)
        e = gc.eos_token_id
        ids.update([e] if isinstance(e, int) else (e or []))
    except Exception:  # noqa: BLE001
        pass
    return sorted(ids)
