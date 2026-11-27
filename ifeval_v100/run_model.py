"""Load one model once, run a list of conditions, score each run.

  python -m ifeval_v100.run_model --model qwen2.5-1.5b --phase 1
  python -m ifeval_v100.run_model --model qwen2.5-0.5b --conditions greedy_plain_r0 --limit 8 --backend hf

Each run writes results/runs/<model>/<condition>/:
  final_prompts.jsonl  exact string + token count the model saw, per item
  generations.jsonl    raw response, output token count, finish_reason, truncated
  manifest.json        model+revision, tokenizer, condition, decoding, env, timings
  per_item.jsonl / scores.json   official scorer output
A run whose manifest says complete is skipped (resumable).
"""
from __future__ import annotations

import argparse
import random
import time

from ifeval_v100 import common, prompts, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--phase")
    g.add_argument("--conditions", nargs="+")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke tests)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--hw-profile", default=None, help="v100 | rtx_8gb | auto (default: $HW_PROFILE or auto)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="override (smoke tests only)")
    a = ap.parse_args()

    cfg = common.load_config()
    m = cfg["models"][a.model]
    hw = common.resolve_hw_profile(cfg, a.hw_profile)
    gen_cfg = common.generation_config(cfg, hw)
    print(f"[hw] profile={hw} max_model_len={gen_cfg['max_model_len']} "
          f"gpu_mem_util={gen_cfg['gpu_memory_utilization']} enforce_eager={gen_cfg.get('enforce_eager', False)}")
    if a.max_new_tokens:
        gen_cfg["max_new_tokens"] = a.max_new_tokens
    conds = cfg["phases"][a.phase] if a.phase else a.conditions
    items = common.load_items(a.limit)
    out_root = common.RUNS if not a.limit else common.RESULTS / "smoke"

    todo = []
    for c in conds:
        d = out_root / a.model / c
        man = d / "manifest.json"
        if man.exists() and not a.force and '"complete": true' in man.read_text():
            print(f"[skip] {a.model}/{c} already complete")
            continue
        todo.append(c)
    if not todo:
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m["hf_id"], revision=m["revision"])
    stop_ids = prompts.stop_token_ids(tok, m["hf_id"], m["revision"])
    from ifeval_v100.backends import make_backend
    backend = make_backend(a.backend, m["hf_id"], m["revision"], gen_cfg, stop_ids, tok)
    env = common.env_info()
    print(f"[load] {a.model} via {backend.name} in {backend.load_seconds:.1f}s")

    for c in todo:
        cond = cfg["conditions"][c]
        variant = cfg["variants"][cond["variant"]]
        d = out_root / a.model / c
        finals = []
        for it in items:
            text, ids = prompts.build_final(tok, it, variant, gen_cfg.get("chat_template_kwargs", {}))
            finals.append({"key": it["key"], "final_prompt": text, "n_prompt_tokens": len(ids), "ids": ids})
        order = list(range(len(items)))
        if cond.get("order_seed"):
            random.Random(cond["order_seed"]).shuffle(order)
        print(f"[run] {a.model}/{c}: {len(items)} items, variant={cond['variant']}, decoding={cond['decoding']}")
        print("----- example final prompt (item 0) -----\n" + finals[0]["final_prompt"] + "\n-----")
        t0 = time.time()
        outs = backend.generate([finals[i]["ids"] for i in order], cond["decoding"], cond.get("seed", 0))
        gen_s = time.time() - t0
        res = [None] * len(items)
        for j, i in enumerate(order):
            res[i] = outs[j]
        common.write_jsonl(d / "final_prompts.jsonl",
                           [{k: v for k, v in f.items() if k != "ids"} for f in finals])
        common.write_jsonl(d / "generations.jsonl", [
            {"key": it["key"], "response": r["response"], "n_output_tokens": r["n_output_tokens"],
             "finish_reason": r["finish_reason"], "truncated": r["truncated"]}
            for it, r in zip(items, res)])
        manifest = {
            "complete": False, "model_key": a.model, "hf_id": m["hf_id"], "revision": m["revision"],
            "tokenizer": {"hf_id": m["hf_id"], "revision": m["revision"], "class": type(tok).__name__,
                          "chat_template_sha256": __import__("hashlib").sha256(
                              (tok.chat_template or "").encode()).hexdigest()},
            "params_b": m.get("params_b"), "condition": c, "condition_spec": cond,
            "variant_spec": variant, "hw_profile": hw, "generation": gen_cfg, "stop_token_ids": stop_ids,
            "precision": gen_cfg["dtype"] if backend.name == "vllm" else backend.info.get("torch_dtype"),
            "backend": backend.info, "n_items": len(items), "limit": a.limit,
            "load_seconds": backend.load_seconds, "generation_seconds": gen_s,
            "gpu_hours_generation": gen_s / 3600,
            "started_unix": t0, "env": env,
        }
        common.write_json(d / "manifest.json", manifest)
        sc = score.score_run_dir(d, "official", 0)
        manifest["complete"] = True
        manifest["scores"] = {k: sc[k] for k in ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose")}
        common.write_json(d / "manifest.json", manifest)
        trunc = sum(r["truncated"] for r in res)
        print(f"[done] {a.model}/{c}: {gen_s/60:.1f} min, prompt_strict={sc['prompt_strict']:.2f} "
              f"inst_strict={sc['inst_strict']:.2f} truncated={trunc}/{len(res)}")


if __name__ == "__main__":
    main()
