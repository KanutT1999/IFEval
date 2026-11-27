"""Two generation backends with one interface.

vllm : primary on the V100 (vLLM 0.6.6.post1, V0 engine, xFormers attention, fp16).
hf   : fallback / CPU smoke test (transformers.generate, left-padded batches).

generate(prompt_ids, decoding, seed) -> list of dicts:
  {"response", "n_output_tokens", "finish_reason", "truncated"}
"""
from __future__ import annotations

import time


class VLLMBackend:
    name = "vllm"

    def __init__(self, hf_id, revision, gen_cfg, stop_ids, seed=0):
        from vllm import LLM
        self.gen_cfg, self.stop_ids = gen_cfg, stop_ids
        t = time.time()
        extra = {"max_num_seqs": int(gen_cfg["max_num_seqs"])} if gen_cfg.get("max_num_seqs") else {}
        self.llm = LLM(model=hf_id, revision=revision, tokenizer_revision=revision,
                       dtype=gen_cfg["dtype"], max_model_len=gen_cfg["max_model_len"],
                       gpu_memory_utilization=gen_cfg["gpu_memory_utilization"],
                       enforce_eager=bool(gen_cfg.get("enforce_eager", False)),
                       seed=seed, trust_remote_code=False, **extra)
        self.load_seconds = time.time() - t
        import vllm
        self.info = {"backend": "vllm", "vllm_version": vllm.__version__}

    def generate(self, prompt_ids, decoding, seed=None):
        from vllm import SamplingParams
        try:
            from vllm.inputs import TokensPrompt
            prompts = [TokensPrompt(prompt_token_ids=p) for p in prompt_ids]
        except ImportError:
            prompts = [{"prompt_token_ids": p} for p in prompt_ids]
        g = self.gen_cfg
        params = []
        for i in range(len(prompt_ids)):
            if decoding == "greedy":
                sp = SamplingParams(temperature=0.0, max_tokens=g["max_new_tokens"],
                                    stop_token_ids=self.stop_ids, skip_special_tokens=True)
            else:  # per-request seed -> reproducible sampling independent of batching
                sp = SamplingParams(temperature=g["sampling"]["temperature"], top_p=g["sampling"]["top_p"],
                                    max_tokens=g["max_new_tokens"], seed=seed * 100003 + i,
                                    stop_token_ids=self.stop_ids, skip_special_tokens=True)
            params.append(sp)
        outs = self.llm.generate(prompts, params, use_tqdm=True)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append({"response": c.text, "n_output_tokens": len(c.token_ids),
                        "finish_reason": c.finish_reason, "truncated": c.finish_reason == "length"})
        return res


class HFBackend:
    name = "hf"

    def __init__(self, hf_id, revision, gen_cfg, stop_ids, seed=0, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.gen_cfg, self.stop_ids, self.tok = gen_cfg, stop_ids, tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        t = time.time()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, dtype=dtype,
                                                              attn_implementation="sdpa")
        except TypeError:  # transformers < 4.56 uses torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, torch_dtype=dtype,
                                                              attn_implementation="sdpa")
        self.model.to(self.device).eval()
        self.load_seconds = time.time() - t
        import transformers
        self.info = {"backend": "hf", "transformers_version": transformers.__version__,
                     "device": self.device, "torch_dtype": str(dtype),
                     "batch_size": gen_cfg["hf_batch_size"]}

    def generate(self, prompt_ids, decoding, seed=None):
        torch = self.torch
        g = self.gen_cfg
        bs = g["hf_batch_size"] if self.device == "cuda" else 4
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop_ids[0]
        order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]))
        res = [None] * len(prompt_ids)
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            L = max(len(prompt_ids[i]) for i in idx)
            inp = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                p = prompt_ids[i]
                inp[r, L - len(p):] = torch.tensor(p)
                att[r, L - len(p):] = 1
            # A fresh GenerationConfig so the model repo's defaults (e.g. Qwen's
            # repetition_penalty=1.05, top_k=20) are NOT silently applied -
            # matching vLLM, where only the SamplingParams we pass are used.
            from transformers import GenerationConfig
            gc = GenerationConfig(max_new_tokens=g["max_new_tokens"], eos_token_id=self.stop_ids,
                                  pad_token_id=pad, repetition_penalty=1.0)
            if decoding == "greedy":
                gc.do_sample = False
            else:
                torch.manual_seed(seed * 100003 + b)
                gc.do_sample, gc.temperature, gc.top_p, gc.top_k = (
                    True, g["sampling"]["temperature"], g["sampling"]["top_p"], None)
            with torch.no_grad():
                out = self.model.generate(input_ids=inp.to(self.device), attention_mask=att.to(self.device),
                                          generation_config=gc)
            for r, i in enumerate(idx):
                gen = out[r, L:].tolist()
                n = len(gen)
                for j, t in enumerate(gen):
                    if t in self.stop_ids:
                        n = j
                        stopped = True
                        break
                else:
                    stopped = False
                toks = gen[:n]
                res[i] = {"response": self.tok.decode(toks, skip_special_tokens=True),
                          "n_output_tokens": n + (1 if stopped else 0),
                          "finish_reason": "stop" if stopped else "length",
                          "truncated": not stopped}
            print(f"  hf batch {b // bs + 1}/{(len(order) + bs - 1) // bs}", flush=True)
        return res


def make_backend(kind, hf_id, revision, gen_cfg, stop_ids, tokenizer, seed=0):
    if kind == "auto":
        try:
            import vllm  # noqa: F401
            import torch
            kind = "vllm" if torch.cuda.is_available() else "hf"
        except ImportError:
            kind = "hf"
    if kind == "vllm":
        return VLLMBackend(hf_id, revision, gen_cfg, stop_ids, seed)
    return HFBackend(hf_id, revision, gen_cfg, stop_ids, seed, tokenizer)
