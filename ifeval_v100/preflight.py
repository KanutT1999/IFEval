"""Pre-flight on the rented box: GPU sanity, model downloads at pinned revisions.

  python -m ifeval_v100.preflight --model-set v100_16gb

Fails fast (before any GPU time is spent on generation) if: no CUDA, a gated
model has no token, or a pinned revision is missing. Writes results/env/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from ifeval_v100 import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-set", default="rtx_8gb")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    cfg = common.load_config()
    env = common.env_info()
    out = common.RESULTS / "env"
    common.write_json(out / "env.json", env)
    common.write_freeze(out / "pip_freeze.txt")
    print(f"GPU: {env.get('gpu_name')} cc={env.get('gpu_capability')} mem={env.get('gpu_mem_gb')}GB "
          f"torch={env['libraries']['torch']} vllm={env['libraries']['vllm']}")

    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        y = x @ x
    torch.cuda.synchronize()
    tf = 20 * 2 * 4096**3 / (time.time() - t) / 1e12
    print(f"fp16 matmul: {tf:.1f} TFLOPS (healthy: V100 ~60-90, RTX 3070 Ti ~40-70)")
    common.write_json(out / "fp16_matmul.json", {"tflops": tf, "finite": bool(torch.isfinite(y).all())})

    mem = env.get("gpu_mem_gb") or 0
    hw = common.resolve_hw_profile(cfg)
    gen = common.generation_config(cfg, hw)
    common.write_json(out / "hw_profile.json", {"profile": hw, "generation": gen})
    print(f"hardware profile: {hw}")
    models = cfg["model_sets"][a.model_set]
    bad = []
    for mk in models:
        m = cfg["models"][mk]
        need = 2 * m["params_b"] + 1.5
        if need > mem * gen["gpu_memory_utilization"]:
            print(f"[warn] {mk}: ~{need:.1f} GB fp16 weights+overhead vs {mem} GB GPU -> may not fit (will be tried last)")
        if a.no_download:
            continue
        try:
            from huggingface_hub import snapshot_download
            p = snapshot_download(m["hf_id"], revision=m["revision"],
                                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
            print(f"[ok] {mk} @ {m['revision'][:10]} -> {p}")
        except Exception as e:  # noqa: BLE001
            bad.append(mk)
            hint = " (gated: accept the license on huggingface.co and export HF_TOKEN)" if m.get("gated") else ""
            print(f"[FAIL] {mk}: {type(e).__name__}: {str(e)[:200]}{hint}")
    if bad:
        (out / "unavailable_models.txt").write_text("\n".join(bad) + "\n")
        print(f"\n{len(bad)} model(s) unavailable, run_all.sh will skip them: {bad}")


if __name__ == "__main__":
    main()
