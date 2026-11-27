"""Post-run completeness check:   python -m ifeval_v100.check_results [--model-set X]

Checks every expected (model, condition) run against the grading checklist and prints a
compact summary to paste back. Exit code 0 = everything required is present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import common

RUN_FILES = ("final_prompts.jsonl", "generations.jsonl", "manifest.json", "per_item.jsonl", "scores.json")
REPORT_FILES = ("REPORT.md", "runs.csv", "per_type.csv", "pairwise.csv",
                "scale.png", "per_type_heatmap.png", "length_hist.png")


def nlines(p: Path) -> int:
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-set", default=None)
    ap.add_argument("--phases", default="1 2 3", help='e.g. "1 2" if phase 3 was not run')
    a = ap.parse_args()
    cfg = common.load_config()
    mset = a.model_set or common.default_model_set(cfg)
    models = cfg["model_sets"][mset]
    unavailable = set()
    ua = common.RESULTS / "env" / "unavailable_models.txt"
    if ua.exists():
        unavailable = {x.strip() for x in ua.read_text().split() if x.strip()}
    all_conds = [c for ph in a.phases.split() for c in cfg["phases"][ph]]
    errors, warns = [], []

    print(f"== results dir: {common.RESULTS}   model set: {mset}")
    print(f"{'model':14s} {'cond':22s} {'n':>4s} {'P-S':>6s} {'I-S':>6s} {'P-L':>6s} {'I-L':>6s} "
          f"{'trunc%':>6s} {'medTok':>6s} {'gpu_h':>6s}  hw/backend")
    total_h, seen_gpus, revs = 0.0, set(), {}
    for m in models:
        if m in unavailable:
            warns.append(f"{m}: unavailable (not downloaded / gated) - skipped")
            continue
        for c in all_conds:
            d = common.RUNS / m / c
            missing = [f for f in RUN_FILES if not (d / f).exists()]
            if missing:
                errors.append(f"{m}/{c}: missing {','.join(missing)}")
                continue
            man = json.loads((d / "manifest.json").read_text())
            sc = json.loads((d / "scores.json").read_text())
            gens = common.read_jsonl(d / "generations.jsonl")
            n_fp = nlines(d / "final_prompts.jsonl")
            if not man.get("complete"):
                errors.append(f"{m}/{c}: manifest complete=false")
            if man.get("limit"):
                errors.append(f"{m}/{c}: run used --limit {man['limit']}")
            if len(gens) != 541 or n_fp != 541 or sc.get("n_prompts") != 541:
                errors.append(f"{m}/{c}: counts gen={len(gens)} prompts={n_fp} scored={sc.get('n_prompts')} (want 541)")
            if sc.get("n_instructions") not in (None, 834):
                errors.append(f"{m}/{c}: n_instructions={sc.get('n_instructions')} (want 834)")
            if sc.get("langdetect_seed") is None:
                warns.append(f"{m}/{c}: langdetect not seeded")
            empty = sum(1 for g in gens if not g["response"].strip())
            if empty > 27:  # >5% empty responses is suspicious
                warns.append(f"{m}/{c}: {empty} empty responses")
            trunc = 100 * sum(g["truncated"] for g in gens) / max(len(gens), 1)
            if trunc > 10:
                warns.append(f"{m}/{c}: {trunc:.1f}% truncated at max_new_tokens")
            toks = sorted(g["n_output_tokens"] for g in gens)
            med = toks[len(toks) // 2] if toks else 0
            h = man.get("gpu_hours_generation") or 0.0
            total_h += h
            env = man.get("env", {})
            if env.get("repo_dirty"):
                warns.append(f"{m}/{c}: repo had uncommitted changes when run")
            gpu = (env.get("nvidia_smi") or "").strip().splitlines()
            if gpu:
                seen_gpus.add(gpu[0][:60])
            revs.setdefault(m, set()).add(man.get("revision"))
            hb = f"{man.get('hw_profile')}/{man.get('backend', {}).get('backend', man.get('backend'))}"
            print(f"{m:14s} {c:22s} {len(gens):4d} {sc['prompt_strict']:6.2f} {sc['inst_strict']:6.2f} "
                  f"{sc['prompt_loose']:6.2f} {sc['inst_loose']:6.2f} {trunc:6.1f} {med:6d} {h:6.3f}  {hb}")

    for m, r in revs.items():
        if len(r) > 1 or cfg["models"][m]["revision"] not in r:
            errors.append(f"{m}: revision mismatch {r}")

    rep = common.RESULTS / "report"
    miss = [f for f in REPORT_FILES if not (rep / f).exists()]
    if miss:
        errors.append(f"report: missing {', '.join(miss)} (run: python -m ifeval_v100.analyze)")
    elif "TODO" in (rep / "REPORT.md").read_text(encoding="utf-8"):
        warns.append("REPORT.md contains TODO markers")
    for f in ("scorer_validation.md", "scorer_validation.json"):
        if not (common.RESULTS / f).exists():
            errors.append(f"missing results/{f}")
    for f in ("env.json", "pip_freeze.txt"):
        if not (common.RESULTS / "env" / f).exists():
            warns.append(f"missing results/env/{f}")
    fl = common.RESULTS / "logs" / "failures.txt"
    if fl.exists() and fl.read_text().strip():
        warns.append("failures.txt:\n    " + fl.read_text().strip().replace("\n", "\n    "))

    print(f"\n== GPU(s) seen: {sorted(seen_gpus) or 'none recorded'}")
    print(f"== total generation GPU-hours: {total_h:.2f}")
    if (rep / "REPORT.md").exists():
        txt = (rep / "REPORT.md").read_text(encoding="utf-8")
        print(f"== REPORT.md: {len(txt.splitlines())} lines; sections: "
              + " | ".join(l[3:40] for l in txt.splitlines() if l.startswith("## ")))
    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print("\nRESULT:", "OK - complete" if not errors else f"{len(errors)} error(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
