"""Validate the scoring instrument before trusting any model number.

1. Reproduce the paper: score the authors' released GPT-4 responses with the
   authors' scorer against the item list those responses were made for
   (input_data.jsonl at commit 066e1eda, 2023-11-27) and compare to Table 3.
2. Scorer determinism: langdetect is unseeded in the official code. Score the
   same responses N times unseeded and report the spread; then seeded.
3. Agreement: official scorer vs. the lm-evaluation-harness port, per
   instruction and per prompt (on GPT-4 responses, and on any model runs given).
4. Item-list drift: diff the paper-era item list vs. the pinned current one,
   and (optionally, needs network) vs. the HF dataset google/IFEval.

CPU only, ~1-3 min. Writes results/scorer_validation.{json,md}.
"""
from __future__ import annotations

import argparse
import glob
import statistics
from pathlib import Path

from ifeval_v100 import common, score

PAPER_ERA_ITEMS = common.DATA_PATH
METRICS = ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose")


def gpt4_setup():
    items = common.read_jsonl(PAPER_ERA_ITEMS)
    by_prompt = {r["prompt"]: r["response"] for r in common.read_jsonl(common.GPT4_RESPONSES)}
    return items, {it["key"]: by_prompt[it["prompt"]] for it in items}


def agreement(a: list[dict], b: list[dict]) -> dict:
    inst_tot = inst_agree = p_agree = 0
    dis = []
    for x, y in zip(a, b):
        assert x["key"] == y["key"]
        for lvl in ("strict", "loose"):
            for i, (u, v) in enumerate(zip(x[lvl], y[lvl])):
                inst_tot += 1
                inst_agree += u == v
                if u != v:
                    dis.append({"key": x["key"], "level": lvl,
                                "instruction": x["instruction_id_list"][i], "a": u, "b": v})
        p_agree += (x["prompt_strict"] == y["prompt_strict"]) + (x["prompt_loose"] == y["prompt_loose"])
    return {"instruction_agreement": inst_agree / inst_tot,
            "prompt_agreement": p_agree / (2 * len(a)),
            "n_disagreements": len(dis), "disagreements": dis[:50]}


def item_drift(skip_hf: bool = False) -> dict:
    cur = {r["key"]: r for r in common.load_items()}          # paper-era, canonical
    new = {r["key"]: r for r in common.read_jsonl(common.LATEST_DATA_PATH)}
    out = {"canonical_vs_latest_github_changed_keys": sorted(k for k in cur if new.get(k) != cur[k])}
    if skip_hf:
        return out
    try:
        from datasets import load_dataset
        name, rev = common.HF_DATASET
        ds = load_dataset(name, split="train", revision=rev)
        hf = {}
        for r in ds:
            r = dict(r)
            r["kwargs"] = [{k: v for k, v in kw.items() if v is not None} for kw in r["kwargs"]]
            hf[r["key"]] = r
        diff = sorted(k for k in cur if k not in hf or any(
            hf[k][f] != cur[k][f] for f in ("prompt", "instruction_id_list", "kwargs")))
        out["hf_dataset"] = f"{name}@{rev}"
        out["hf_vs_canonical_changed_keys"] = diff
    except Exception as e:  # noqa: BLE001
        out["hf_dataset_error"] = repr(e)[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseeded-repeats", type=int, default=30)
    ap.add_argument("--runs-glob", default=str(common.RUNS / "*" / "greedy_plain_r0" / "generations.jsonl"))
    ap.add_argument("--skip-hf", action="store_true")
    a = ap.parse_args()

    cfg = common.load_config()
    items, resp = gpt4_setup()
    rep = {"paper_gpt4": cfg["paper"]["GPT-4"]}

    seeded = score.score_responses(items, resp, "official", langdetect_seed=0)
    rep["official_seeded"] = score.metrics(seeded)

    unseeded = [score.metrics(score.score_responses(items, resp, "official", langdetect_seed=None))
                for _ in range(a.unseeded_repeats)]
    rep["official_unseeded"] = {m: {"min": min(u[m] for u in unseeded), "max": max(u[m] for u in unseeded),
                                    "mean": statistics.mean(u[m] for u in unseeded)} for m in METRICS}
    rep["official_unseeded_n"] = a.unseeded_repeats
    rep["paper_within_unseeded_range"] = {
        m: rep["official_unseeded"][m]["min"] - 1e-9 <= cfg["paper"]["GPT-4"][m] <= rep["official_unseeded"][m]["max"] + 1e-9
        for m in METRICS}

    lme = score.score_responses(items, resp, "lmeval", langdetect_seed=0)
    rep["lmeval_seeded"] = score.metrics(lme)
    rep["agreement_gpt4_official_vs_lmeval"] = agreement(seeded, lme)

    model_agree = {}
    all_items = {it["key"]: it for it in common.load_items()}
    for g in sorted(glob.glob(a.runs_glob)):
        gens = common.read_jsonl(g)
        sel = [all_items[x["key"]] for x in gens]
        r = {x["key"]: x["response"] for x in gens}
        model_agree["/".join(Path(g).parts[-3:-1])] = agreement(
            score.score_responses(sel, r, "official", 0), score.score_responses(sel, r, "lmeval", 0))
    rep["agreement_model_runs_official_vs_lmeval"] = model_agree

    rep["item_drift"] = item_drift(skip_hf=a.skip_hf)

    common.write_json(common.RESULTS / "scorer_validation.json", rep)

    p = cfg["paper"]["GPT-4"]
    L = ["# Scorer validation", "",
         "Authors' GPT-4 responses (2023-11-07), authors' item list at commit 066e1eda (2023-11-27),",
         f"authors' scorer at commit {common.OFFICIAL_SCORER_COMMIT[:8]}.", "",
         "| metric | paper Table 3 | official, langdetect seeded | official, unseeded min–max (n=%d) | lm-eval port, seeded |" % a.unseeded_repeats,
         "|---|---|---|---|---|"]
    for m in METRICS:
        u = rep["official_unseeded"][m]
        L.append(f"| {m} | {p[m]:.2f} | {rep['official_seeded'][m]:.2f} | {u['min']:.2f}–{u['max']:.2f} | {rep['lmeval_seeded'][m]:.2f} |")
    ag = rep["agreement_gpt4_official_vs_lmeval"]
    L += ["", f"Official vs lm-eval port on GPT-4 responses (both seeded): instruction-level agreement "
              f"{100*ag['instruction_agreement']:.2f}%, prompt-level {100*ag['prompt_agreement']:.2f}% "
              f"({ag['n_disagreements']} instruction disagreements)."]
    for k, v in model_agree.items():
        L.append(f"- {k}: instruction agreement {100*v['instruction_agreement']:.2f}%, "
                 f"prompt {100*v['prompt_agreement']:.2f}% ({v['n_disagreements']} disagreements)")
    d = rep["item_drift"]
    inr = rep["paper_within_unseeded_range"]
    L += ["", "Paper value inside the unseeded-scorer range: " + ", ".join(f"{m} {'yes' if v else 'NO'}" for m, v in inr.items())
          + ". The official scorer is not deterministic (langdetect is unseeded); we seed it (DetectorFactory.seed=0) for all model runs.",
          "", "Item list: canonical = authors' file at commit 066e1eda (2023-11-27), the list the paper's GPT-4 responses answer. "
          f"The authors' latest file differs on keys {d['canonical_vs_latest_github_changed_keys']}."]
    if "hf_vs_canonical_changed_keys" in d:
        L.append(f"HF {d['hf_dataset']} vs canonical: differ on keys {d['hf_vs_canonical_changed_keys'] or 'none (identical)'}.")
    (common.RESULTS / "scorer_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
