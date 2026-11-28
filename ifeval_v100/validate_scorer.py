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
    reseeded = [score.score_responses(items, resp, "official", langdetect_seed=0) for _ in range(3)]
    rep["seeded_is_deterministic"] = all(r == seeded for r in reseeded)

    def spread(ld, rs, n):
        per = [score.score_responses(items, resp, "official", langdetect_seed=ld, random_seed=rs) for _ in range(n)]
        ms = [score.metrics(x) for x in per]
        flips = sorted({items[j]["key"] for j in range(len(items))
                        if len({(tuple(x[j]["strict"]), tuple(x[j]["loose"])) for x in per}) > 1})
        return ({m: {"min": min(u[m] for u in ms), "max": max(u[m] for u in ms),
                     "mean": statistics.mean(u[m] for u in ms)} for m in METRICS}, flips)

    n = a.unseeded_repeats
    rep["official_unseeded"], flips_all = spread(None, None, n)            # original behaviour
    rep["only_random_unseeded"], flips_rand = spread(0, None, max(n // 3, 5))   # langdetect seeded
    rep["only_langdetect_unseeded"], flips_ld = spread(None, 0, max(n // 3, 5))  # random seeded
    rep["official_unseeded_n"] = n
    by_key = {it["key"]: it for it in items}
    rep["flipping_items"] = {
        "any": flips_all, "from_python_random": flips_rand, "from_langdetect": flips_ld,
        "detail": {k: {"instructions": by_key[k]["instruction_id_list"], "kwargs": by_key[k]["kwargs"]}
                   for k in sorted(set(flips_all) | set(flips_rand) | set(flips_ld))}}
    # Exact attainable range: re-score each flipping item under many random draws (items are
    # independent), then add the per-item extremes to the fixed contribution of all other items.
    flip_idx = [j for j, it in enumerate(items) if it["key"] in set(flips_all) | set(flips_rand)]
    att = {}
    if flip_idx:
        opts = {j: [score.score_responses([items[j]], resp, "official", langdetect_seed=0, random_seed=s)[0]
                    for s in range(300)] for j in flip_idx}
        n_inst = sum(len(r["strict"]) for r in seeded)
        for m in METRICS:
            mode, lvl = m.split("_")[1], m.split("_")[0]
            val = (lambda r: int(r[f"prompt_{mode}"])) if lvl == "prompt" else (lambda r: sum(r[mode]))
            den = len(items) if lvl == "prompt" else n_inst
            base = sum(val(r) for j, r in enumerate(seeded) if j not in opts)
            lo = base + sum(min(val(r) for r in o) for o in opts.values())
            hi = base + sum(max(val(r) for r in o) for o in opts.values())
            att[m] = {"min": 100 * lo / den, "max": 100 * hi / den}
    else:
        att = {m: {"min": rep["official_seeded"][m], "max": rep["official_seeded"][m]} for m in METRICS}
    rep["official_attainable_range"] = att
    # Table 3 is printed to 2 decimals (e.g. 712/834 = 85.3717 -> "85.37"), so compare at that precision.
    rep["paper_within_unseeded_range"] = {
        m: round(att[m]["min"], 2) <= cfg["paper"]["GPT-4"][m] <= round(att[m]["max"], 2)
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
         "| metric | paper Table 3 | official, seeded (ours) | official, original unseeded min–max (n=%d) "
         "| exact attainable range | only `random` unseeded | only langdetect unseeded | lm-eval port, seeded |" % a.unseeded_repeats,
         "|---|---|---|---|---|---|---|---|"]
    rng = lambda u: f"{u['min']:.2f}–{u['max']:.2f}"
    for m in METRICS:
        L.append(f"| {m} | {p[m]:.2f} | {rep['official_seeded'][m]:.2f} | {rng(rep['official_unseeded'][m])} | {rng(att[m])} "
                 f"| {rng(rep['only_random_unseeded'][m])} | {rng(rep['only_langdetect_unseeded'][m])} "
                 f"| {rep['lmeval_seeded'][m]:.2f} |")
    fl = rep["flipping_items"]
    L += ["", f"Seeded scoring repeated 4x gives identical per-instruction results: "
              f"{'yes' if rep['seeded_is_deterministic'] else 'NO'}.",
          f"Items whose score changes between unseeded runs: {fl['any']} "
          f"(from Python `random`: {fl['from_python_random']}; from langdetect: {fl['from_langdetect']})."]
    for k, v in fl["detail"].items():
        L.append(f"- key {k}: {v['instructions']} kwargs={v['kwargs']}")
    ag = rep["agreement_gpt4_official_vs_lmeval"]
    L += ["", f"Official vs lm-eval port on GPT-4 responses (both seeded): instruction-level agreement "
              f"{100*ag['instruction_agreement']:.2f}%, prompt-level {100*ag['prompt_agreement']:.2f}% "
              f"({ag['n_disagreements']} instruction disagreements)."]
    for k, v in model_agree.items():
        L.append(f"- {k}: instruction agreement {100*v['instruction_agreement']:.2f}%, "
                 f"prompt {100*v['prompt_agreement']:.2f}% ({v['n_disagreements']} disagreements)")
    d = rep["item_drift"]
    inr = rep["paper_within_unseeded_range"]
    L += ["", "Paper value inside the exact attainable range of the original scorer: " + ", ".join(f"{m} {'yes' if v else 'NO'}" for m, v in inr.items())
          + ". The official scorer is not deterministic. Main cause: `keywords:letter_frequency` with a non-letter "
          "(e.g. '#', '!') silently checks a random a-z letter drawn from Python `random`. We reseed `random` per item "
          "and seed langdetect for all model runs (`--no-seed` restores the original behaviour).",
          "", "Item list: canonical = authors' file at commit 066e1eda (2023-11-27), the list the paper's GPT-4 responses answer. "
          f"The authors' latest file differs on keys {d['canonical_vs_latest_github_changed_keys']}."]
    if "hf_vs_canonical_changed_keys" in d:
        L.append(f"HF {d['hf_dataset']} vs canonical: differ on keys {d['hf_vs_canonical_changed_keys'] or 'none (identical)'}.")
    (common.RESULTS / "scorer_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
