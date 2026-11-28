"""Aggregate every completed run into the tables the brief asks for.

  python -m ifeval_v100.analyze                 # results/runs -> results/report/
  python -m ifeval_v100.analyze --runs-dir results/smoke

Outputs (results/report/):
  REPORT.md                 all tables + auto-filled claim sentences
  runs.csv                  one row per run: 4 metrics, 95% CIs, timing, truncation
  per_type.csv              inst-level strict accuracy per instruction type x model
  pairwise.csv              paired-bootstrap differences between models (headline)
  *.png                     scale plot, per-type heatmap, length histograms
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ifeval_v100 import common

METRICS = ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose")
B = 2000


# ---------------------------------------------------------------- loading
def load_runs(runs_dir: Path) -> dict:
    runs = {}
    for man in sorted(runs_dir.glob("*/*/manifest.json")):
        m = json.loads(man.read_text())
        if not m.get("complete"):
            continue
        d = man.parent
        per = common.read_jsonl(d / "per_item.jsonl")
        gens = common.read_jsonl(d / "generations.jsonl")
        runs[(m["model_key"], m["condition"])] = {"manifest": m, "per_item": per, "gens": gens}
    return runs


def arrays(per: list[dict]) -> dict:
    return {
        "ps": np.array([r["prompt_strict"] for r in per], float),
        "pl": np.array([r["prompt_loose"] for r in per], float),
        "is": np.array([sum(r["strict"]) for r in per], float),
        "il": np.array([sum(r["loose"]) for r in per], float),
        "ni": np.array([len(r["strict"]) for r in per], float),
    }


def point(a: dict, idx=None) -> dict:
    if idx is None:
        idx = slice(None)
    return {
        "prompt_strict": 100 * a["ps"][idx].mean(-1),
        "inst_strict": 100 * a["is"][idx].sum(-1) / a["ni"][idx].sum(-1),
        "prompt_loose": 100 * a["pl"][idx].mean(-1),
        "inst_loose": 100 * a["il"][idx].sum(-1) / a["ni"][idx].sum(-1),
    }


def boot_idx(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def with_ci(a: dict) -> dict:
    """Point estimate + 95% percentile bootstrap CI (resampling prompts; for
    instruction-level metrics this is a cluster bootstrap over prompts)."""
    n = len(a["ps"])
    idx = boot_idx(n)
    p, bs = point(a), point(a, idx)
    return {m: (float(p[m]), float(np.percentile(bs[m], 2.5)), float(np.percentile(bs[m], 97.5))) for m in METRICS}


def paired_diff(a: dict, b: dict, metric: str) -> tuple[float, float, float]:
    idx = boot_idx(len(a["ps"]), seed=1)
    d = point(a, idx)[metric] - point(b, idx)[metric]
    return float(point(a)[metric] - point(b)[metric]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt(t):
    return f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=common.RUNS)
    ap.add_argument("--out", type=Path, default=common.RESULTS / "report")
    a = ap.parse_args()
    cfg = common.load_config()
    runs = load_runs(a.runs_dir)
    if not runs:
        raise SystemExit(f"no completed runs under {a.runs_dir}")
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    H = cfg["headline_condition"]
    models = [m for m in cfg["models"] if any(k[0] == m for k in runs)]
    models.sort(key=lambda m: (cfg["models"][m]["family"], cfg["models"][m]["params_b"]))
    R = []  # report lines

    # ---------- per-run table
    rows, cache = [], {}
    for (mk, cond), r in sorted(runs.items()):
        arr = arrays(r["per_item"])
        cache[(mk, cond)] = arr
        ci = with_ci(arr)
        g = r["gens"]
        man = r["manifest"]
        row = {"model": mk, "condition": cond, "n": len(g)}
        for m in METRICS:
            row[m], row[m + "_lo"], row[m + "_hi"] = ci[m]
        row["truncation_rate"] = 100 * np.mean([x["truncated"] for x in g])
        row["mean_out_tokens"] = np.mean([x["n_output_tokens"] for x in g])
        row["gen_minutes"] = man["generation_seconds"] / 60
        row["gpu_hours_gen"] = man["generation_seconds"] / 3600
        row["load_seconds"] = man["load_seconds"]
        row["backend"] = man["backend"]["backend"]
        rows.append(row)
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out / "runs.csv", index=False)

    # ---------- 1. headline: four metrics with intervals
    R += ["# IFEval on small open-weight models — measured on one GPU", "",
          "Generated by `python -m ifeval_v100.analyze`. Every number traces to a run directory under "
          "`results/runs/<model>/<condition>/` (raw generations, exact final prompts, manifest).", "",
          f"**Headline metric: prompt-level strict accuracy** (the strictest, and the one the paper leads with). "
          f"Headline condition: `{H}` = model's own chat template, prompt verbatim as the single user turn, "
          f"greedy, fp16, max {cfg['generation']['max_new_tokens']} new tokens. Brackets: 95% bootstrap CI over the "
          f"541 prompts ({B} resamples; instruction-level metrics use a cluster bootstrap over prompts).", "",
          "## 1. All four metrics (headline condition)", ""]
    t = []
    for mk in models:
        if (mk, H) not in runs:
            continue
        ci = with_ci(cache[(mk, H)])
        man = runs[(mk, H)]["manifest"]
        t.append({"model": mk, "params (B)": cfg["models"][mk]["params_b"],
                  "revision": man["revision"][:10], **{m: fmt(ci[m]) for m in METRICS}})
    for name, p in cfg["paper"].items():
        t.append({"model": f"paper: {name}", "params (B)": "?", "revision": "API 2023",
                  **{m: f"{p[m]:.1f}" for m in METRICS}})
    R += [md_table(pd.DataFrame(t)), ""]

    # ---------- 2. spread: repeats, rewordings, sampling
    R += ["## 2. Spread — is a difference bigger than our own noise?", "",
          "Greedy repeats differ only in prompt submission order (batch composition → fp16 numerics). "
          "Rewordings are meaning-preserving wrapper changes (system prompt / prefix / label). "
          "Sampling is a deliberate variance experiment (T=%.1f, top-p=%.2f, 5 seeds), never the headline." %
          (cfg["generation"]["sampling"]["temperature"], cfg["generation"]["sampling"]["top_p"]), ""]
    spread_rows, spread = [], {}
    for mk in models:
        row = {"model": mk}
        for label, conds in [("greedy repeats", cfg["repeat_conditions"]),
                             ("rewordings", cfg["rewording_conditions"]),
                             ("sampling", cfg["sampling_conditions"])]:
            vals = [100 * cache[(mk, c)]["ps"].mean() for c in conds if (mk, c) in cache]
            if vals:
                row[label] = (f"{min(vals):.1f}–{max(vals):.1f} (n={len(vals)}"
                              + (f", sd {np.std(vals, ddof=1):.2f})" if len(vals) > 1 else ")"))
        allg = [c for c in dict.fromkeys(cfg["repeat_conditions"] + cfg["rewording_conditions"]) if (mk, c) in cache]
        vals = [100 * cache[(mk, c)]["ps"].mean() for c in allg]
        spread[mk] = (min(vals), max(vals), len(vals)) if vals else None
        if vals:
            row["all greedy runs: range (width)"] = f"{min(vals):.1f}–{max(vals):.1f} ({max(vals)-min(vals):.1f} pp, n={len(vals)})"
        spread_rows.append(row)
    R += ["Prompt-level strict, %:", "", md_table(pd.DataFrame(spread_rows)), ""]

    # ---------- 3. per-instruction-type breakdown
    R += ["## 3. Per-instruction-type breakdown (headline, instruction-level strict %)", ""]
    type_rows = defaultdict(dict)
    counts = {}
    for mk in models:
        if (mk, H) not in runs:
            continue
        acc = defaultdict(list)
        for r in runs[(mk, H)]["per_item"]:
            for iid, ok in zip(r["instruction_id_list"], r["strict"]):
                acc[iid].append(ok)
        for iid, v in acc.items():
            type_rows[iid][mk] = 100 * np.mean(v)
            counts[iid] = len(v)
    pt = pd.DataFrame(type_rows).T.sort_index()
    pt.insert(0, "n", [counts[i] for i in pt.index])
    pt.to_csv(out / "per_type.csv")
    show = pt.copy()
    for c in show.columns[1:]:
        show[c] = show[c].map(lambda v: f"{v:.0f}" if pd.notna(v) else "")
    show.insert(0, "instruction type", show.index)
    R += [md_table(show), "",
          f"{len(pt)} instruction types. Types where *every* model scores < 25% (collapse at this scale): "
          + (", ".join(f"`{i}`" for i in pt.index if (pt.loc[i].iloc[1:] < 25).all()) or "none") + ".",
          "Types where every model scores > 90% (no discrimination in this range): "
          + (", ".join(f"`{i}`" for i in pt.index if (pt.loc[i].iloc[1:] > 90).all()) or "none") + ".", ""]

    # ---------- 4. response length + truncation
    R += ["## 4. Response length and truncation (headline)", ""]
    lr = []
    for mk in models:
        if (mk, H) not in runs:
            continue
        g = runs[(mk, H)]["gens"]
        n = np.array([x["n_output_tokens"] for x in g])
        w = np.array([len(x["response"].split()) for x in g])
        lr.append({"model": mk, "median tokens": int(np.median(n)), "mean": f"{n.mean():.0f}",
                   "p90": int(np.percentile(n, 90)), "max": int(n.max()),
                   "median words": int(np.median(w)), "empty responses": int(sum(not x["response"].strip() for x in g)),
                   "truncated (hit max_new_tokens)": f"{100*np.mean([x['truncated'] for x in g]):.1f}%"})
    R += [md_table(pd.DataFrame(lr)), "", "Histogram: `length_hist.png`.", ""]

    # ---------- 5. compute
    R += ["## 5. GPU-hours", ""]
    cr = []
    for mk in models:
        sub = runs_df[runs_df.model == mk]
        h = sub[sub.condition == H]
        cr.append({"model": mk, "backend": sub.backend.iloc[0],
                   "GPU-min per full pass (headline, generation only)": f"{h.gen_minutes.iloc[0]:.1f}" if len(h) else "",
                   "GPU-hours per full pass": f"{h.gpu_hours_gen.iloc[0]:.3f}" if len(h) else "",
                   "model load (s)": f"{sub.load_seconds.mean():.0f}",
                   "runs": len(sub), "GPU-hours, all runs": f"{sub.gpu_hours_gen.sum():.2f}"})
    gpu = runs[next(iter(runs))]["manifest"]["env"].get("gpu_name", "CPU/unknown")
    R += [f"Device: {gpu}. Total generation GPU-hours across all runs: {runs_df.gpu_hours_gen.sum():.2f}.", "",
          md_table(pd.DataFrame(cr)), ""]

    # ---------- 6. gap to the paper
    R += ["## 6. Gap to the paper's reported figures (ours − paper, percentage points)", ""]
    gr = []
    for mk in models:
        if (mk, H) not in runs:
            continue
        p = point(cache[(mk, H)])
        for name, pp in cfg["paper"].items():
            gr.append({"model": mk, "vs": name,
                       **{m: f"{p[m]-pp[m]:+.1f}" for m in METRICS}})
    R += [md_table(pd.DataFrame(gr)), "",
          "Direction: negative = our small model scores below the paper's model.", "",
          "**This gap is confounded with** (we changed all of these at once): (1) different model family/weights, "
          "(2) scale (0.5–3B vs GPT-4 / PaLM 2 S, sizes undisclosed), (3) precision and hardware (fp16 on one local GPU vs. provider "
          "serving), (4) prompt template (each model's own chat template incl. any injected default system prompt, "
          "vs. API-side formatting), (5) decoding (greedy, 1280-token cap vs. unstated API settings), plus "
          "(6) time: the paper's responses were scraped in 2023 and the item list has since been edited "
          "(key 2785, June 2024; see scorer_validation.md). None of these can be attributed to size alone.", ""]

    # ---------- 7. does the benchmark discriminate in this range?
    R += ["## 7. Does IFEval discriminate between small models?", "",
          "Paired bootstrap over the same 541 prompts, prompt-level strict (row − column). "
          "A difference counts only if its CI excludes 0 **and** it exceeds both models' own greedy spread width.", ""]
    pr = []
    hm = [m for m in models if (m, H) in cache]
    for i, x in enumerate(hm):
        for y in hm[i + 1:]:
            d, lo, hi = paired_diff(cache[(x, H)], cache[(y, H)], "prompt_strict")
            sw = max((spread[x][1] - spread[x][0]) if spread.get(x) else 0,
                     (spread[y][1] - spread[y][0]) if spread.get(y) else 0)
            pr.append({"A": x, "B": y, "A−B (pp)": f"{d:+.1f}", "95% CI": f"[{lo:+.1f}, {hi:+.1f}]",
                       "max own spread (pp)": f"{sw:.1f}",
                       "real gap?": "yes" if (lo > 0 or hi < 0) and abs(d) > sw else "no"})
    pd.DataFrame(pr).to_csv(out / "pairwise.csv", index=False)
    R += [md_table(pd.DataFrame(pr)) if pr else "(needs ≥2 models)", ""]

    # ---------- 8. harness: template ablation + scorer
    R += ["## 8. Harness experiments", "", "### 8a. Prompt-template ablation (chat template vs raw prompt)", ""]
    ab = []
    for mk in models:
        if (mk, H) in cache and (mk, "greedy_no_template") in cache:
            for m in ("prompt_strict", "inst_strict"):
                d, lo, hi = paired_diff(cache[(mk, H)], cache[(mk, "greedy_no_template")], m)
                ab.append({"model": mk, "metric": m, "chat template − raw (pp)": f"{d:+.1f}",
                           "95% CI": f"[{lo:+.1f}, {hi:+.1f}]"})
    R += [md_table(pd.DataFrame(ab)) if ab else "(phase 3 not run)", ""]
    sv = common.RESULTS / "scorer_validation.md"
    R += ["### 8b. Scorer validation and the scorer-determinism fix", "",
          (sv.read_text().replace("# Scorer validation", "").strip() if sv.exists()
           else "(run `python -m ifeval_v100.validate_scorer`)"), ""]

    # ---------- 9. claim sentences
    R += ["## 9. Claims (fill-in-the-blanks from the brief, auto-filled)", ""]
    pg = cfg["paper"]["GPT-4"]["prompt_strict"]
    pp2 = cfg["paper"]["PaLM 2 S"]["prompt_strict"]
    for mk in models:
        if (mk, H) not in runs:
            continue
        man = runs[(mk, H)]["manifest"]
        x = point(cache[(mk, H)])["prompt_strict"]
        s = spread.get(mk)
        rng = f"range [{s[0]:.1f}, {s[1]:.1f}] over n={s[2]} greedy runs (repeats + rewordings)" if s else "n=1 run"
        R.append(f"- “{man['hf_id']} at revision {man['revision'][:10]}, {man['precision']} on a {gpu}, greedy, "
                 f"template = model's own chat template (`{H}`, final strings in `final_prompts.jsonl`), scored "
                 f"{x:.1f}% prompt-level strict ({rng}); the paper reports {pg:.1f}% (GPT-4) and {pp2:.1f}% "
                 f"(PaLM 2 S) for different and larger models. The difference is confounded with model family, "
                 f"scale, precision, chat template and decoding (§6).”")
    R += ["", "**Residual.** What this study accounts for: (a) scorer noise — the original scorer's random "
          "letter in 2 items moves a score by at most 2/541 = 0.37 pp prompt-level (§8b; removed here by seeding); "
          "(b) our own run-to-run spread (§2); (c) the chat-template effect on our models (§8a). What it does "
          "*not* account for: the gap to GPT-4 / PaLM 2 S itself. Model family, scale, serving precision/hardware, "
          "the API-side prompt format and decoding all differ at once (§6) and none was varied against the "
          "paper's models, so the whole gap beyond scorer noise and our spread remains unattributed.", ""]

    (out / "REPORT.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    plots(cfg, runs, cache, models, H, pt, spread, out)
    print(f"wrote {out/'REPORT.md'}")


def plots(cfg, runs, cache, models, H, pt, spread, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # scale plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fams = defaultdict(list)
    for mk in models:
        if (mk, H) in cache:
            fams[cfg["models"][mk]["family"]].append(mk)
    for fam, ms in fams.items():
        xs = [cfg["models"][m]["params_b"] for m in ms]
        cis = [with_ci(cache[(m, H)])["prompt_strict"] for m in ms]
        ys = [c[0] for c in cis]
        err = [[c[0] - c[1] for c in cis], [c[2] - c[0] for c in cis]]
        ax.errorbar(xs, ys, yerr=err, marker="o", capsize=3, label=fam)
        for m, xv in zip(ms, xs):
            if spread.get(m):
                ax.vlines(xv * 1.04, spread[m][0], spread[m][1], colors="gray", linewidth=4, alpha=0.4)
    for name, p in cfg["paper"].items():
        ax.axhline(p["prompt_strict"], ls="--", lw=1, color="k" if name == "GPT-4" else "gray")
        ax.text(0.45, p["prompt_strict"] + 1, f"paper: {name}", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (B, log)")
    ax.set_ylabel("prompt-level strict accuracy (%)")
    gpu = runs[next(iter(runs))]["manifest"]["env"].get("gpu_name", "CPU/unknown").replace("NVIDIA GeForce ", "")
    ax.set_title(f"IFEval, greedy, fp16, {gpu} (bars: 95% CI; grey: greedy-run spread)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "scale.png", dpi=150)
    plt.close(fig)

    # per-type heatmap
    if len(pt):
        data = pt.iloc[:, 1:].values.astype(float)
        fig, ax = plt.subplots(figsize=(max(9, 1.2 * data.shape[1] + 6), 0.32 * data.shape[0] + 1.5))
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(data.shape[1]), pt.columns[1:], rotation=30, ha="right")
        ax.set_yticks(range(data.shape[0]), [f"{i} (n={n})" for i, n in zip(pt.index, pt["n"])], fontsize=7)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if data[i, j] < 50 else "black")
        fig.colorbar(im, ax=ax, label="inst-level strict %")
        fig.tight_layout()
        fig.savefig(out / "per_type_heatmap.png", dpi=150)
        plt.close(fig)

    # length histograms
    fig, ax = plt.subplots(figsize=(7, 4))
    mx = cfg["generation"]["max_new_tokens"]
    for mk in models:
        if (mk, H) in runs:
            n = [x["n_output_tokens"] for x in runs[(mk, H)]["gens"]]
            ax.hist(n, bins=40, range=(0, mx + 10), histtype="step", label=mk)
    ax.axvline(mx, color="red", ls=":", label="max_new_tokens")
    ax.set_xlabel("output tokens")
    ax.set_ylabel("responses")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "length_hist.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
