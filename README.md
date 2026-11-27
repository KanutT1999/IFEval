# IFEval on small open-weight models, one V100

An instrument for measuring small open-weight instruction models on **IFEval**
(Zhou et al. 2023, arXiv:2311.07911): 541 prompts, 25 verifiable instruction types,
scored by the authors' own code. The paper's models (GPT-4, PaLM 2 S) cannot be run here;
the goal is a measurement a stranger can re-run and a gap read honestly against Table 3.

## Re-run it

```bash
bash setup.sh          # pinned env (.venv), NLTK data, checksums, scorer validation (CPU)
bash smoke.sh          # 2-3 min GPU check: 20 items, smallest model
bash run_all.sh        # everything -> results/report/REPORT.md
```

Options: `MODEL_SET=v100_32gb bash run_all.sh` (adds Qwen2.5-7B), `MODEL_SET=open_only` (no gated
Llama), `PHASES="1"` (headline only), `BACKEND=hf` (transformers fallback). Runs are resumable.
Llama 3.2 is gated: accept the license on Hugging Face and `export HF_TOKEN=...` first.

## What is pinned, and where it is logged

| What | Pinned to | Logged in |
|---|---|---|
| Items | authors' `input_data.jsonl` @ 066e1eda (2023-11-27) = HF `google/IFEval@966cd895`, SHA-256 checked | `ifeval_v100/common.py`, `scorer_validation.md` |
| Scorer | authors' `instruction_following_eval` @ e6890f85, unmodified; langdetect seeded (see below) | each `scores.json` |
| Models | HF commit SHA per model (`configs/grid.yaml`) | each `manifest.json` |
| Tokenizer / template | same revision as model; chat-template SHA-256; Llama `date_string` pinned | `manifest.json` |
| Prompt path | exact final string per item, incl. any default system prompt the template injects | `final_prompts.jsonl` |
| Decoding | greedy, max 1280 new tokens, stop = tokenizer + generation_config EOS; model-repo sampling defaults are **not** applied | `manifest.json` |
| Precision | fp16 (V100 has no bf16) | `manifest.json` |
| Libraries | `requirements.txt` (vLLM 0.6.6.post1, torch 2.5.1, transformers 4.47.1) + full `pip freeze` | `results/env/` |
| Raw output | every response, token count, finish reason, truncation flag | `generations.jsonl` |
| Parser discards | none: the official scorer consumes the raw string; loose mode's 8 transformations are the authors' | `per_item.jsonl` (strict and loose per instruction) |

## Design

```
configs/grid.yaml          models, revisions, prompt variants, conditions, phases, paper Table 3
ifeval_v100/
  common.py                paths, checksums, env capture
  prompts.py               item -> messages -> final string -> token ids
  backends.py              vLLM (primary) and transformers (fallback) behind one interface
  run_model.py             load a model once, run conditions, write raw outputs, score
  score.py                 thin wrapper around the authors' evaluation_lib
  validate_scorer.py       reproduce Table 3 from the authors' GPT-4 responses; scorer determinism;
                           official vs lm-eval agreement; item-list drift
  analyze.py               CIs, spread, per-type table, lengths, GPU-hours, gap, pairwise, plots
  preflight.py             GPU check + pinned downloads before spending GPU time
third_party/               vendored scorer + data (see PROVENANCE.md)
```

**Conditions** (per model). Headline: `greedy_plain_r0`, the model's own chat template with the
prompt verbatim as the single user turn.

- **Repeats:** `greedy_plain_r1`, `r2` change only the prompt submission order, which changes batch
  makeup and so fp16 numerics.
- **Rewordings:** `sys_helpful`, `prefix_please` and `labelled` wrap the prompt differently without
  changing its meaning.
- **Template ablation:** `greedy_no_template` feeds the raw prompt with no chat template.
- **Sampling variance:** `sample_plain_s0..s4` (T=0.7, top-p=0.95, per-request seeds).

Phases run phase-major, so a budget cut still leaves every model with its headline.

**Intervals.** 95% percentile bootstrap over the 541 prompts (2000 resamples). Instruction-level
metrics use a cluster bootstrap over prompts. Model-vs-model differences use a paired bootstrap on
the same resamples. A difference counts as real only if its CI excludes 0 *and* it exceeds the
models' own greedy spread.

## Scorer findings so far (CPU, no GPU needed)

- **Table 3 is reproduced within scorer noise.** Scoring the authors' released GPT-4 responses with
  the authors' scorer, each of the paper's four numbers (76.89 / 83.57 / 79.30 / 85.37) appears among
  repeated unseeded scoring runs. The exact numbers per run are in `results/scorer_validation.md`.
- **The official scorer is not deterministic.** `langdetect` is unseeded, so repeated scoring of
  *identical* responses moves prompt-strict between 76.71 and 77.08. We seed it
  (`DetectorFactory.seed = 0`) for every model run and report the unseeded spread
  (`results/scorer_validation.md`). This is a harness fix, measured against the original.
- **lm-eval port vs official:** about 99.9% instruction-level agreement on GPT-4 responses (1–2
  disagreeing instructions out of 1,668 strict+loose checks); listed in `scorer_validation.json`.
- **Item drift:** the authors edited key 2785 in June 2024. HF and lm-eval still serve the
  2023-11-27 list, which we use; the difference is reported.
