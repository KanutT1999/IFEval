"""Scoring with the authors' code (google-research/instruction_following_eval).

The official functions are called unmodified. The only thing this wrapper adds
is the option to seed langdetect (used by `language:response_language` and a
few other checks), which is otherwise random per call -> the official scorer is
not deterministic. Seeding is ON by default and recorded; `--no-seed-langdetect`
reproduces the original behaviour. See validate_scorer.py for the measurement.

Usage:
  python -m ifeval_v100.score --generations results/runs/<model>/<cond>/generations.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ifeval_v100 import common  # sets sys.path for third_party

import nltk


NLTK_DIR = common.ROOT / "nltk_data"   # repo-local, populated by setup.sh


def ensure_nltk() -> None:
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    for res, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, download_dir=str(NLTK_DIR), quiet=True)


def seed_langdetect(seed: int | None = 0) -> None:
    from langdetect import DetectorFactory
    DetectorFactory.seed = seed if seed is not None else None


def _official():
    from instruction_following_eval import evaluation_lib
    return evaluation_lib


def _lmeval():
    from lm_eval_ifeval import utils
    return utils


def score_one(item: dict, response: str, scorer: str = "official") -> dict:
    """Return strict/loose per-instruction booleans for one item."""
    if scorer == "official":
        ev = _official()
        inp = ev.InputExample(key=item["key"], instruction_id_list=item["instruction_id_list"],
                              prompt=item["prompt"], kwargs=item["kwargs"])
        p2r = {item["prompt"]: response}
        s = ev.test_instruction_following_strict(inp, p2r)
        l = ev.test_instruction_following_loose(inp, p2r)
    elif scorer == "lmeval":
        ut = _lmeval()
        inp = ut.InputExample(key=item["key"], instruction_id_list=item["instruction_id_list"],
                              prompt=item["prompt"], kwargs=item["kwargs"])
        s = ut.test_instruction_following_strict(inp, response)
        l = ut.test_instruction_following_loose(inp, response)
    else:
        raise ValueError(scorer)
    return {
        "key": item["key"],
        "instruction_id_list": item["instruction_id_list"],
        "strict": [bool(x) for x in s.follow_instruction_list],
        "loose": [bool(x) for x in l.follow_instruction_list],
        "prompt_strict": bool(s.follow_all_instructions),
        "prompt_loose": bool(l.follow_all_instructions),
    }


def score_responses(items: list[dict], responses: dict, scorer: str = "official",
                    langdetect_seed: int | None = 0) -> list[dict]:
    ensure_nltk()
    seed_langdetect(langdetect_seed)
    out = []
    for it in items:
        r = responses.get(it["key"])
        if r is None:
            raise KeyError(f"no response for key {it['key']}")
        out.append(score_one(it, r, scorer))
    return out


def metrics(per_item: list[dict]) -> dict:
    n_p = len(per_item)
    inst_s = [x for r in per_item for x in r["strict"]]
    inst_l = [x for r in per_item for x in r["loose"]]
    return {
        "prompt_strict": 100 * sum(r["prompt_strict"] for r in per_item) / n_p,
        "inst_strict": 100 * sum(inst_s) / len(inst_s),
        "prompt_loose": 100 * sum(r["prompt_loose"] for r in per_item) / n_p,
        "inst_loose": 100 * sum(inst_l) / len(inst_l),
        "n_prompts": n_p,
        "n_instructions": len(inst_s),
    }


def score_run_dir(run_dir: Path, scorer: str = "official", langdetect_seed: int | None = 0) -> dict:
    """Score generations.jsonl in a run dir -> per_item.jsonl + scores.json."""
    gens = common.read_jsonl(run_dir / "generations.jsonl")
    items = {it["key"]: it for it in common.load_items()}
    sel = [items[g["key"]] for g in gens]
    per_item = score_responses(sel, {g["key"]: g["response"] for g in gens}, scorer, langdetect_seed)
    m = metrics(per_item)
    m.update({"scorer": scorer, "langdetect_seed": langdetect_seed,
              "official_scorer_commit": common.OFFICIAL_SCORER_COMMIT})
    suffix = "" if scorer == "official" else f"_{scorer}"
    common.write_jsonl(run_dir / f"per_item{suffix}.jsonl", per_item)
    common.write_json(run_dir / f"scores{suffix}.json", m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, type=Path)
    ap.add_argument("--scorer", default="official", choices=["official", "lmeval"])
    ap.add_argument("--no-seed-langdetect", action="store_true")
    a = ap.parse_args()
    m = score_run_dir(a.generations.parent, a.scorer, None if a.no_seed_langdetect else 0)
    for k in ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose"):
        print(f"{k:14s} {m[k]:6.2f}")


if __name__ == "__main__":
    main()
