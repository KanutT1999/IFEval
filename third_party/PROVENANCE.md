# Vendored third-party code and data

## instruction_following_eval/  (the paper's own scorer and items)
- Source: https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval
- Files copied byte-for-byte: evaluation_lib.py, evaluation_main.py, instructions.py,
  instructions_registry.py, instructions_util.py, requirements.txt, README.md, data/input_data.jsonl,
  data/input_response_data_gpt4_20231107_145030.jsonl. Added: empty `__init__.py`. License: Apache-2.0.
- data/input_data_066e1eda_2023-11-27.jsonl: data/input_data.jsonl at commit
  066e1eda43f4785922e3994e95429e496080231f (2023-11-27). This is the item list the paper's GPT-4 responses
  answer and is byte-identical in content to HF `google/IFEval@966cd895`. **It is the canonical item list here.**
  The authors' later commit 26d8ccda (2024-06-11, "Fix an eval prompt") edited key 2785 only.
- SHA-256 checksums are enforced in ifeval_v100/common.py.

## lm_eval_ifeval/  (re-implementation, used only for the agreement check)
- Source: https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13/lm_eval/tasks/ifeval
- Only change: `from lm_eval.tasks.ifeval import X` -> `from lm_eval_ifeval import X` (sed), plus `__init__.py`. License: MIT.
