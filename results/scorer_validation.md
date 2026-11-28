# Scorer validation

Authors' GPT-4 responses (2023-11-07), authors' item list at commit 066e1eda (2023-11-27),
authors' scorer at commit e6890f85.

| metric | paper Table 3 | official, seeded (ours) | official, original unseeded min–max (n=30) | exact attainable range | only `random` unseeded | only langdetect unseeded | lm-eval port, seeded |
|---|---|---|---|---|---|---|---|
| prompt_strict | 76.89 | 77.08 | 76.71–77.08 | 76.71–77.08 | 76.71–77.08 | 77.08–77.08 | 77.08 |
| inst_strict | 83.57 | 83.69 | 83.45–83.69 | 83.45–83.69 | 83.45–83.69 | 83.69–83.69 | 83.69 |
| prompt_loose | 79.30 | 79.67 | 79.48–79.67 | 79.30–79.67 | 79.30–79.67 | 79.67–79.67 | 79.67 |
| inst_loose | 85.37 | 85.61 | 85.49–85.61 | 85.37–85.61 | 85.37–85.61 | 85.61–85.61 | 85.61 |

Seeded scoring repeated 4x gives identical per-instruction results: yes.
Items whose score changes between unseeded runs: [1122, 1129] (from Python `random`: [1122, 1129]; from langdetect: []).
- key 1122: ['change_case:english_lowercase', 'keywords:letter_frequency'] kwargs=[{}, {'let_relation': 'at least', 'letter': '#', 'let_frequency': 4}]
- key 1129: ['keywords:letter_frequency', 'combination:repeat_prompt'] kwargs=[{'let_relation': 'at least', 'letter': '!', 'let_frequency': 6}, {'prompt_to_repeat': 'Write a short startup pitch for a new kind of ice cream called "Sunnis ice cream". The ice cream should be gentle on the stomach. Contain 6 or more exclamation marks "!" in your response.'}]

Official vs lm-eval port on GPT-4 responses (both seeded): instruction-level agreement 100.00%, prompt-level 100.00% (0 instruction disagreements).
- qwen2.5-0.5b/greedy_plain_r0: instruction agreement 100.00%, prompt 100.00% (0 disagreements)
- qwen2.5-1.5b/greedy_plain_r0: instruction agreement 100.00%, prompt 100.00% (0 disagreements)
- qwen2.5-3b/greedy_plain_r0: instruction agreement 100.00%, prompt 100.00% (0 disagreements)
- smollm2-1.7b/greedy_plain_r0: instruction agreement 100.00%, prompt 100.00% (0 disagreements)

Paper value inside the exact attainable range of the original scorer: prompt_strict yes, inst_strict yes, prompt_loose yes, inst_loose yes. The official scorer is not deterministic. Main cause: `keywords:letter_frequency` with a non-letter (e.g. '#', '!') silently checks a random a-z letter drawn from Python `random`. We reseed `random` per item and seed langdetect for all model runs (`--no-seed` restores the original behaviour).

Item list: canonical = authors' file at commit 066e1eda (2023-11-27), the list the paper's GPT-4 responses answer. The authors' latest file differs on keys [2785].
