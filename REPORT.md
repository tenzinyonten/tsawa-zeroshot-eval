# Report: zero-shot LLMs vs fine-tuned mmBERT on tsawa detection

Test split, 22 books, 2,734 gold spans. Everything below can be reproduced from the files
in this repository (`python scripts/score_spans.py ...`, see the README).

## 1. Task and data

**Tsawa** (རྩ་བ) is the root text of a commentary: the verse or prose the author lifts out
piece by piece so that the commentary can explain it. The task is to mark those passages,
by character offsets, in whole books. Quotations from *other* works are not tsawa, and
telling the two apart is the main difficulty.

| Split | Books | Gold spans |
|---|---:|---:|
| train | 84 | 13,129 |
| validation | 18 | 1,700 |
| **test** | **22** | **2,734** |



**LLMs, zero-shot.** One prompt (`prompts/gemini_tsawa_anchors_v1.md`) for both models.
Each book is cut into 16,000-character windows with 2,000 overlap, each window cut just
after the nearest shad. The model returns anchors (`label`, optional `frame`, `head`,
`tail`, about 20 characters at each end; spans of 40 characters or fewer go whole in
`head`), never offsets. A locator finds the anchors in the window (forward cursor, then one
retry from the window start, head-to-tail at most 2,000 characters), and duplicates found
in the overlap are removed at IoU ≥ 0.5. The 22 test books are 361 windows.

| | Claude Sonnet 5 | Gemini 3.1 Flash Lite |
|---|---|---|
| Model id | `claude-sonnet-5` | `gemini-3.1-flash-lite` |
| Output | JSON schema (`label`, `head`, `tail` required; `frame` optional) | JSON MIME type |
| Thinking | adaptive, effort `low` | thinking level `low` |
| Sampling | Sonnet 5 accepts no temperature | temperature 1.0 |
| `max_tokens` | 64,000 (streamed) | default |
| Caching | 1-hour prefix cache on the prompt | none |

## Results

### Test split

| | Claude Sonnet 5 | Gemini 3.1 Flash Lite | mmBERT v6 |
|---|---:|---:|---:|
| **All 22 books** | **0.603** (P .705, R .526) | 0.556 (P .650, R .486) | 0.521 (P .485, R .562) |
| Without `IF3ACC3E1` | **0.714** (P .705, R .724) | 0.661 (P .653, R .669) | 0.596 (P .485, R .774) |
| Old-batch (10 books) | **0.672** (P .672, R .673) | 0.649 (P .658, R .639) | 0.573 (P .476, R .720) |
| New-batch (12 books) | **0.531** (P .754, R .410) | 0.465 (P .639, R .366) | 0.466 (P .497, R .438) |
| Median book | **0.767** | 0.703 | 0.693 |
| Predicted spans (gold 2,734) | 2,039 | 2,044 | 3,169 |

Validation (18 books, 1,700 gold), Gemini only: F1 0.568 (P .509, R .642). Claude and mmBERT
were scored on test only.

### Per book (F1, test split)

| Book | Batch | Gold | Claude Sonnet 5 | Gemini 3.1 Flash Lite | mmBERT v6 |
|---|---|---:|---:|---:|---:|
| `P000118` | old | 52 | 0.990 | 0.917 | 0.945 |
| `P000056` | old | 108 | 0.986 | 0.991 | 0.977 |
| `I9B6A4525` | new | 200 | 0.978 | 0.940 | 0.786 |
| `I575514A8` | new | 33 | 0.969 | 0.952 | 0.815 |
| `P000067` | old | 207 | 0.967 | 0.933 | 0.950 |
| `I9D9C7AC9` | new | 43 | 0.965 | 0.941 | 0.911 |
| `IC05A6BE0` | new | 166 | 0.963 | 0.748 | 0.894 |
| `P000164` | old | 183 | 0.950 | 0.956 | 0.917 |
| `P000013` | old | 89 | 0.903 | 0.798 | 0.892 |
| `I319DAFF7` | new | 17 | 0.828 | 0.828 | 0.591 |
| `I9AEEF96A` | new | 19 | 0.800 | 0.919 | 0.667 |
| `I0FCFA88F` | new | 144 | 0.734 | 0.659 | 0.749 |
| `I3F4A91F5` | new | 58 | 0.685 | 0.628 | 0.487 |
| `P000027` | old | 51 | 0.566 | 0.641 | 0.718 |
| `P000083` | old | 190 | 0.512 | 0.503 | 0.434 |
| `IC6F06BCD` | new | 14 | 0.488 | 0.333 | 0.324 |
| `IE5895799` | new | 29 | 0.309 | 0.216 | 0.257 |
| `P000144` | old | 113 | 0.249 | 0.230 | 0.245 |
| `P000242` | old | 171 | 0.229 | 0.201 | 0.146 |
| `ICDC84458` | new | 61 | 0.194 | 0.191 | 0.217 |
| `P000269` | old | 39 | 0.022 | 0.136 | 0.169 |
| `IF3ACC3E1` | new | 747 | 0.000 | 0.000 | 0.000 |


