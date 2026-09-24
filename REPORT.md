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

Gold is the v6 tsawa layer: annotated spans from the OpenPecha layers, adjacent spans
separated only by punctuation merged, and a small list of unreliable spans removed
(`data/tsawa_gold_v6.csv`). Books come from two annotation batches: **old** (10 test books,
1,203 gold spans) and **new** (12 test books, 1,531 gold spans). The split is frozen; the
test split was used only for the final scores.

## 2. Metric

Per book, a prediction is a hit if its character IoU with a gold span is at least 0.5,
matched greedily best-first and one-to-one (offsets inclusive). True positives, predictions
and gold counts are summed over books (micro-average) to give P, R and F1. This is the same
matcher as the team's `v2_metrics`, applied to whole-book character spans. Token-level
figures reported elsewhere (per-window token IoU) are a different metric and are not
comparable to the numbers here.

## 3. Systems

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

The API settings are not identical (Claude used the JSON schema and prompt caching, which
the Gemini path did not), so the comparison holds prompt, windows, locator and scorer fixed,
not every API parameter.

**mmBERT v6, no features.** `jhu-clsp/mmBERT-base` fine-tuned for token tagging with three
labels (O / B-TSAWA / I-TSAWA), inverse-frequency class weights, trained on 8,192-token
windows with a 5,120 stride. Predictions are Viterbi-decoded (I may only follow B or I) with
a break penalty of 4.0, mapped back to characters and de-duplicated across windows. The
reported spans come from a GPU evaluation dump converted by `scripts/mmbert_dump_to_chars.py`.

**How the prompt was made.** Its numbers were measured on the train and validation books only:
70 % of tsawa spans are glossed by the prose that follows them (the *gloss test*, the primary
cue); 45 % have `ནི` within 30 characters before them and only 16 % sit right after a
closing `༽`; 40 % are followed by a closer such as `ཞེས་གསུངས`; 3.3 % are preceded by a
source marker such as `ལས།`; 59 % contain a line break; the median span is 96 characters and a
third are 45 characters or shorter. The prompt was checked on three validation books before
the test run, and the test books were not read while writing it. One version (v1) was
evaluated; there was no tuning against test.

## 4. Results

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

Claude beats Gemini on 14 books, Gemini beats Claude on 4, and 4 are within 0.005. Claude
beats mmBERT on 16 of 22.

## 5. Findings

1. **Zero-shot LLMs beat the trained model.** Claude Sonnet 5 is best on every row of the
   table, Gemini is second, and mmBERT third, with no training for either LLM. The gap
   between Claude and mmBERT is 0.08 on all books and 0.12 without the outlier book.
2. **Different error profiles.** mmBERT over-predicts (3,169 spans against 2,734 gold) and
   has the highest recall on 21 books (0.774) but the lowest precision (0.485). The LLMs
   return about 2,040 spans each, with precision 0.65–0.71. Claude's precision is highest of all
   (0.754 on new-batch) but its new-batch recall is low (0.410): it under-finds root text
   there.
3. **Old-batch books are easier for every system**, by 0.18 (Gemini 0.649 vs 0.465), 0.14
   (Claude 0.672 vs 0.531) and 0.11 (mmBERT 0.573 vs 0.466). A model that never saw the
   training data shows the same gap as the trained one, so the gap comes from the data and
   its annotation, not from the pipeline.
4. **One book dominates the failures.** `IF3ACC3E1` has 747 gold spans in 35,731 characters:
   the gold marks tiny fragments (a span every ~48 characters), while the LLMs return whole
   passages and mmBERT predicts none. Every system scores exactly 0.000 on it, and because it
   holds 27 % of all test gold it moves each headline number by about 0.08–0.11. Several
   validation books show the same style (for example `I895C519A`, median gold span 9
   characters). Other books that fail for all systems: `ICDC84458`, `P000242`, `P000144`,
   `IE5895799`.
5. **Gold errors are not what limits the scores.** Among consecutive gold spans split at
   `ཞེས་དང༌`, 30 pairs have a named source between them (for example `ཞེས་དང༌། མགོན་པོ་ཀླུས།`), which
   marks them as chained *quotations* mislabelled as tsawa; 21 are in train, 1 in validation
   and 8 in test (13 test spans, 0.5 % of test gold). They are listed in
   `data/gold_audit/named_source_pairs.csv`. Dropping them from gold and from predictions
   changes each system's score by about +0.002 (Gemini 0.556 → 0.558, Claude 0.603 → 0.605,
   mmBERT 0.521 → 0.522), so they do not explain the gaps between systems.

## 6. Caveats

- **One run per system.** Gemini ran at temperature 1.0 and Claude's sampling is fixed, so
  there is run-to-run noise; no confidence intervals were computed. With 22 books, book-level
  differences below about 0.05 should not be over-read.
- **Micro-averages are dominated by large books.** `IF3ACC3E1`, `P000067`, `P000164`,
  `I9B6A4525` and `P000242` together hold about 55 % of test gold; the median-book row is the
  fairer summary of typical behaviour.
- **mmBERT numbers come from a GPU run's dump.** The provided `mmbert_predict.py` reproduces
  the model from texts but is not bit-identical to that run on the two small books checked
  (F1 0.564 vs 0.591, and 0.891 vs 0.911); the cause was not resolved (not a bf16/fp32
  difference; the model config is fp32). The reported mmBERT break penalty is 4.0, while the
  training config default was 5.0.
- **Gold annotation quality varies by book**, especially fragment-style books (finding 4);
  some "false positives" on these books may be gaps or convention differences in the gold.
- **Data rights.** The texts are from OpenPecha; their terms apply to any redistribution.

## 7. Files

Scripts and layout are described in the README. Per-model outputs are in
`results/<model>/<split>/`: `spans/<book>.json` (predicted inclusive character offsets) and,
for the LLMs, `summary.json` (per-book P/R/F1 plus token counts). Raw window replies are not
included because they quote the book text.
