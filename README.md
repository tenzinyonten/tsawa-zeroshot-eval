# Tsawa (root-text) detection: zero-shot LLMs vs a fine-tuned mmBERT

Evaluation code, prompt and predictions for finding **tsawa** (རྩ་བ, the root text a
commentary quotes piece by piece and then explains) in classical Tibetan
commentaries. Three systems are scored on the same 22 held-out test books with the
same metric:

| System | How it works |
|---|---|
| **Claude Sonnet 5** (`claude-sonnet-5`) | zero-shot, one prompt, 16K-character windows |
| **Gemini 3.1 Flash Lite** (`gemini-3.1-flash-lite`) | zero-shot, same prompt and windows |
| **mmBERT v6 (no features)** | `jhu-clsp/mmBERT-base` fine-tuned for BIO token tagging, Viterbi-decoded |

## Results (test split, 22 books, 2,734 gold spans)

Character-level IoU ≥ 0.5, greedy one-to-one matching, micro-averaged over books.

| | Claude Sonnet 5 | Gemini 3.1 Flash Lite | mmBERT v6 |
|---|---:|---:|---:|
| **All 22 books** | **F1 0.603** (P .705 / R .526) | 0.556 (P .650 / R .486) | 0.521 (P .485 / R .562) |
| Without book `IF3ACC3E1` ¹ | **0.714** (P .705 / R .724) | 0.661 (P .653 / R .669) | 0.596 (P .485 / R .774) |
| Old-batch books (10) | **0.672** | 0.649 | 0.573 |
| New-batch books (12) | **0.531** | 0.465 | 0.466 |
| Median book F1 | **0.767** | 0.703 | 0.693 |

¹ `IF3ACC3E1` has 747 gold spans in 35,731 characters (one every ~48 characters, 27 % of all
test gold). All three systems score 0.000 on it, so the headline moves by about 0.1 on this
one book. See [REPORT.md](REPORT.md) for why, and for per-book numbers.

**In short:** the two zero-shot LLMs beat the trained model, and Claude leads on every row.
The LLMs are conservative (few false positives); mmBERT over-predicts (3,169 spans vs
2,734 gold). Old-batch books are easier than new-batch for every system.
Gemini on the 18 validation books scored 0.568 (used only to develop the prompt).

## Try it in two minutes (no texts, no API keys)

The shipped predictions are stored as character offsets, so scoring needs nothing else.

```bash
pip install -r requirements.txt

python scripts/score_spans.py --split test --per-book \
  --model claude-sonnet-5=results/claude-sonnet-5/test/spans \
  --model gemini-3.1-flash-lite=results/gemini-3.1-flash-lite/test/spans \
  --model mmbert-v6-nofeat=results/mmbert-v6-nofeat/test/spans
```

This reproduces the table above exactly. Add `--mask-audit` to drop the suspect gold
spans listed in `data/gold_audit/` (it changes each score by about +0.002).

## Re-running the systems

The book texts are **not** included (see [Data](#data)). Fetch them once:

```bash
python scripts/fetch_texts.py --ids-file data/ids.txt --raw-dir data/raw_opf \
    --manifest data/raw_opf/_manifest.csv
```

**Gemini / Claude** (keys are read from the environment and never written to disk):

```bash
export GEMINI_API_KEY=...        # or ANTHROPIC_API_KEY=...
python scripts/zeroshot_run.py --texts-dir data/raw_opf --books P000067 \
    --provider gemini --model gemini-3.1-flash-lite --out runs/gemini
python scripts/zeroshot_run.py --texts-dir data/raw_opf --books P000067 \
    --provider anthropic --model claude-sonnet-5 --out runs/claude
```

Raw replies are cached per window, so a re-run costs nothing for windows already done;
`--stop-after 1` pauses after each book so you can check the score before continuing.
The full 22-book test set is 361 windows; the Claude Sonnet 5 run cost about **$22** with
prompt caching (Gemini cost was not tracked). The score line printed at the end uses the
same metric as `score_spans.py`.

**mmBERT** — the model is [`Yontenn/mmbert-tsawa-v6-nofeat`](https://huggingface.co/Yontenn/mmbert-tsawa-v6-nofeat)
(if that page returns 404, the repository is still private: ask the owner for access).

```bash
python scripts/mmbert_predict.py --model Yontenn/mmbert-tsawa-v6-nofeat \
    --texts-dir data/raw_opf --books I319DAFF7 I9D9C7AC9 \
    --out results/mmbert-v6-nofeat/test --break-penalty 4.0
```

Use a GPU for the whole split (639 windows of 8,192 tokens; tens of seconds per window on
CPU). Note that on the two small books I checked, this script is close to but **not
bit-identical** with the GPU run behind the reported numbers (F1 0.564 vs 0.591 and
0.891 vs 0.911); the reported mmBERT figures come from that GPU run's dump, converted with
`scripts/mmbert_dump_to_chars.py`.

## Method in brief

- **Windows.** 16,000 characters, 2,000 overlap, each cut just after the nearest shad (།).
- **Output format.** The model does not return offsets or whole passages. It returns
  *anchors* `{label, frame, head, tail}` (~20 characters at each end; spans of 40
  characters or fewer go whole in `head`). A locator maps them back to offsets with a
  forward cursor plus one retry from the window start. Spans found twice in the overlap
  are de-duplicated at IoU ≥ 0.5.
- **Prompt** (`prompts/gemini_tsawa_anchors_v1.md`). Built from statistics measured on the
  train + validation books only; the test books were never read while writing it. Its main
  idea is the *gloss test*: if the commentary right after a passage repeats and explains
  the passage's own words, it is root text. A passage introduced by a named source
  (`<title>ལས།`, `ཇི་སྐད་དུ།`) is a quotation and is not marked.
- **Claude settings.** `max_tokens` 64000, adaptive thinking, effort `low`, JSON-schema
  output, 1-hour prompt cache. Sonnet 5 does not accept a temperature. **Gemini
  settings.** temperature 1.0, thinking level low, JSON output.
- **mmBERT.** 8,192-token windows with a 5,120 stride, three labels
  (O / B-TSAWA / I-TSAWA), inverse-frequency class weights, Viterbi decoding with a break
  penalty of 4.0.

## Data

- `data/tsawa_gold_v6.csv` — gold tsawa spans (`pecha_id, start, end_exclusive`), 17,563 spans
  over 124 books, after the v6 cleaning (adjacent spans separated only by punctuation merged,
  the listed ignore-spans removed).
- `data/books_manifest.csv` — split (84 train / 18 val / 22 test), source batch (`old`/`new`),
  length and gold count per book.
- `data/gold_audit/named_source_pairs.csv` — gold spans that look like mislabelled quotations
  (offsets only).
- **Book texts, tokenized datasets and raw model replies are deliberately not included.**
  The texts come from [OpenPecha](https://github.com/OpenPecha-Data) repositories; the fetch
  script's own notes say data-rights questions are still open, so please check the terms of
  those repositories before redistributing any text. The annotation offsets here are derived
  from the OpenPecha layers.

## Repository layout

```
README.md                         this file
REPORT.md                         full report: setup, per-book results, findings, caveats
prompts/gemini_tsawa_anchors_v1.md   the prompt used for both LLMs
scripts/
  score_spans.py                  scores span-offset files against gold (start here)
  zeroshot_run.py                 windows -> LLM -> locator -> scores (Gemini or Claude)
  mmbert_predict.py               run the mmBERT model straight from texts
  mmbert_dump_to_chars.py         convert the GPU token-span dump to character offsets
  fetch_texts.py                  clone the OpenPecha texts
src/                              window cutter, anchor locator, Viterbi decoder
data/                             gold, manifest, gold audit (no text)
results/                          predicted spans and summaries per model
```

`src/gemini_chunks.py` and `src/gemini_locate.py` are copied unchanged from the quotation
benchmark (Tibetan-quotation-detection) so both tasks share one windowing and one locator.
