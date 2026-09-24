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
| Old-batch books (10) | **0.672** | 0.649 | 0.573 |
| New-batch books (12) | **0.531** | 0.465 | 0.466 |
| Median book F1 | **0.767** | 0.703 | 0.693 |


**mmBERT** — the model is [`Yontenn/mmbert-tsawa-v6-nofeat`](https://huggingface.co/Yontenn/mmbert-tsawa-v6-nofeat)



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

- `data/tsawa_gold_v6.csv` _ gold tsawa spans (`pecha_id, start, end_exclusive`), 17,563 spans
  over 124 books, after the v6 cleaning.

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
