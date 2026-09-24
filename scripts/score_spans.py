#!/usr/bin/env python3
"""
score_spans.py - score predicted character spans against the v6 tsawa gold.

Needs only offsets (no book text): data/tsawa_gold_v6.csv, data/books_manifest.csv
and one span directory per model, each holding <book>.json = {"spans": [[s, e]]}
with INCLUSIVE character offsets.

Metric (same as the team's v2_metrics): per book, character IoU >= 0.5, greedy
best-first one-to-one matching; totals are micro-averaged over books.

    python scripts/score_spans.py --split test \
        --model gemini-3.1-flash-lite=results/gemini-3.1-flash-lite/test/spans \
        --model claude-sonnet-5=results/claude-sonnet-5/test/spans \
        --model mmbert-v6-nofeat=results/mmbert-v6-nofeat/test/spans

Rows: all books; without IF3ACC3E1 (a book whose gold is 747 tiny fragments in
36K characters, which no model scores on); old-batch; new-batch; median book.
--mask-audit additionally drops every gold span and prediction that overlaps a
suspect gold span from data/gold_audit/named_source_pairs.csv.
--per-book prints a per-book F1 table.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTLIER = "IF3ACC3E1"


def iou(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


def score(gold, pred, thr: float = 0.5) -> dict:
    cands = sorted(((iou(g, p), gi, pi) for gi, g in enumerate(gold)
                    for pi, p in enumerate(pred) if iou(g, p) >= thr), reverse=True)
    ug, up = set(), set()
    for _v, gi, pi in cands:
        if gi in ug or pi in up:
            continue
        ug.add(gi)
        up.add(pi)
    tp = len(ug)
    return {"tp": tp, "n_pred": len(pred), "n_gold": len(gold)}


def prf(tp: int, n_pred: int, n_gold: int):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def load_gold(split_books: set[str]):
    gold = {b: [] for b in split_books}
    for r in csv.DictReader(open(ROOT / "data/tsawa_gold_v6.csv", encoding="utf-8")):
        if r["pecha_id"] in gold:
            gold[r["pecha_id"]].append((int(r["start"]), int(r["end_exclusive"]) - 1))
    return {b: sorted(v) for b, v in gold.items()}


def load_suspects(split: str):
    sus: dict[str, list] = {}
    path = ROOT / "data/gold_audit/named_source_pairs.csv"
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["split"] != split or r["category"] == "outline_heading":
            continue
        for s, e in ((r["a_start"], r["a_end"]), (r["b_start"], r["b_end"])):
            sus.setdefault(r["pecha_id"], []).append((int(s), int(e) - 1))
    return sus


def overlaps(span, regions) -> bool:
    return any(span[0] <= e and s <= span[1] for s, e in regions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", action="append", required=True, metavar="NAME=SPAN_DIR")
    ap.add_argument("--mask-audit", action="store_true")
    ap.add_argument("--per-book", action="store_true")
    ap.add_argument("--json", default="", help="also write all numbers to this file")
    args = ap.parse_args()

    manifest = {r["pecha_id"]: r for r in csv.DictReader(
        open(ROOT / "data/books_manifest.csv", encoding="utf-8"))}
    books = sorted(b for b, r in manifest.items() if r["split"] == args.split)
    gold = load_gold(set(books))
    sus = load_suspects(args.split) if args.mask_audit else {}

    per: dict[str, dict[str, dict]] = {}
    for spec in args.model:
        name, d = spec.split("=", 1)
        per[name] = {}
        for b in books:
            f = Path(d) / f"{b}.json"
            if not f.is_file():
                raise SystemExit(f"{name}: missing {f}")
            pred = [tuple(x) for x in json.loads(f.read_text())["spans"]]
            g = gold[b]
            if sus.get(b):
                g = [x for x in g if not overlaps(x, sus[b])]
                pred = [x for x in pred if not overlaps(x, sus[b])]
            per[name][b] = score(g, pred)

    groups = [
        (f"all {len(books)} books", books),
        (f"without {OUTLIER}", [b for b in books if b != OUTLIER]),
        ("old-batch", [b for b in books if manifest[b]["source_batch"] == "old"]),
        ("new-batch", [b for b in books if manifest[b]["source_batch"] == "new"]),
    ]
    names = list(per)
    width = 26
    title = f"split={args.split}" + ("  (suspect gold spans masked)" if args.mask_audit else "")
    print(title)
    print(f"{'':20}" + "".join(f"{n:<{width}}" for n in names))
    out = {}
    for label, bs in groups:
        cells = []
        for n in names:
            tp = sum(per[n][b]["tp"] for b in bs)
            npd = sum(per[n][b]["n_pred"] for b in bs)
            ng = sum(per[n][b]["n_gold"] for b in bs)
            f1, p, r = prf(tp, npd, ng)
            out.setdefault(n, {})[label] = {"f1": f1, "precision": p, "recall": r,
                                            "tp": tp, "n_pred": npd, "n_gold": ng}
            cells.append(f"F1 {f1:.3f} P {p:.3f} R {r:.3f}")
        print(f"{label:20}" + "".join(f"{c:<{width}}" for c in cells))
    med = []
    for n in names:
        m = statistics.median(prf(per[n][b]["tp"], per[n][b]["n_pred"], per[n][b]["n_gold"])[0]
                              for b in books)
        out[n]["median_book_f1"] = m
        med.append(f"F1 {m:.3f}")
    print(f"{'median book':20}" + "".join(f"{c:<{width}}" for c in med))

    if args.per_book:
        print(f"\n{'book':11}{'batch':6}{'gold':>6}  " + "  ".join(f"{n[:18]:>18}" for n in names))
        order = sorted(books, key=lambda b: -prf(*[per[names[0]][b][k] for k in ("tp", "n_pred", "n_gold")])[0])
        for b in order:
            cells = [f"{prf(*[per[n][b][k] for k in ('tp', 'n_pred', 'n_gold')])[0]:>18.3f}" for n in names]
            print(f"{b:11}{manifest[b]['source_batch']:6}{per[names[0]][b]['n_gold']:>6}  " + "  ".join(cells))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
