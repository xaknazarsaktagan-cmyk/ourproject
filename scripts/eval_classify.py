"""Step 5: honest evaluation on validation tickets that carry real category/priority.

Aggregates only (no ticket texts). Output: reports/classification.md.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from app import classify as cl
from app.normalize import detect_lang
from app.schemas import Message, TriageRequest
from scripts._data import norm_ticket_text, player_text, read_jsonl

LANGS = ("ru", "kk", "uz", "en")


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list[str]]:
    rows, total = [], 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            total += m.mean() * gap
            rows.append(f"| {lo:.1f}-{hi:.1f} | {int(m.sum())} | {conf[m].mean():.3f} | {correct[m].mean():.3f} |")
    return float(total), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickets", default="data/tickets_valid.jsonl")
    ap.add_argument("--out", default="reports/classification.md")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cl.load_resources()
    rec = []
    for t in read_jsonl(args.tickets):
        if "category" not in t:
            continue
        text = norm_ticket_text(t)
        lang = detect_lang(text, t.get("server_lang") or t.get("lang"))
        req = TriageRequest(kind=t.get("kind", "unknown"), messages=[Message(role="player", text=player_text(t)[:32000])])
        res = cl.classify(req, text, lang)
        rec.append((t["category"], res.category, t.get("priority"), res.priority, res.confidence, lang, res.is_spam))
        if args.limit and len(rec) >= args.limit:
            break
    if not rec:
        raise SystemExit("no validation tickets with a `category` field found")
    yc, pc, yp, pp, conf, lang, spam = (list(x) for x in zip(*rec))
    correct = np.array([a == b for a, b in zip(yc, pc)], dtype=float)
    conf = np.array(conf)
    lines = ["# Classification report (B)", "", f"Validation tickets: **{len(rec)}**. Model trained on train only; weak labels, valid used for evaluation only.", ""]
    lines += [f"- category accuracy: **{accuracy_score(yc, pc):.4f}**", f"- category macro-F1: **{f1_score(yc, pc, average='macro'):.4f}**"]
    pairs = [(a, b) for a, b in zip(yp, pp) if a]
    if pairs:
        ta, tb = zip(*pairs)
        lines.append(f"- priority macro-F1: **{f1_score(ta, tb, average='macro'):.4f}** (n={len(pairs)})")
        p0 = [(a, b) for a, b in pairs if a == "p0" or b == "p0"]
        tp = sum(a == "p0" and b == "p0" for a, b in p0)
        fn = sum(a == "p0" and b != "p0" for a, b in p0)
        fp = sum(a != "p0" and b == "p0" for a, b in p0)
        lines.append(f"- p0 recall: {tp / max(1, tp + fn):.3f}, precision: {tp / max(1, tp + fp):.3f} (support {tp + fn})")
    lines += ["", "## By language (detected)", "| lang | n | accuracy | macro-F1 |", "|---|---:|---:|---:|"]
    for lg in (*LANGS, "other"):
        ix = [i for i, x in enumerate(lang) if (x == lg if lg != "other" else x not in LANGS)]
        if ix:
            a, b = [yc[i] for i in ix], [pc[i] for i in ix]
            lines.append(f"| {lg} | {len(ix)} | {accuracy_score(a, b):.3f} | {f1_score(a, b, average='macro'):.3f} |")
    e, rows = ece(conf, correct)
    lines += ["", f"## Calibration (ECE = {e:.3f})", "| confidence bin | n | mean conf | accuracy |", "|---|---:|---:|---:|", *rows]
    lo, hi = conf < 0.5, conf >= 0.5
    if lo.any() and hi.any():
        lines += ["", f"Accuracy at confidence < 0.5: {correct[lo].mean():.3f} (n={int(lo.sum())}); >= 0.5: {correct[hi].mean():.3f} (n={int(hi.sum())})."]
    errs = Counter((a, b) for a, b in zip(yc, pc) if a != b)
    lines += ["", "## Top confusions (true -> predicted, counts)", "| true | predicted | n |", "|---|---|---:|"]
    lines += [f"| {a} | {b} | {n} |" for (a, b), n in errs.most_common(15)]
    lines += ["", f"Spam flagged: {sum(spam)} of {len(spam)}."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
