"""Step 1: aggregate-only data profile -> reports/data_notes_B.md (no player texts, no excerpts)."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from scripts._data import player_text, read_jsonl

MIN_SHOWN = 5  # rarer values are folded into "<5" so the report cannot identify single tickets


def table(counter: Counter, top: int = 30) -> list[str]:
    total = sum(counter.values()) or 1
    rows, rare = ["| value | count | share |", "|---|---:|---:|"], 0
    for value, n in counter.most_common():
        if n < MIN_SHOWN or len(rows) - 2 >= top:
            rare += n
        else:
            rows.append(f"| {value} | {n} | {n / total:.1%} |")
    if rare:
        rows.append(f"| other (<{MIN_SHOWN} or beyond top {top}) | {rare} | {rare / total:.1%} |")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/tickets_train.jsonl")
    ap.add_argument("--out", default="reports/data_notes_B.md")
    ap.add_argument("--fields", nargs="*", default=["kind", "lang", "server_lang", "verdict", "violation", "category", "priority"])
    args = ap.parse_args()
    counters = {f: Counter() for f in args.fields}
    lengths, empty, n, images = [], 0, 0, Counter()
    for t in read_jsonl(args.train):
        n += 1
        for f in args.fields:
            if f in t:
                v = t[f]
                counters[f][str(v) if not isinstance(v, (list, dict)) else "<structured>"] += 1
            else:
                counters[f]["<missing>"] += 1
        text = player_text(t)
        lengths.append(len(text))
        empty += not text.strip()
        images[min(int(t.get("images_count") or 0), 5)] += 1
    lines = [f"# Data notes (B)\n\nTickets: **{n}**. Aggregates only.\n"]
    if lengths:
        q = np.percentile(lengths, [5, 25, 50, 75, 95, 99])
        lines += [
            f"Empty player text: **{empty}** ({empty / n:.1%}).",
            "",
            "Player text length (chars) percentiles 5/25/50/75/95/99: "
            + " / ".join(f"{int(x)}" for x in q) + f"; max {max(lengths)}.",
            "",
            "## images_count (5 = 5+)",
            *table(images),
        ]
    for f, c in counters.items():
        if sum(c.values()) and set(c) != {"<missing>"}:
            lines += ["", f"## {f}", *table(c)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({n} tickets)")


if __name__ == "__main__":
    main()
