"""Step 4 (a, b): weak category labels for train tickets.

(a) ticket["violation"] -> rule number -> category through app/config/rule_to_category.json
(b) keyword rules per category from app/config/keyword_rules.json (only unambiguous matches)

Both config files are hand-checked tables. `--init-templates` creates empty skeletons from
taxonomy.json / rules.json; the script never invents a category name and drops any mapping
whose category is not in taxonomy.json.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from app.extract import _NUM_RE, _canon
from scripts._data import norm_ticket_text, read_jsonl, taxonomy_categories


def violation_numbers(value) -> list[str]:
    items = value if isinstance(value, list) else [value]
    nums = []
    for it in items:
        if isinstance(it, dict):
            it = it.get("id") or it.get("rule") or it.get("rule_id") or ""
        nums += [_canon(n) for n in re.findall(_NUM_RE, str(it))]
    return nums


def init_templates(taxonomy: str, rules: str | None, mapping: Path, keywords: Path) -> None:
    cats = taxonomy_categories(taxonomy)
    keywords.write_text(json.dumps({c: [] for c in cats}, ensure_ascii=False, indent=2), "utf-8")
    ids: list[str] = []
    if rules and Path(rules).is_file():
        data = json.loads(Path(rules).read_text(encoding="utf-8"))
        items = data.get("rules", data) if isinstance(data, dict) else data
        ids = [str(k) for k in items] if isinstance(items, dict) else [str(i["id"]) for i in items if "id" in i]
    mapping.write_text(json.dumps({i: None for i in ids}, ensure_ascii=False, indent=2), "utf-8")
    print(f"templates written: {mapping} ({len(ids)} rules), {keywords} ({len(cats)} categories)")
    print("Fill them in by hand from taxonomy.json / rules.json and review before training.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickets", default="data/tickets_train.jsonl")
    ap.add_argument("--taxonomy", default="data/taxonomy.json")
    ap.add_argument("--rules", default="data/rules.json")
    ap.add_argument("--mapping", default="app/config/rule_to_category.json")
    ap.add_argument("--keywords", default="app/config/keyword_rules.json")
    ap.add_argument("--out", default="data/weak_labels.jsonl")
    ap.add_argument("--init-templates", action="store_true")
    args = ap.parse_args()
    mapping_p, keywords_p = Path(args.mapping), Path(args.keywords)
    if args.init_templates:
        return init_templates(args.taxonomy, args.rules, mapping_p, keywords_p)
    cats = set(taxonomy_categories(args.taxonomy))
    raw_map = json.loads(mapping_p.read_text("utf-8")) if mapping_p.is_file() else {}
    rule_map = {}
    for key, cat in raw_map.items():
        if cat is None:
            continue
        if cat not in cats:
            raise SystemExit(f"mapping: category {cat!r} for rule {key!r} is not in taxonomy")
        m = re.search(_NUM_RE, key)
        rule_map[_canon(m.group(0)) if m else key] = cat
    kw = json.loads(keywords_p.read_text("utf-8")) if keywords_p.is_file() else {}
    unknown = [c for c in kw if c not in cats]
    if unknown:
        raise SystemExit(f"keyword_rules: categories not in taxonomy: {unknown}")
    kw_re = {c: re.compile("|".join(p), re.I | re.U) for c, p in kw.items() if p}
    stats, n = Counter(), 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for t in read_jsonl(args.tickets):
            n += 1
            cat, source = None, None
            cands = {rule_map[x] for x in violation_numbers(t.get("violation")) if x in rule_map}
            if len(cands) == 1:
                cat, source = cands.pop(), "rule"
            elif len(cands) > 1:
                stats["skipped_conflicting_rules"] += 1
            if cat is None and kw_re:
                text = norm_ticket_text(t)
                hits = {c for c, rx in kw_re.items() if rx.search(text)}
                if len(hits) == 1:
                    cat, source = hits.pop(), "keyword"
                elif len(hits) > 1:
                    stats["skipped_ambiguous_keywords"] += 1
            if cat is None:
                stats["unlabeled"] += 1
                continue
            stats[f"labeled_{source}"] += 1
            row = {"ticket_id": t.get("ticket_id"), "category": cat, "source": source}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"tickets={n}", dict(stats))


if __name__ == "__main__":
    main()
