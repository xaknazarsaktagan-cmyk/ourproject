"""Step 4: category (and optional priority) classifier. `make train`.

TF-IDF (char_wb 2-5 + word 1-2) -> LogisticRegression(class_weight=balanced), probability
calibration with CalibratedClassifierCV. Trains only on train tickets + weak labels; the
validation split is never read here, so scripts.eval_classify metrics are honest.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from app.classify import is_spam
from scripts._data import find_label, norm_ticket_text, read_jsonl, taxonomy_categories


def features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=300_000)),
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200_000)),
        ]
    )


def base_clf() -> LogisticRegression:
    return LogisticRegression(C=5.0, max_iter=400, class_weight="balanced")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickets", default="data/tickets_train.jsonl")
    ap.add_argument("--labels", default="data/weak_labels.jsonl")
    ap.add_argument("--taxonomy", default="data/taxonomy.json")
    ap.add_argument("--out", default="models/category.joblib")
    ap.add_argument("--report", default="reports/train_B.json")
    ap.add_argument("--min-class", type=int, default=5)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    labels = {r["ticket_id"]: r for r in read_jsonl(args.labels)}
    texts, cats, prios = [], [], []
    for t in read_jsonl(args.tickets):
        lab = labels.get(t.get("ticket_id"))
        if not lab:
            continue
        text = norm_ticket_text(t)
        if not text or is_spam(text):  # spam is handled by rules, not learned as a category
            continue
        texts.append(text)
        cats.append(lab["category"])
        prios.append(lab.get("priority"))
    counts = Counter(cats)
    keep = {c for c, n in counts.items() if n >= args.min_class}
    idx = [i for i, c in enumerate(cats) if c in keep]
    X, y = [texts[i] for i in idx], np.array([cats[i] for i in idx])
    if len(keep) < 2:
        raise SystemExit(f"need >=2 categories with >={args.min_class} examples, got {dict(counts)}")

    feats = features()
    t0 = time.time()
    # Honest CV macro-F1 of the uncalibrated model (features refit per fold inside the pipeline).
    cv = StratifiedKFold(n_splits=min(args.cv, min(Counter(y).values())), shuffle=True, random_state=0)
    cv_pred = cross_val_predict(Pipeline([("f", features()), ("c", base_clf())]), X, y, cv=cv, n_jobs=1)
    cv_f1 = float(f1_score(y, cv_pred, average="macro"))
    Xf = feats.fit_transform(X)
    cal_cv = max(2, min(3, min(Counter(y).values())))
    clf = CalibratedClassifierCV(base_clf(), method="sigmoid", cv=cal_cv).fit(Xf, y)
    cat_model = Pipeline([("f", feats), ("c", clf)])

    prio_model = None
    pr = [(X[k], prios[i]) for k, i in enumerate(idx) if prios[i]]
    if len({p for _, p in pr}) >= 2:
        prio_model = Pipeline([("f", features()), ("c", base_clf())]).fit(*zip(*pr))

    taxonomy = taxonomy_categories(args.taxonomy) if Path(args.taxonomy).is_file() else []
    meta = {
        "version": 1,
        "labels": sorted(keep),
        "spam_category": find_label(taxonomy, r"spam|спам"),
        "fallback_category": find_label(taxonomy, r"^(other|others|unknown|прочее|другое)$"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"category": cat_model, "priority": prio_model, "meta": meta}, args.out, compress=3)
    t1 = time.time()
    sample = X[: min(100, len(X))]
    s = time.time()
    for text in sample:
        cat_model.predict_proba([text])
    ms = (time.time() - s) / max(1, len(sample)) * 1000
    report = {
        "n_train": len(X),
        "dropped_rare_classes": {c: n for c, n in counts.items() if c not in keep},
        "class_counts": dict(Counter(y)),
        "cv_macro_f1_weak_labels": round(cv_f1, 4),
        "predict_ms_per_ticket": round(ms, 2),
        "train_seconds": round(t1 - t0, 1),
        "has_priority_head": prio_model is not None,
        "meta": meta,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "class_counts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
