"""Participant B: category / priority / spam classification.

Nothing in this module invents taxonomy labels. Category names come only from the trained model
bundle (models/category.joblib, built from taxonomy.json + weak labels by scripts.train_classifier)
and priority overrides only from app/config/priority_rules.json (derived from taxonomy criteria).
Without those files classify() degrades to the safe default (unknown, p3, confidence 0), while
spam detection works on its own.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from app.schemas import ClassResult, TriageRequest

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("CLASSIFIER_MODEL", "models/category.joblib"))
PRIORITY_RULES_PATH = Path(os.getenv("PRIORITY_RULES", "app/config/priority_rules.json"))
_PRIORITY_ORDER = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}

_state: dict[str, Any] = {"bundle": None, "rules": [], "loaded": False}
_lock = threading.Lock()

# ------------------------------------------------------------------------------------- spam

_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|ru|kz|uz|net|org|io|me|xyz|shop)\b/?\S*", re.I)
_INVITE_RE = re.compile(r"discord\.gg/|t\.me/|wa\.me/|vk\.cc/|bit\.ly/|tinyurl\.com/|@[\w]{4,}bot\b", re.I)
_AD_RE = re.compile(
    r"купи\w*|продам|продаю|скидк\w*|промокод\w*|подпишись|подписывай\w*|заработ\w*|казино|"
    r"ставк\w*|розыгрыш\w*|акци[яи]\b|бонус\w*|дёшево|дешево|быстрый заработок|"
    r"\bbuy\b|\bcheap\b|discount|promo\s?code|giveaway|casino|bonus|free\s+(?:money|gold|coins)|"
    r"click here|subscribe|earn\s+\$?\d+|"
    r"сатып ал\w*|жеңілдік\w*|тегін\s+(?:ақша|алтын)|"
    r"sotib ol\w*|arzon|chegirma\w*|bepul\s+(?:pul|oltin)|ставка|юқори даромад",
    re.I,
)
_VOWELS = set("aeiouyаеёиоуыэюяәөұүіiӘӨҰҮІ")
_CONSONANT_ONLY_MIN_LEN = 6


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def spam_reason(text: str) -> str | None:
    """Returns a reason code when the text looks like spam/flood/garbage, else None."""
    stripped = (text or "").strip()
    if not stripped:
        return "empty"
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return "no_letters"
    if len(stripped) >= 8 and len({c.lower() for c in letters}) <= 2:
        return "repeated_chars"
    urls = _URL_RE.findall(stripped)
    ad = len(_AD_RE.findall(stripped))
    if (urls or _INVITE_RE.search(stripped)) and ad:
        return "advert_link"
    if _INVITE_RE.search(stripped) and len(_tokens(stripped)) < 25:
        return "invite_link"
    if len(urls) >= 3:
        return "link_flood"
    if ad >= 2:
        return "advert"
    toks = _tokens(stripped)
    if len(toks) >= 10 and len(set(toks)) / len(toks) < 0.2:
        return "flood"
    if toks:
        mash = [t for t in toks if len(t) >= _CONSONANT_ONLY_MIN_LEN and not (set(t) & _VOWELS)]
        if mash and len(mash) / len(toks) > 0.5:
            return "gibberish"
    return None


def is_spam(text: str) -> bool:
    return spam_reason(text) is not None


# ----------------------------------------------------------------------------- resources


def _load_priority_rules(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rules = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        if item.get("priority") in _PRIORITY_ORDER and item.get("pattern"):
            rules.append(
                {
                    "id": item.get("id", ""),
                    "priority": item["priority"],
                    "re": re.compile(item["pattern"], re.IGNORECASE | re.UNICODE),
                    "categories": set(item.get("categories") or []),
                }
            )
    return rules


def _load_impl() -> None:
    with _lock:
        if _state["loaded"]:
            return
        try:
            _state["rules"] = _load_priority_rules(PRIORITY_RULES_PATH)
        except (OSError, ValueError, re.error) as exc:
            logger.warning("priority rules not loaded: %s", type(exc).__name__)
        if MODEL_PATH.is_file():
            import joblib  # local trusted artifact produced by scripts.train_classifier

            _state["bundle"] = joblib.load(MODEL_PATH)
        else:
            logger.warning("classifier model %s not found: safe defaults in use", MODEL_PATH)
        _state["loaded"] = True


def load_resources() -> None:
    """Loads the model bundle and priority rules once (called by the runtime at startup)."""
    _load_impl()


def reset_resources() -> None:
    with _lock:
        _state.update(bundle=None, rules=[], loaded=False)


# --------------------------------------------------------------------------- inference


def _rule_priority(text: str, category: str | None) -> str | None:
    best = None
    for rule in _state["rules"]:
        if rule["categories"] and category not in rule["categories"]:
            continue
        if rule["re"].search(text) and (
            best is None or _PRIORITY_ORDER[rule["priority"]] < _PRIORITY_ORDER[best]
        ):
            best = rule["priority"]
    return best


def predict_text(text: str) -> tuple[str, float, str | None]:
    """(category, calibrated confidence, model priority) for already-normalized text."""
    bundle = _state["bundle"]
    if not bundle:
        return "unknown", 0.0, None
    cat_model = bundle["category"]
    proba = cat_model.predict_proba([text])[0]
    top = int(proba.argmax())
    category, conf = str(cat_model.classes_[top]), float(proba[top])
    priority = None
    if bundle.get("priority") is not None:
        priority = str(bundle["priority"].predict([text])[0])
    return category, conf, priority


def classify(req: TriageRequest, norm_text: str, lang: str) -> ClassResult:
    if not _state["loaded"]:
        _load_impl()  # private on purpose: the runtime owns the public load_resources() call
    text = norm_text or ""
    reason = spam_reason(text)
    meta = (_state["bundle"] or {}).get("meta", {})
    if reason == "empty":
        # Empty input is reported by the pipeline itself (empty_text); flagging it as spam would
        # skip retrieval and hide stage failures.
        return ClassResult()
    if reason:
        # The spam label is used only when the taxonomy really contains one.
        return ClassResult(
            category=meta.get("spam_category") or "unknown",
            priority="p3",
            confidence=0.0 if reason == "no_letters" else 0.9,
            is_spam=True,
        )
    if not _state["bundle"]:
        return ClassResult(priority=_rule_priority(text, None) or "p3")  # type: ignore[arg-type]
    try:
        category, conf, model_priority = predict_text(text)
    except Exception as exc:  # never fail the pipeline on odd input
        logger.warning("classifier failed: %s", type(exc).__name__)
        return ClassResult(category=meta.get("fallback_category") or "unknown")
    priority = model_priority if model_priority in _PRIORITY_ORDER else "p3"
    forced = _rule_priority(text, category)
    if forced and _PRIORITY_ORDER[forced] < _PRIORITY_ORDER[priority]:
        priority = forced
    return ClassResult(
        category=category,
        priority=priority,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, conf)),
        is_spam=False,
    )