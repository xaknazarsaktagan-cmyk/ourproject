"""Participant B: fact extraction (nickname, rule refs, evidence claim, server).

Rule ids are never invented: references are mapped onto ids that really exist (from
app.rag.all_rule_ids() and/or the rules file). `extract_rule_refs_detail` also returns the
references the player made to rules that do not exist, so A/D can add a reason hint without
changing the Extracted schema.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from app.schemas import Extracted, TriageRequest

# ----------------------------------------------------------------------------- nickname

_NICK_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Z][a-z]{1,19}(?:[A-Z][a-z]{1,9})?_[A-Z][a-z]{1,19}\d{0,3})(?![A-Za-z0-9_])")
_PLACEHOLDER_NICKS = {
    "name_surname", "nick_name", "player_name", "nickname_here", "firstname_lastname",
    "first_last", "your_nick", "ник_нейм",
}  # fmt: skip
_SELF_CTX_RE = re.compile(
    r"(?:\bя\b|меня зовут|мой (?:ник|никнейм)|менің (?:атым|никім)|менинг (?:исмим|никим)|"
    r"my (?:nick(?:name)?|name) is|\bi am\b|\bi'm\b|mening (?:nikim|ismim))\W{0,3}$",
    re.IGNORECASE,
)


def _nick_candidates(text: str) -> list[tuple[str, bool]]:
    out = []
    for m in _NICK_RE.finditer(text):
        nick = m.group(1)
        if nick.lower() in _PLACEHOLDER_NICKS:
            continue
        before = text[max(0, m.start() - 30) : m.start()]
        out.append((nick, bool(_SELF_CTX_RE.search(before))))
    return out


def extract_nickname(req: TriageRequest, norm_text: str) -> str | None:
    banned = set()
    for msg in req.messages:
        if msg.role == "admin":
            banned.update(n for n, _ in _nick_candidates(msg.text))
    cands = [(n, own) for n, own in _nick_candidates(norm_text) if n not in banned]
    for nick, own in cands:  # first nick that is not the reporter introducing themself
        if not own:
            return nick
    return None


# ----------------------------------------------------------------------------- rule refs

_NUM_RE = r"\d{1,2}(?:\.\d{1,2}){1,3}"
_MARKER = (
    r"(?:п\.?\s*п?\.?|пункт\w*|пп\.?|оп|ереже\w*|тармақ\w*|тарм\.?|rules?|points?|art(?:icle)?\.?|"
    r"clause|section|band\w*|qoida\w*|modda\w*|банд\w*|қоида\w*|модда\w*|§)"
)
_CODE_PREFIX = r"(?:оп|пгс|пп|ап|op|gp|pp)"
_MARKED_RE = re.compile(
    rf"(?:(?P<pre>{_MARKER})\s*[:№#\-]?\s*)(?P<nums>{_NUM_RE}(?:\s*(?:,|;|и|и/или|,\s*и|and|және|va|ва|/)\s*{_NUM_RE})*)\s*(?P<post>{_CODE_PREFIX})?",
    re.IGNORECASE,
)
_ANY_NUM_RE = re.compile(rf"(?<![\d.]){_NUM_RE}(?![\d.]|:\d)")
_DATE_LIKE_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.(?:\d{2}|\d{4})\b")


def _canon(num: str) -> str:
    return ".".join(str(int(p)) for p in num.split("."))


@lru_cache(maxsize=1)
def _rule_index() -> dict[str, list[str]]:
    """canonical number -> real rule ids carrying it; built from rag and/or the rules file."""
    ids: set[str] = set()
    try:
        from app import rag

        ids |= set(rag.all_rule_ids())
    except Exception:  # rag is another owner's module; never fail extraction because of it
        pass
    path = Path(os.getenv("RULES_PATH", "data/rules.json"))
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data.get("rules", data) if isinstance(data, dict) else data
            if isinstance(items, dict):
                ids |= {str(k) for k in items}
            else:
                ids |= {str(i["id"]) for i in items if isinstance(i, dict) and "id" in i}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    index: dict[str, list[str]] = {}
    for rid in ids:
        m = re.search(_NUM_RE, rid)
        if m:
            index.setdefault(_canon(m.group(0)), []).append(rid)
    return index


def reset_rule_cache() -> None:
    _rule_index.cache_clear()


def _resolve(num: str, prefix: str | None, index: dict[str, list[str]]) -> str | None:
    ids = index.get(num, [])
    if len(ids) == 1:
        return ids[0]
    if prefix:
        pref = [i for i in ids if i.lower().startswith(prefix.lower())]
        if len(pref) == 1:
            return pref[0]
    return None


def extract_rule_refs_detail(req: TriageRequest, norm_text: str) -> tuple[list[str], list[str]]:
    """Returns (existing rule ids, canonical numbers the player cited that do not exist)."""
    index = _rule_index()
    found: list[tuple[str, str | None, bool]] = []  # (number, code prefix, marked)
    cleaned = _DATE_LIKE_RE.sub(" ", norm_text)
    marked_spans = []
    for m in _MARKED_RE.finditer(cleaned):
        prefix = (m.group("pre") or "") + " " + (m.group("post") or "")
        code = re.search(_CODE_PREFIX, prefix, re.IGNORECASE)
        for n in re.findall(_NUM_RE, m.group("nums")):
            found.append((_canon(n), code.group(0) if code else None, True))
        marked_spans.append(m.span())
    for m in _ANY_NUM_RE.finditer(cleaned):  # unmarked numbers count only if they are real rules
        if not any(a <= m.start() < b for a, b in marked_spans):
            found.append((_canon(m.group(0)), None, False))
    raw = req.rule_refs_raw
    raw_items = [raw] if isinstance(raw, str) else list(raw or [])
    for item in raw_items:
        for n in re.findall(_NUM_RE, str(item)):
            code = re.search(_CODE_PREFIX, str(item), re.IGNORECASE)
            found.append((_canon(n), code.group(0) if code else None, True))
    refs: list[str] = []
    unknown: list[str] = []
    for num, prefix, marked in found:
        rid = _resolve(num, prefix, index) if index else None
        if rid:
            if rid not in refs:
                refs.append(rid)
        elif marked and num not in unknown:
            # Without any rules loaded we cannot tell "missing" from "unknown": report nothing.
            if index:
                unknown.append(num)
    return refs, unknown


# ----------------------------------------------------------------------------- evidence

_EVIDENCE_RE = re.compile(
    r"скрин\w*|скриншот\w*|видео\w*|запис[ьиюе]\w*|доказательств\w*|фото\w*|пруф\w*|"
    r"screens?hots?|screen\s?shots?|video\w*|record(?:ing|ed)?\b|clip\b|proofs?\b|evidence|"
    r"footage|photos?\b|"
    r"скриншот\w*|бейне\w*|жазба\w*|дәлел\w*|сурет\w*|"
    r"skrin\w*|skrinshot\w*|yozuv\w*|dalil\w*|isbot\w*|rasm\w*|"
    r"ёзув\w*|далил\w*|исбот\w*|расм\w*|"
    r"captura\w*|vídeo|prueba\w*|gravaç\w*|grabaci\w*|print\b|prova\w*",
    re.IGNORECASE,
)
_NEG_BEFORE_RE = re.compile(
    r"(?:нет|без|не\s+(?:\w+\s+)?(?:сделал|снял|записал|успел)|no|without|don't have|"
    r"жоқ|жок|yo'q|yoq|йўқ|sin|sem)\s*(?:\w+\s+){0,2}$",
    re.IGNORECASE,
)


_NEG_AFTER_RE = re.compile(
    r"^\W{0,3}(?:\w+\s+){0,1}(?:нет|отсутствуют|жоқ|жок|yo'q|yoq|йўқ|no|none|ausente)\b", re.IGNORECASE
)


def detect_evidence(req: TriageRequest, norm_text: str) -> bool:
    """True when the player attached files or claims to have evidence (a claim, not a proof)."""
    if any(m.attachments > 0 for m in req.messages if m.role != "admin"):
        return True
    for m in _EVIDENCE_RE.finditer(norm_text):
        before = norm_text[max(0, m.start() - 28) : m.start()]
        if not _NEG_BEFORE_RE.search(before) and not _NEG_AFTER_RE.search(norm_text[m.end() : m.end() + 20]):
            return True
    return False


# ----------------------------------------------------------------------------- server

_SERVER_WORD = r"(?:сервер\w*|server\w*|srv|сервер|servidor\w*)"
_SERVER_NUM_RE = re.compile(
    rf"{_SERVER_WORD}\s*(?:[№#:]|номер|no\.?|number|nomer|№\s*)?\s*[:\-]?\s*(\d{{1,3}})\b|"
    rf"\b(\d{{1,3}})\s*[-–]?\s*(?:й|ый|ой|-?й)?\s*{_SERVER_WORD}",
    re.IGNORECASE,
)
_SERVER_NAME_RE = re.compile(
    r"(?:сервер\w*|server|servidor)\s+(?:\"|«|')?([A-ZА-ЯЁӘҒҚҢӨҰҮҺ][\wА-Яа-яёәғқңөұүһ]{1,19})\b"
)
_NOT_SERVER_NAMES = {"Админ", "Admin", "Игрок", "Player", "Я", "Мне", "Не", "Это", "The"}


def extract_server(norm_text: str) -> str | None:
    m = _SERVER_NUM_RE.search(norm_text)
    if m:
        return m.group(1) or m.group(2)
    m = _SERVER_NAME_RE.search(norm_text)  # case-sensitive: names are capitalized
    if m and m.group(1) not in _NOT_SERVER_NAMES:
        return m.group(1)
    return None


# ----------------------------------------------------------------------------- entry point


def extract_facts(req: TriageRequest, norm_text: str) -> Extracted:
    refs, _unknown = extract_rule_refs_detail(req, norm_text)
    return Extracted(
        nickname=extract_nickname(req, norm_text),
        rule_refs=refs,
        has_evidence=detect_evidence(req, norm_text),
        server=extract_server(norm_text),
    )
