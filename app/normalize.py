"""Participant B: text normalization and offline language detection.

normalize_text -- NFKC, cp1251 Kazakh-letter repair, stretches, Latin/Cyrillic homoglyphs.
detect_lang    -- script + marker letters + frequent-word dictionaries, no network.
"""

from __future__ import annotations

import re
import unicodedata

SUPPORTED_LANGS = {"ru", "kk", "uz", "en", "es", "pt"}

# cp1251-style substitutions for Kazakh letters (lower, upper).
_KK_FIX = str.maketrans(
    {
        "ќ": "қ", "ґ": "ғ", "ѕ": "ң", "є": "ә", "ј": "ө", "ў": "ұ", "ї": "ү",
        "Ќ": "Қ", "Ґ": "Ғ", "Ѕ": "Ң", "Є": "Ә", "Ј": "Ө", "Ў": "Ұ", "Ї": "Ү",
    }
)  # fmt: skip
_KK_FIX_CHARS = set("ќґѕєјўїЌҐЅЄЈЎЇ")
# Letters that exist in Kazakh but in neither Russian nor Ukrainian: safe evidence of Kazakh text.
_KK_STRONG = set("әғқңөұүһӘҒҚҢӨҰҮҺ")
_MASK_RE = re.compile(r"<[^<>\s]{1,40}>")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_WS_RE = re.compile(r"[^\S\n]+")
_NL_RE = re.compile(r"\n{3,}")
_STRETCH_RE = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)
_PUNCT_RUN_RE = re.compile(r"([!?.,\-*#~=_])\1{3,}")

_LAT2CYR = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "k": "к",
    "m": "м", "t": "т", "h": "н", "b": "в",
    "A": "А", "C": "С", "E": "Е", "O": "О", "P": "Р", "X": "Х", "Y": "У", "K": "К",
    "M": "М", "T": "Т", "H": "Н", "B": "В",
}  # fmt: skip
_CYR2LAT = {v: k for k, v in _LAT2CYR.items()}


def _is_cyr(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04ff"


def _is_lat(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _fix_homoglyphs(word: str) -> str:
    cyr = sum(_is_cyr(c) for c in word)
    lat = sum(_is_lat(c) for c in word)
    if not cyr or not lat:
        return word
    if cyr >= lat:  # Cyrillic word with Latin lookalikes
        return "".join(_LAT2CYR.get(c, c) if _is_lat(c) else c for c in word)
    # Latin word with Cyrillic lookalikes: only when every Cyrillic char has a Latin twin.
    if all(c in _CYR2LAT for c in word if _is_cyr(c)):
        return "".join(_CYR2LAT.get(c, c) if _is_cyr(c) else c for c in word)
    return word


def _kk_repair_needed(text: str, server_lang: str | None) -> bool:
    if not any(c in _KK_FIX_CHARS for c in text):
        return False
    if (server_lang or "").lower() == "kk":
        return True
    # Elsewhere only with unambiguous Kazakh letters present (ru/uk may legally contain є, ї, ј;
    # Uzbek Cyrillic legally uses ў, so uz alone is not a trigger).
    return any(c in _KK_STRONG for c in text)


def _normalize_segment(seg: str) -> str:
    seg = _STRETCH_RE.sub(r"\1\1", seg)
    seg = _PUNCT_RUN_RE.sub(lambda m: m.group(1) * 3, seg)
    return _WORD_RE.sub(lambda m: _fix_homoglyphs(m.group(0)), seg)


def normalize_text(text: str, server_lang: str | None = None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")
    if _kk_repair_needed(text, server_lang):
        text = text.translate(_KK_FIX)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    # Masked tokens like <nick> are kept verbatim.
    out, pos = [], 0
    for m in _MASK_RE.finditer(text):
        out.append(_normalize_segment(text[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(_normalize_segment(text[pos:]))
    return "".join(out).strip()


# --------------------------------------------------------------------------- language detection

_STOP = {
    "ru": "и в не на что я он она это как мне меня мой моя его ее их но с по за из у от для так вы "
    "мы был была было были есть нет да уже еще очень игрок игрока админ сервер сервере пожалуйста "
    "помогите нарушил нарушает оскорбил оскорбляет читы чит бан забанили скрин видео аккаунт "
    "деньги тебя тебе ты они когда если почему можно надо будет пишу прошу".split(),
    "kk": "мен сен ол біз сіз олар бұл сол үшін және емес бар жоқ деп болды болып маған ойыншы "
    "ойыншыны өтінемін көмектесіңіз бұзды қорлады ақша да де туралы бойынша кейін қазір әлі өте "
    "жақсы екен сияқты неге қалай".split(),
    "uz": "men sen u biz siz ular bu shu uchun va emas bor yoq yo'q deb bo'ldi menga o'yinchi "
    "iltimos yordam buzdi haqorat qildi akkaunt pul ham lekin juda yaxshi nega qanday endi hali "
    "bilan мен сен у биз сиз улар бу шу учун ва эмас бор йўқ деб бўлди менга ўйинчи илтимос ёрдам "
    "бузди ҳақорат қилди ҳам лекин жуда яхши нега қандай энди ҳали билан".split(),
    "en": "the a an is are was were be been and or but to of in on at for with from by this that "
    "it he she they we you i my me his her their our your not no yes please help player admin "
    "hacker cheater cheating banned ban account money scammed insulted rule rules report has have "
    "had does did what when why how can could would should".split(),
    "es": "el la los las un una unos unas que de y en es por con para mi me se su sus no si pero "
    "como esta este eso muy jugador administrador servidor favor ayuda cuenta dinero hackeo "
    "insultó baneado tengo tiene hay fue era".split(),
    "pt": "o a os as um uma uns umas que de e em é por com para meu minha me se seu sua não sim mas "
    "como está este isso muito jogador administrador servidor favor ajuda conta dinheiro "
    "banido tenho tem há foi era você".split(),
}
_STOP_SETS = {k: set(v) for k, v in _STOP.items()}
_CYR_LANGS = {"ru", "kk"}
_LAT_LANGS = {"en", "es", "pt"}

_KK_ONLY = set("әөұүңһӘӨҰҮҢҺ")  # shared қ/ғ belong to both kk and uz
_UZ_CYR_MARK = set("ўҳЎҲ")
_UZ_LAT_RE = re.compile(r"[og][ʻʼ’‘`´']", re.IGNORECASE)
_ES_MARK = set("ñ¿¡Ñ")
_PT_MARK = set("ãõÃÕ")
_APOS_RE = re.compile(r"[ʻʼ’‘`´]")


def _tokens(text: str) -> list[str]:
    text = _APOS_RE.sub("'", text.lower())
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text, flags=re.UNICODE)


def detect_lang(text: str, hint: str | None = None) -> str:
    hint = hint if hint in SUPPORTED_LANGS else None
    text = _MASK_RE.sub(" ", text or "")
    toks = _tokens(text)
    if len(toks) < 3:
        return hint or "und"
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return hint or "und"
    cyr = sum(_is_cyr(c) for c in letters)
    lat = sum(_is_lat(c) for c in letters)
    score = dict.fromkeys(SUPPORTED_LANGS, 0.0)
    for tok in toks:
        is_cyr_tok = _is_cyr(tok[0])
        for lang, words in _STOP_SETS.items():
            if tok in words:
                if lang in _CYR_LANGS and not is_cyr_tok:
                    continue
                if lang in _LAT_LANGS and is_cyr_tok:
                    continue
                score[lang] += 1.0
    kk_only = sum(c in _KK_ONLY for c in text)
    kk_shared = sum(c in "қғҚҒ" for c in text)
    uz_cyr = sum(c in _UZ_CYR_MARK for c in text)
    if cyr >= lat:
        for lang in _LAT_LANGS:
            score[lang] = 0.0
        score["kk"] += 2.0 * min(kk_only, 3) + (1.0 if kk_shared and not uz_cyr else 0.0)
        score["uz"] += 2.5 * min(uz_cyr, 3)
        score["ru"] += 0.3  # Cyrillic default prior
    else:
        for lang in _CYR_LANGS:
            score[lang] = 0.0
        score["uz"] += 2.0 * min(len(_UZ_LAT_RE.findall(text)), 3)
        score["es"] += 2.0 * min(sum(c in _ES_MARK for c in text), 3)
        score["pt"] += 2.0 * min(sum(c in _PT_MARK for c in text), 3)
        score["en"] += 0.3
    if hint:
        score[hint] += 0.75  # weak prior: the data field is heuristic and sometimes wrong
    best = max(score, key=lambda k: score[k])
    if score[best] < 1.0:
        return hint or "und"
    return best
