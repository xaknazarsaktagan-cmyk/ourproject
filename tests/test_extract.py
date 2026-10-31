import pytest

from app import extract
from app.schemas import Message, TriageRequest

RULES = {"1.1", "2.1", "2.2", "3.4", "5.1"}


@pytest.fixture(autouse=True)
def rules(monkeypatch):
    extract.reset_rule_cache()
    monkeypatch.setattr(extract, "_rule_index", lambda: {r: [r] for r in RULES})
    yield


def req(text="", attachments=0, raw=None, extra=None):
    msgs = [Message(role="player", text=text, attachments=attachments)]
    msgs += extra or []
    return TriageRequest(kind="complaint", messages=msgs, rule_refs_raw=raw)


def facts(text, **kw):
    r = req(text, **kw)
    return extract.extract_facts(r, text)


def test_nickname():
    assert facts("Игрок Arman_Petrov оскорбляет меня").nickname == "Arman_Petrov"
    assert facts("hacker John_Smith12 used aimbot").nickname == "John_Smith12"
    assert facts("ойыншы Aidos_Nurlan қорлады").nickname == "Aidos_Nurlan"
    assert facts("o'yinchi Timur_Karimov haqorat qildi").nickname == "Timur_Karimov"
    assert facts("никого не называю").nickname is None


def test_nickname_skips_placeholders_admins_and_self():
    assert facts("Name_Surname и Bad_Guy нарушают").nickname == "Bad_Guy"
    admin = Message(role="admin", text="Админ Boss_Admin ответил")
    r = req("Boss_Admin сказал, а Bad_Guy обманул", extra=[admin])
    assert extract.extract_facts(r, "Boss_Admin сказал, а Bad_Guy обманул").nickname == "Bad_Guy"
    assert facts("Меня зовут Good_Player, меня оскорбил Bad_Guy").nickname == "Bad_Guy"
    assert facts("my nick is Good_Player, Bad_Guy scammed me").nickname == "Bad_Guy"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("нарушил пункт 2.1", ["2.1"]),
        ("п.2.1 ОП", ["2.1"]),
        ("нарушение 2.1, 3.4", ["2.1", "3.4"]),
        ("ОП 2.1", ["2.1"]),
        ("OП 2.1", ["2.1"]),  # Latin O + Cyrillic П must be fixed by normalize in the pipeline
        ("rule 5.1 broken", ["5.1"]),
        ("2.1-тармақ бұзылды", ["2.1"]),
        ("qoida 3.4 buzildi", ["3.4"]),
        ("в 12.30 случилось", []),
        ("12.05.2024 был случай", []),
        ("версия 9.9 клиента", []),
    ],
)
def test_rule_refs_from_text(text, expected):
    refs = facts(text).rule_refs
    if text.startswith("OП"):  # pipeline normalizes homoglyphs first
        from app.normalize import normalize_text

        refs = extract.extract_facts(req(text), normalize_text(text, "ru")).rule_refs
    assert refs == expected


def test_rule_refs_raw_and_unknown():
    assert facts("", raw="2.1, 3.4").rule_refs == ["2.1", "3.4"]
    assert facts("", raw=["п.2.1 ОП", "5.1"]).rule_refs == ["2.1", "5.1"]
    refs, unknown = extract.extract_rule_refs_detail(req("пункт 9.9 и 2.1"), "пункт 9.9 и 2.1")
    assert refs == ["2.1"] and unknown == ["9.9"]


def test_no_rules_loaded_returns_nothing(monkeypatch):
    monkeypatch.setattr(extract, "_rule_index", lambda: {})
    assert facts("пункт 2.1").rule_refs == []


def test_prefixed_ids_are_resolved(monkeypatch):
    idx = {"2.1": ["ОП-2.1", "ПГС-2.1"], "3.4": ["ОП-3.4"]}
    monkeypatch.setattr(extract, "_rule_index", lambda: idx)
    assert facts("п. 3.4").rule_refs == ["ОП-3.4"]
    assert facts("ОП 2.1").rule_refs == ["ОП-2.1"]
    assert facts("п. 2.1").rule_refs == []  # ambiguous: never guess


@pytest.mark.parametrize(
    "text,expected",
    [
        ("есть скрин", True),
        ("записал видео", True),
        ("have a screenshot", True),
        ("бейне бар", True),
        ("skrinshot bor, video ham", True),
        ("доказательств нет", False),
        ("нет скрина", False),
        ("без видео", False),
        ("no screenshot", False),
        ("просто оскорбил", False),
    ],
)
def test_has_evidence_text(text, expected):
    assert facts(text).has_evidence is expected


def test_has_evidence_attachments():
    assert facts("оскорбил", attachments=2).has_evidence is True
    assert facts("оскорбил", attachments=0).has_evidence is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("на сервере 5 это было", "5"),
        ("server 12", "12"),
        ("сервер №7", "7"),
        ("server #3 lag", "3"),
        ("5 сервер", "5"),
        ("сервер Red нарушает", "Red"),
        ("сервер лагает", None),
        ("серверге кірмеймін", None),
        ("ничего", None),
    ],
)
def test_server(text, expected):
    assert extract.extract_server(text) == expected
