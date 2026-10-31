import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from app import classify as cl
from app.schemas import Message, TriageRequest


def req(text):
    return TriageRequest(kind="report", messages=[Message(role="player", text=text)])


@pytest.fixture(autouse=True)
def clean():
    cl.reset_resources()
    yield
    cl.reset_resources()


@pytest.mark.parametrize(
    "text,reason",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("!!! ??? 123", "no_letters"),
        ("ааааааааааа", "repeated_chars"),
        ("купи дешево скидка t.me/shop_bot", "advert_link"),
        ("join discord.gg/abcdef", "invite_link"),
        ("a.com b.com c.com", "link_flood"),
        ("слово " * 20, "flood"),
        ("ksdjfhgk sdkjfhsdkjf qwrtpsdf", "gibberish"),
    ],
)
def test_spam_detected(text, reason):
    assert cl.spam_reason(text) == reason


@pytest.mark.parametrize(
    "text",
    [
        "Игрок Arman_Petrov оскорбляет меня, есть скрин",
        "доказательства https://youtube.com/watch?v=abc",  # evidence link is not spam
        "Мені ойыншы қорлады",
        "The player used wallhack on server 5",
        "help",
    ],
)
def test_legit_not_spam(text):
    assert cl.spam_reason(text) is None


def test_classify_safe_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(cl, "PRIORITY_RULES_PATH", tmp_path / "missing.json")
    res = cl.classify(req("Игрок оскорбляет меня"), "Игрок оскорбляет меня", "ru")
    assert (res.category, res.priority, res.confidence, res.is_spam) == ("unknown", "p3", 0.0, False)
    empty = cl.classify(req(""), "", "und")
    assert not empty.is_spam and empty.confidence == 0.0  # pipeline reports empty_text itself
    junk = cl.classify(req("!!!"), "!!! ???", "und")
    assert junk.is_spam and junk.confidence == 0.0


def test_classify_with_model_and_rules(tmp_path, monkeypatch):
    texts = ["оскорбил меня", "оскорбляет в чате", "взломал аккаунт", "украл аккаунт взлом"] * 5
    labels = ["cat_a", "cat_a", "cat_b", "cat_b"] * 5
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200))
    model.fit(texts, labels)
    path = tmp_path / "m.joblib"
    joblib.dump({"category": model, "priority": None, "meta": {"fallback_category": "cat_a"}}, path)
    rules = tmp_path / "r.json"
    rules.write_text(json.dumps([{"id": "t", "priority": "p0", "pattern": "угрож\\w+ убить"}]))
    monkeypatch.setattr(cl, "MODEL_PATH", path)
    monkeypatch.setattr(cl, "PRIORITY_RULES_PATH", rules)
    res = cl.classify(req("взломал аккаунт"), "взломал аккаунт", "ru")
    assert res.category == "cat_b" and 0 < res.confidence <= 1 and res.priority == "p3"
    res = cl.classify(req("x"), "угрожает убить, оскорбил меня", "ru")
    assert res.priority == "p0"


def test_classify_never_raises_on_broken_model(tmp_path, monkeypatch):
    class Broken:
        classes_ = ["a"]

        def predict_proba(self, X):
            raise RuntimeError("boom")

    cl._state.update(bundle={"category": Broken(), "meta": {"fallback_category": "other"}}, loaded=True)
    res = cl.classify(req("текст"), "обычный текст про игру", "ru")
    assert res.category == "other" and res.confidence == 0.0