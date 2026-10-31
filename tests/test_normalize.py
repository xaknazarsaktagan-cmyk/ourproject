import pytest

from app.normalize import detect_lang, normalize_text


def test_kk_repair_with_server_lang():
    assert normalize_text("мєн ќазаќша жазамын", "kk") == "мән қазақша жазамын"
    assert normalize_text("Ґалым Ќ", "kk") == "Ғалым Қ"


def test_kk_repair_by_strong_marker_without_server_lang():
    # contains unambiguous Kazakh letters (ә) -> substitutions are repaired
    assert normalize_text("мєн ќазаќша, бұл әдемі", None) == "мән қазақша, бұл әдемі"


def test_ukrainian_and_russian_untouched():
    assert normalize_text("Їжак є в лісі", "ru") == "Їжак є в лісі"
    assert normalize_text("Їжак є в лісі", None) == "Їжак є в лісі"
    # Uzbek Cyrillic uses ў legitimately
    assert normalize_text("ўйинчи", "uz") == "ўйинчи"


def test_nfkc_whitespace_and_stretches():
    assert normalize_text("  a\u00a0\u00a0b   c ", None) == "a b c"
    assert normalize_text("аааааааа", "ru") == "аа"
    assert normalize_text("нууууу пожалуйстааааа", "ru") == "нуу пожалуйстаа"
    assert normalize_text("ａｂｃ", None) == "abc"
    assert normalize_text("зачем!!!!!!!!", "ru") == "зачем!!!"


def test_numbers_not_collapsed():
    assert normalize_text("1000000 и 2.1", None) == "1000000 и 2.1"


def test_homoglyphs_in_russian_words():
    assert normalize_text("он оскорбил меня", None) == "он оскорбил меня"
    mixed = "oскорбил"  # Latin o + Cyrillic rest
    assert normalize_text(mixed, "ru") == "оскорбил"
    assert normalize_text("читep", "ru") == "читер"


def test_latin_nick_and_english_untouched():
    assert normalize_text("Ivan_Petrov hacked me", None) == "Ivan_Petrov hacked me"
    assert normalize_text("Alex Morgan", None) == "Alex Morgan"


def test_masked_tokens_kept():
    assert normalize_text("<nick> оскорбил <phone>", "ru") == "<nick> оскорбил <phone>"
    assert normalize_text("<nick>   cc", None) == "<nick> cc"


def test_empty_and_control_chars():
    assert normalize_text("", "ru") == ""
    assert normalize_text("a\x00b\u200bc", None) == "abc"


@pytest.mark.parametrize(
    "text,hint,expected",
    [
        ("Меня оскорбил игрок на сервере, помогите пожалуйста", None, "ru"),
        ("Мені ойыншы қорлады, көмектесіңіз өтінемін", None, "kk"),
        ("Ойыншы мені қорлады және ақшамды алды", "ru", "kk"),
        ("Menga o'yinchi haqorat qildi, iltimos yordam bering", None, "uz"),
        ("Мени ўйинчи ҳақорат қилди илтимос ёрдам беринг", None, "uz"),
        ("The player insulted me and I have a screenshot, please help", None, "en"),
        ("El jugador me insultó y no tengo pruebas, por favor ayuda", None, "es"),
        ("O jogador me insultou e eu não tenho prova, por favor ajuda", None, "pt"),
        ("ok", "kk", "kk"),
        ("ok", None, "und"),
        ("12345 !!!", None, "und"),
    ],
)
def test_detect_lang(text, hint, expected):
    assert detect_lang(text, hint) == expected


def test_detect_lang_hint_is_weak():
    # wrong data hint must not beat clear evidence
    assert detect_lang("The player insulted me, please help with the ban", "ru") == "en"
    assert detect_lang("Меня оскорбил игрок на сервере, помогите пожалуйста", "kk") == "ru"


def test_detect_lang_invalid_hint():
    assert detect_lang("hi", "xx") == "und"
