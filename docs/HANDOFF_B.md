# B: данные и классификация

## Что готово и проверено
- `app/normalize.py`: `normalize_text` (NFKC, cp1251-подмены казахских букв только при
  `server_lang == "kk"` или при однозначно казахских буквах в тексте, растяжки, гомоглифы
  внутри слов, `<nick>` не трогается), `detect_lang` (ru/kk/uz/en/es/pt/und, словари + буквы,
  hint со слабым весом, <3 слов -> hint или und).
- `app/extract.py`: ник `Имя_Фамилия` (без админских, без плейсхолдеров и «я — Nick_Name»),
  `has_evidence` (вложения или заявление; простые отрицания «нет скрина»), `server`,
  `rule_refs` (только существующие id; ссылки на несуществующие пункты возвращает
  `extract_rule_refs_detail(req, text) -> (refs, unknown_numbers)` для reasons: схему
  `Extracted` не менял, согласовать с A, если нужно поле в ответе).
- `app/classify.py`: `is_spam`/`spam_reason` (пустое, повторы символов, реклама, инвайты,
  ссылочный флуд, флуд, бессмыслица), `load_resources()`, `classify()` без падений.
- Тесты `tests/test_normalize.py`, `test_extract.py`, `test_classify.py` (75 кейсов).
- Скрипты: `profile_data` (Шаг 1), `build_weak_labels` (Шаг 4а/б), `train_classifier`
  (TF-IDF char 2-5 + word, LogReg balanced, калибровка), `eval_classify` (Шаг 5).

## Чего нет, потому что в архиве нет данных
В `first-player-main.zip` нет `data/`, `taxonomy.json`, `rules.json`. Поэтому НЕ сделано и
не измерено: сводка `reports/data_notes_B.md`, таблица `rule_to_category.json`, ключевые слова,
правила приоритета, обученная модель, `reports/classification.md`, замер согласия detect_lang
с полем `lang`. Скрипты проверены только на синтетических данных (метрики там ничего не значат).

## Как довести (когда появятся данные)
1. Положить `data/tickets_train.jsonl`, `data/tickets_valid.jsonl`, `data/taxonomy.json`, `data/rules.json`.
2. `make profile-data` -> `reports/data_notes_B.md`.
3. `python -m scripts.build_weak_labels --init-templates`, заполнить `app/config/*.json` вручную.
4. `make weak-labels && make train && make eval-classify`.
5. Проверить `reports/classification.md` (калибровка, p0, языки), поправить правила.
Если формат taxonomy/rules/тикетов отличается от предположений (поля `violation`, `category`,
`priority`, `ticket_id`), править `scripts/_data.py` и `build_weak_labels.py`.
