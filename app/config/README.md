# Конфиги B (заполняются по реальным taxonomy.json / rules.json)

- `rule_to_category.json` — {"<id пункта правил>": "<category из taxonomy>"}; шаблон:
  `python -m scripts.build_weak_labels --init-templates`. Проверить по смыслу вручную.
- `keyword_rules.json` — {"<category>": ["regex ru", "regex kk", ...]}; используется только для
  однозначных совпадений (если подходят две категории, тикет не размечается).
- `priority_rules.json` — [{"id": "...", "priority": "p0", "pattern": "regex", "categories": [...]}];
  критерии берутся из taxonomy.json (угрозы жизни, взлом, платежи и т.д.). Правило может только
  повысить приоритет относительно модели.

Названий категорий и приоритетов здесь намеренно нет: они не выдумываются, только из таксономии.
