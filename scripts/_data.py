"""Shared helpers for participant B scripts (ticket I/O, taxonomy discovery)."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from pathlib import Path

from app.normalize import normalize_text


def read_jsonl(path: str | Path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def player_text(ticket: dict) -> str:
    """Only the player's own messages are model input (admin replies leak the answer)."""
    return "\n".join(
        m.get("text") or "" for m in ticket.get("messages", []) if m.get("role") == "player"
    )


def norm_ticket_text(ticket: dict) -> str:
    return normalize_text(player_text(ticket)[:32000], ticket.get("server_lang"))


def taxonomy_categories(path: str | Path) -> list[str]:
    """Category ids from taxonomy.json; tolerant to list / dict / {"categories": ...} layouts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("categories", "category", "labels", "classes"):
            if key in data:
                data = data[key]
                break
    if isinstance(data, dict):
        return [str(k) for k in data]
    out = []
    for item in data:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for key in ("id", "key", "name", "slug", "category"):
                if key in item:
                    out.append(str(item[key]))
                    break
    return out


def find_label(categories: list[str], pattern: str) -> str | None:
    for c in categories:
        if re.search(pattern, c, re.IGNORECASE):
            return c
    return None
