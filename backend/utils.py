"""Shared utilities: terminology normalization and document cache."""

import json
import re
import time
from pathlib import Path
from typing import Any

# --- Terminology normalization ---

DEFAULT_TERMINOLOGY_PATH = Path(__file__).resolve().parent.parent / "schema" / "terminology_map.json"


def load_terminology_map(path: Path | str | None = None) -> dict[str, list[str]]:
    """Load canonical -> synonyms from JSON. Returns dict; empty if file missing or invalid."""
    p = path if path is not None else DEFAULT_TERMINOLOGY_PATH
    p = Path(p)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): [str(s) for s in v] if isinstance(v, list) else [] for k, v in data.items()}


def _extract_quoted_placeholders(text: str) -> tuple[str, list[str]]:
    quoted: list[str] = []
    placeholder = "\x00QUOTE\x00"

    def repl(m: re.Match) -> str:
        quoted.append(m.group(0))
        return f"{placeholder}{len(quoted) - 1}{placeholder}"

    t = re.sub(r'"[^"]*"', repl, text)
    t = re.sub(r"'[^']*'", repl, t)
    return t, quoted


def _restore_quoted(text: str, quoted: list[str]) -> str:
    placeholder = "\x00QUOTE\x00"
    for i, q in enumerate(quoted):
        text = text.replace(f"{placeholder}{i}{placeholder}", q)
    return text


def normalize_text(text: str, terminology_map: dict[str, list[str]] | None = None) -> str:
    """Replace synonym phrases with canonical terms. Whole-phrase only; leaves quoted text unchanged."""
    if not text.strip():
        return text
    if terminology_map is None:
        terminology_map = load_terminology_map()
    if not terminology_map:
        return text
    work, quoted = _extract_quoted_placeholders(text)
    pairs: list[tuple[str, str]] = []
    for canonical, synonyms in terminology_map.items():
        for syn in synonyms:
            if syn.strip():
                pairs.append((syn.strip(), canonical))
    pairs.sort(key=lambda x: -len(x[0]))
    for synonym, canonical in pairs:
        pattern = r"(?<!\w)" + re.escape(synonym) + r"(?!\w)"
        work = re.sub(pattern, canonical, work, flags=re.IGNORECASE)
    return _restore_quoted(work, quoted)


# --- Document cache (TTL) ---

CACHE_TTL_SECONDS = 300
_cache: dict[str, tuple[float, Any]] = {}


def cache_get(key: str) -> Any | None:
    if key not in _cache:
        return None
    expiry, value = _cache[key]
    if time.time() > expiry:
        del _cache[key]
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.time() + CACHE_TTL_SECONDS, value)


def cache_invalidate(key: str) -> None:
    _cache.pop(key, None)


def cache_clear() -> None:
    _cache.clear()
