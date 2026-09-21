from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def _locale_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here.parents[2] / "locale",
        here.parent / "locale",
        Path.cwd() / "locale",
    ]


class LanguageManager:
    _instance: LanguageManager | None = None

    def __init__(self) -> None:
        self._lang = "vi"
        self._strings: dict[str, dict[str, str]] = {"vi": {}, "en": {}}
        self._subscribers: list[Callable[[str], None]] = []
        self._load()

    @classmethod
    def instance(cls) -> LanguageManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        for lang in ("vi", "en"):
            for folder in _locale_dirs():
                path = folder / f"{lang}.json"
                if path.exists():
                    self._strings[lang] = json.loads(path.read_text(encoding="utf-8"))
                    break

    @property
    def language(self) -> str:
        return self._lang

    def t(self, key: str, **kwargs: object) -> str:
        table = self._strings.get(self._lang, {})
        fallback = self._strings.get("vi", {})
        text = table.get(key) or fallback.get(key) or key
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, IndexError):
                return text
        return text

    def set_language(self, lang: str) -> None:
        if lang not in ("vi", "en") or lang == self._lang:
            return
        self._lang = lang
        for callback in list(self._subscribers):
            callback(lang)

    def subscribe(self, callback: Callable[[str], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)


i18n = LanguageManager.instance()
