from __future__ import annotations

from dataclasses import dataclass

from inferbench.engine.batcher import DynamicBatcher
from inferbench.engine.model_runner import ModelRunner


@dataclass
class ModelEntry:
    name: str
    runner: ModelRunner
    batcher: DynamicBatcher | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._default: str | None = None

    def register(
        self,
        name: str,
        runner: ModelRunner,
        batcher: DynamicBatcher | None = None,
        default: bool = False,
    ) -> None:
        if name in self._entries:
            raise ValueError(f"model {name!r} already registered")
        self._entries[name] = ModelEntry(name=name, runner=runner, batcher=batcher)
        if default or self._default is None:
            self._default = name

    def get(self, name: str) -> ModelEntry | None:
        return self._entries.get(name)

    def default(self) -> ModelEntry | None:
        if self._default is None:
            return None
        return self._entries[self._default]

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def all(self) -> list[ModelEntry]:
        return list(self._entries.values())
