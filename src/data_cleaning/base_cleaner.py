from __future__ import annotations

from abc import ABC, abstractmethod

from .records import CleaningStats


class BaseCleaner(ABC):
    """Dataset cleaner contract used by pipeline dispatch."""

    @abstractmethod
    def run(self, dry_run: bool = False, limit: int | None = None) -> CleaningStats:
        raise NotImplementedError
