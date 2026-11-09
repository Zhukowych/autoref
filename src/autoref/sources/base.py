"""PaperSource interface + shared HTTP plumbing (politeness delay, retry/backoff)."""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import date

import httpx
from tenacity import (
    retry, retry_if_exception, stop_after_attempt, wait_random_exponential,
)

from ..models import CitationContext, ExternalIds, PaperRecord, ReferenceRecord

log = logging.getLogger(__name__)

USER_AGENT = "autoref/0.1 (literature-review CLI)"


class SourceError(RuntimeError):
    pass


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError,))


class PaperSource(ABC):
    """Common interface for all bibliographic sources.

    Every method returns source-agnostic records (`PaperRecord`, `ReferenceRecord`) so the
    pipeline can swap/merge sources freely. Methods a source cannot support return [] / None
    and the capability is absent from `capabilities`.
    """

    name: str = "base"
    capabilities: frozenset[str] = frozenset()  # search, get, references, citations, contexts, abstract, pdf, date_window
    min_delay: float = 1.0  # seconds between requests

    def __init__(self, settings=None, client: httpx.Client | None = None):
        self.settings = settings
        if settings is not None and getattr(settings, "request_delay", None) is not None:
            self.min_delay = settings.request_delay
        self._client = client or httpx.Client(timeout=60.0, follow_redirects=True,
                                              headers={"User-Agent": USER_AGENT})
        self._last = 0.0
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- interface
    @abstractmethod
    def search(self, query: str, limit: int = 100, year_from: int | None = None,
               year_to: int | None = None, date_from: date | None = None,
               date_to: date | None = None) -> list[PaperRecord]:
        """Keyword search. `date_from/date_to` restrict by publication/submission date."""

    def get(self, ids: ExternalIds, title: str | None = None) -> PaperRecord | None:
        return None

    def references(self, paper: PaperRecord, limit: int = 1000) -> list[ReferenceRecord]:
        """Papers cited BY `paper` (with citation contexts if the source has them)."""
        return []

    def citations(self, paper: PaperRecord, limit: int = 1000, date_from: date | None = None
                  ) -> list[ReferenceRecord]:
        """Papers CITING `paper`; ReferenceRecord.cited holds the *citing* paper here."""
        return []

    def citation_contexts(self, citing: PaperRecord, cited: PaperRecord) -> list[CitationContext]:
        return []

    def has(self, cap: str) -> bool:
        return cap in self.capabilities

    def close(self):
        self._client.close()

    # ---------------------------------------------------------------- http
    def _throttle(self):
        with self._lock:
            wait = self.min_delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    max_consecutive_failures = 2  # circuit breaker: disable source for the rest of the run

    def _get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        if getattr(self, "_disabled", False):
            raise SourceError(f"{self.name} disabled after repeated failures (rate limit / outage)")
        try:
            r = self._get_retrying(url, params, headers)
        except Exception as e:
            if _retryable(e):
                self._fails = getattr(self, "_fails", 0) + 1
                if self._fails >= self.max_consecutive_failures:
                    self._disabled = True
                    log.warning("%s disabled for this run after %d failed requests: %s", self.name, self._fails, e)
            raise
        self._fails = 0
        return r

    @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(5),
           wait=wait_random_exponential(multiplier=2, max=30), reraise=True)
    def _get_retrying(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        self._throttle()
        r = self._client.get(url, params=params, headers=headers)
        if r.status_code == 404:
            return r
        r.raise_for_status()
        return r
