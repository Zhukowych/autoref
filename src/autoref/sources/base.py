"""PaperSource interface + shared HTTP plumbing (politeness delay, retry/backoff)."""
from __future__ import annotations

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from ..models import CitationContext, ExternalIds, PaperRecord, ReferenceRecord

log = logging.getLogger(__name__)

USER_AGENT = "autoref/0.1 (literature-review CLI)"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class SourceError(RuntimeError):
    pass


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """`Retry-After` is either delta-seconds or an HTTP date. Returns seconds to wait (>= 0) or None."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - (now or datetime.now(timezone.utc))).total_seconds())


class PaperSource(ABC):
    """Common interface for all bibliographic sources.

    Every method returns source-agnostic records (`PaperRecord`, `ReferenceRecord`) so the
    pipeline can swap/merge sources freely. Methods a source cannot support return [] / None
    and the capability is absent from `capabilities`.
    """

    name: str = "base"
    capabilities: frozenset[str] = frozenset()  # search, get, references, citations, contexts, abstract, pdf, date_window
    min_delay: float = 1.0  # seconds between requests (politeness floor)

    # --- retry / backoff policy -------------------------------------------------------------
    max_attempts = 6            # 1 try + 5 retries
    backoff_base = 2.0          # wait before retry n = min(cap, base * 2**n), with jitter
    backoff_cap = 60.0
    retry_after_cap = 300.0     # server asks for longer than this -> give up on this request
    max_slowdown = 8.0          # adaptive request spacing: up to min_delay * this after rate limits
    recover_after = 20          # successful requests in a row before spacing shrinks again
    max_consecutive_failures = 2  # circuit breaker: requests that exhausted all retries, in a row

    def __init__(self, settings=None, client: httpx.Client | None = None):
        self.settings = settings
        if settings is not None and getattr(settings, "request_delay", None) is not None:
            self.min_delay = settings.request_delay
        self._client = client or httpx.Client(timeout=60.0, follow_redirects=True,
                                              headers={"User-Agent": USER_AGENT})
        self._last = 0.0
        self._lock = threading.Lock()
        self._delay = self.min_delay  # current spacing; grows after 429, decays on success
        self._ok_streak = 0
        self._fails = 0
        self._disabled = False
        self._sleep = time.sleep              # replaceable in tests
        self._uniform = random.uniform

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
            wait = self._delay - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()

    def _slow_down(self):
        """Rate-limited: double the spacing between requests (bounded)."""
        self._ok_streak = 0
        self._delay = min(max(self._delay, self.min_delay, 0.05) * 2, max(self.min_delay, 0.05) * self.max_slowdown)

    def _on_success(self):
        self._fails = 0
        self._ok_streak += 1
        if self._delay > self.min_delay and self._ok_streak >= self.recover_after:
            self._delay = max(self.min_delay, self._delay / 2)
            self._ok_streak = 0

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with 'equal jitter' (uniform in [w/2, w]); never earlier than
        the server's Retry-After."""
        w = min(self.backoff_cap, self.backoff_base * (2 ** attempt))
        wait = self._uniform(w / 2, w)
        if retry_after is not None:
            wait = max(wait, retry_after + self._uniform(0, 1.0))
        return wait

    def _get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        if self._disabled:
            raise SourceError(f"{self.name} disabled after repeated failures (rate limit / outage)")
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            retry_after = None
            try:
                self._throttle()
                r = self._client.get(url, params=params, headers=headers)
            except httpx.TransportError as e:  # timeouts, connection resets, DNS, ...
                last_exc = e
                reason = type(e).__name__
            else:
                if r.status_code not in RETRY_STATUS:
                    if r.status_code != 404:
                        r.raise_for_status()  # other 4xx: our bug, not retryable, not an outage
                    self._on_success()
                    return r
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    last_exc = e
                reason = f"HTTP {r.status_code}"
                retry_after = parse_retry_after(r.headers.get("Retry-After"))
                if r.status_code == 429:
                    self._slow_down()
                if retry_after is not None and retry_after > self.retry_after_cap:
                    log.warning("%s: server asked to wait %.0fs (> %.0fs cap), giving up on %s",
                                self.name, retry_after, self.retry_after_cap, url)
                    break
            if attempt == self.max_attempts - 1:
                break
            wait = self._backoff(attempt, retry_after)
            log.info("%s: %s on %s, retry %d/%d in %.1fs%s", self.name, reason, url, attempt + 1,
                     self.max_attempts - 1, wait, " (Retry-After)" if retry_after is not None else "")
            self._sleep(wait)
        # all attempts exhausted -> count towards the circuit breaker
        self._ok_streak = 0
        self._fails += 1
        if self._fails >= self.max_consecutive_failures:
            self._disabled = True
            log.warning("%s disabled for this run after %d failed requests: %s", self.name, self._fails, last_exc)
        raise last_exc
