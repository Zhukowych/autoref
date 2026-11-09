"""HTTP backoff: Retry-After, exponential growth with jitter, adaptive spacing, circuit breaker."""
from datetime import datetime, timezone

import httpx
import pytest
import respx

from autoref.sources.base import SourceError, parse_retry_after
from autoref.sources.openalex import OpenAlexSource

URL = "https://api.openalex.org/works"
OK = httpx.Response(200, json={"results": [], "meta": {"next_cursor": None}})


@pytest.fixture
def src():
    s = OpenAlexSource()
    s.min_delay = s._delay = 0.0
    s.sleeps = []
    s._sleep = s.sleeps.append          # record instead of sleeping
    s._uniform = lambda a, b: b         # deterministic: upper end of the jitter range
    yield s
    s.close()


def test_parse_retry_after():
    assert parse_retry_after("7") == 7
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("-3") == 0
    assert parse_retry_after(None) is None and parse_retry_after("soon") is None
    now = datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Mon, 09 Nov 2026 12:01:30 GMT", now=now) == 90
    assert parse_retry_after("Mon, 09 Nov 2026 11:00:00 GMT", now=now) == 0  # date in the past


@respx.mock
def test_exponential_growth_then_success(src):
    respx.get(URL).mock(side_effect=[httpx.Response(503)] * 4 + [OK])
    src.search("x", limit=1)
    assert src.sleeps == [2, 4, 8, 16]               # base * 2**n, jitter pinned to the top


@respx.mock
def test_backoff_is_capped(src):
    src.max_attempts = 9
    respx.get(URL).mock(side_effect=[httpx.Response(500)] * 8 + [OK])
    src.search("x", limit=1)
    assert src.sleeps[:6] == [2, 4, 8, 16, 32, 60] and max(src.sleeps) == 60


@respx.mock
def test_jitter_range(src):
    src._uniform = lambda a, b: a                    # bottom of the range
    respx.get(URL).mock(side_effect=[httpx.Response(502)] * 2 + [OK])
    src.search("x", limit=1)
    assert src.sleeps == [1, 2]                      # w/2


@respx.mock
def test_retry_after_seconds_is_honored(src):
    respx.get(URL).mock(side_effect=[httpx.Response(429, headers={"Retry-After": "45"}), OK])
    src.search("x", limit=1)
    assert 45 <= src.sleeps[0] <= 46                  # longer than the 2 s backoff, plus <1 s jitter


@respx.mock
def test_retry_after_shorter_than_backoff_does_not_shorten_it(src):
    respx.get(URL).mock(side_effect=[httpx.Response(429, headers={"Retry-After": "0"}), OK])
    src.search("x", limit=1)
    assert src.sleeps[0] >= 2


@respx.mock
def test_retry_after_beyond_cap_gives_up_immediately(src):
    route = respx.get(URL).mock(return_value=httpx.Response(429, headers={"Retry-After": "7200"}))
    with pytest.raises(httpx.HTTPStatusError):
        src.search("x", limit=1)
    assert route.call_count == 1 and src.sleeps == []


@respx.mock
def test_transport_errors_are_retried(src):
    respx.get(URL).mock(side_effect=[httpx.ConnectTimeout("t"), httpx.ConnectError("c"), OK])
    src.search("x", limit=1)
    assert len(src.sleeps) == 2


@respx.mock
def test_client_errors_and_404_are_not_retried(src):
    route = respx.get(URL).mock(return_value=httpx.Response(400))
    with pytest.raises(httpx.HTTPStatusError):
        src.search("x", limit=1)
    assert route.call_count == 1 and src._fails == 0 and not src._disabled
    route.mock(return_value=httpx.Response(404))
    assert src._get(URL).status_code == 404 and route.call_count == 2


@respx.mock
def test_spacing_grows_after_429_and_recovers(src):
    src.min_delay = 1.0
    src._delay = 1.0
    src.recover_after = 3
    respx.get(URL).mock(side_effect=[httpx.Response(429), httpx.Response(429), OK] + [OK] * 12)
    src._get(URL)
    assert src._delay == 4.0                          # 1 -> 2 -> 4
    for _ in range(8):
        src._get(URL)
    assert src._delay == 1.0                          # halves every 3 successes, floor = min_delay
    src._delay = 1.0
    for _ in range(5):
        src._slow_down()
    assert src._delay == 8.0                          # bounded by max_slowdown


@respx.mock
def test_circuit_breaker_after_exhausted_retries(src):
    route = respx.get(URL).mock(return_value=httpx.Response(429))
    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            src.search("x", limit=1)
    assert route.call_count == 2 * src.max_attempts and src._disabled
    n = route.call_count
    with pytest.raises(SourceError):
        src.search("x", limit=1)
    assert route.call_count == n                      # no HTTP once disabled


@respx.mock
def test_success_resets_failure_count(src):
    respx.get(URL).mock(side_effect=[httpx.Response(500)] * 6 + [OK])
    with pytest.raises(httpx.HTTPStatusError):
        src._get(URL)
    assert src._fails == 1
    src._get(URL)
    assert src._fails == 0 and not src._disabled
