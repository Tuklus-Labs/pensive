"""Gate for Host-header validation on every route this daemon serves.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    The daemon answers only to hostnames it was configured to answer to. A
    request arriving with an attacker-controlled Host is refused before any
    handler runs, and a request from the configured reverse proxy still works.

Why this is not covered by the existing Origin check. A blue-team review
(gpt-daybreak, 2026-08-13) traced a DNS-rebinding path to the new
unauthenticated L1 routes:

  1. the operator loads http://attacker.example:5999 with Referrer-Policy: no-referrer
  2. DNS for attacker.example is rebound to 127.0.0.1
  3. a same-origin GET may omit Origin entirely, and Referer is suppressed
  4. `checkRequestOrigin` explicitly ALLOWS both headers absent, because
     non-browser clients send neither and requiring them would break every
     wired agent on this box
  5. the browser believes the response is same-origin, so CORS never applies

The Origin check is therefore correct and insufficient: the hole is the
permitted absence, and GET is exactly the method allowed to omit it. Host is the
header the attack CANNOT forge away, because the browser sets it from the URL it
believes it is talking to. `/brief` had the same exposure and is not a safe
precedent for the new routes; it inherits this guard too.

BLAST RADIUS, measured before writing the guard rather than after: the
pensive-shep Caddy block reverse-proxies to 127.0.0.1:5999 with NO
``header_up Host``, and Caddy preserves the original Host by default, so the
daemon really does receive ``pensive.example.com`` on Shepherd's bridge. A
loopback-only allowlist would have severed that connector silently. The default
here is loopback-only because that is the safe default for a daemon; the
deployment adds its proxy hostname explicitly.
"""
import pytest

from serve.tee import checkAllowedHost, allowedHostsFromEnv, DEFAULT_ALLOWED_HOSTS


LOOPBACK = DEFAULT_ALLOWED_HOSTS


def _headers(host):
    return {} if host is None else {"host": host}


# --------------------------------------------------------------------------- #
# the rebinding case
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("host", [
    "attacker.example:5999",
    "attacker.example",
    "evil.test:5999",
    "127.0.0.1.attacker.example",       # prefix trick
    "attacker.example#127.0.0.1",
])
def test_an_unknown_host_is_refused(host):
    rejection = checkAllowedHost(_headers(host), LOOPBACK)
    assert rejection is not None, f"Host {host!r} was accepted"
    status, _ = rejection
    assert status == 403


@pytest.mark.parametrize("host", [
    "127.0.0.1:5999", "127.0.0.1", "localhost:5999", "localhost",
    "[::1]:5999", "[::1]",
])
def test_loopback_hosts_are_accepted(host):
    assert checkAllowedHost(_headers(host), LOOPBACK) is None


def test_a_configured_proxy_hostname_is_accepted():
    """Shepherd's bridge arrives through Caddy carrying the PUBLIC hostname,
    because that block sets no header_up Host and Caddy preserves the original.
    Refusing it would have silently cut a live consumer."""
    allowed = LOOPBACK | {"pensive.example.com"}
    assert checkAllowedHost(_headers("pensive.example.com"), allowed) is None
    # and it is still refused when NOT configured, so the allowance is explicit
    assert checkAllowedHost(_headers("pensive.example.com"), LOOPBACK) is not None


def test_a_missing_host_is_refused():
    """HTTP/1.1 requires Host. An absent one is a malformed request or a
    hand-rolled client, and this is the one header the rebinding attack cannot
    strip -- so unlike Origin, absence here is NOT allowed."""
    assert checkAllowedHost(_headers(None), LOOPBACK) is not None


def test_the_comparison_is_case_insensitive_and_port_tolerant():
    assert checkAllowedHost(_headers("LOCALHOST:5999"), LOOPBACK) is None
    assert checkAllowedHost(_headers("127.0.0.1:8080"), LOOPBACK) is None


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_the_default_is_loopback_only(monkeypatch):
    """A daemon that ships trusting a public hostname trusts it everywhere it is
    ever deployed. The proxy hostname is added by the deployment that has one."""
    monkeypatch.delenv("PENSIVE_V3_ALLOWED_HOSTS", raising=False)
    assert allowedHostsFromEnv() == DEFAULT_ALLOWED_HOSTS
    assert not any("tukluslabs" in h for h in DEFAULT_ALLOWED_HOSTS)


def test_extra_hosts_are_added_not_replaced(monkeypatch):
    """Replacing rather than extending would let a deployment silently drop
    loopback and lock every local agent out of its own memory."""
    monkeypatch.setenv("PENSIVE_V3_ALLOWED_HOSTS", "pensive.example.com, other.host")
    got = allowedHostsFromEnv()
    assert "pensive.example.com" in got
    assert "other.host" in got
    assert DEFAULT_ALLOWED_HOSTS <= got, "loopback was dropped"


def test_blank_and_whitespace_entries_are_ignored(monkeypatch):
    monkeypatch.setenv("PENSIVE_V3_ALLOWED_HOSTS", " , ,  ")
    assert allowedHostsFromEnv() == DEFAULT_ALLOWED_HOSTS


# --------------------------------------------------------------------------- #
# malformed and duplicate authorities                                          #
# --------------------------------------------------------------------------- #
#
# Found by a blue-team review (gpt-daybreak) after the guard landed. The parser
# split on the first ':' and took what was left, which accepts several shapes
# that are not the authority they look like:
#
#     Host: [::1].evil.example        parsed as  [::1]
#     Host: localhost:80@evil.example parsed as  localhost
#     duplicate Host headers          last one silently wins
#
# Scored LOW because a standard browser cannot emit any of them, so the DNS
# rebinding path this guard exists to close stays closed. It is still wrong: a
# raw client or a sloppy intermediary can, and a parser that accepts a string
# it should reject is one intermediary away from being the whole defense.


@pytest.mark.parametrize("host", [
    "[::1].evil.example",
    "localhost:80@evil.example",
    "127.0.0.1@evil.example",
    "localhost:80:90",
    "[::1",
    "localhost:notaport",
])
# "localhost " with trailing space is NOT here on purpose: RFC 7230 permits
# optional whitespace around a field value and servers strip it, so that form is
# legal HTTP. Rejecting it would make the parser stricter than the spec to
# satisfy a case I wrote carelessly, which is a worse defect than the one being
# fixed.
def test_a_malformed_authority_is_refused(host):
    assert checkAllowedHost(_headers(host), LOOPBACK) is not None, (
        f"malformed authority {host!r} was accepted"
    )


def test_a_valid_authority_with_a_port_is_still_accepted():
    for host in ("127.0.0.1:5999", "localhost:80", "[::1]:5999"):
        assert checkAllowedHost(_headers(host), LOOPBACK) is None, host
