"""Overlay URL signatures: what forgery looks like when it is refused.

An overlay path is ``/{job_id}/{name}.png`` where the client knows the first segment and can guess
the second. Job ids are minted with a sortable millisecond prefix, so without a signature an outsider
who knows roughly when someone uploaded can walk the prefix and read other people's heatmaps — which
is the exact promise the privacy page makes. These tests are the refusal cases, not the happy path.

One thing they deliberately do *not* claim: the signature is not what stops a traversal. The pipe
delimiters in the signed message mean a ``job_id`` containing a pipe could be re-aligned against the
name field, which is why ``scratch.path_safe`` runs before verification at the route and why the
allowlist rejects that character. The two layers are tied together here so that a change to either
one shows up as a failure.
"""

from __future__ import annotations

import hashlib
import hmac
from urllib.parse import parse_qs, urlparse

import pytest

from app.core.job import new_job_id
from app.core.scratch import path_safe
from app.core.signing import OverlaySigner, verify_signature

KEY = "unit-test-key"
NOW = 1_700_000_000.0


@pytest.fixture()
def signer() -> OverlaySigner:
    return OverlaySigner(KEY, ttl_s=540)


@pytest.fixture()
def job_id() -> str:
    return new_job_id()


def _split(url: str) -> tuple[str, str, str, str]:
    """(job_id, name, exp, sig) out of a minted URL, so tests speak in URL terms."""
    parsed = urlparse(url)
    tail = parsed.path.split("/")
    qs = parse_qs(parsed.query)
    return tail[-2], tail[-1].removesuffix(".png"), qs["exp"][0], qs["sig"][0]


# --------------------------------------------------------------------- minting


def test_a_fresh_url_verifies(signer, job_id):
    assert signer.check_url(signer.url(job_id, "ela"))


def test_empty_key_is_refused_at_construction():
    """A signer with no key would sign everything with one fixed, publicly-derivable HMAC."""
    with pytest.raises(ValueError):
        OverlaySigner("")


def test_url_is_relative_and_names_both_segments(signer, job_id):
    url = signer.url(job_id, "copy_move")
    assert url.startswith("/api/overlays/")
    assert f"{job_id}/copy_move.png" in url


def test_a_ttl_override_may_shorten_but_never_extend(signer, job_id):
    _, _, exp, _ = _split(signer.url(job_id, "ela", now=NOW, ttl_override=60))
    assert int(exp) == NOW + 60
    with pytest.raises(ValueError):
        signer.url(job_id, "ela", now=NOW, ttl_override=signer.ttl_s + 1)


def test_a_minted_url_lands_on_a_path_the_route_will_agree_to_serve(signer, job_id):
    """The signer will sign any string, so the pairing is what has to hold."""
    job, name, _, _ = _split(signer.url(job_id, "jpeg_ghost"))
    assert path_safe(job, name)


# ------------------------------------------------------------------- forgeries


def test_a_flipped_signature_byte_is_refused(signer, job_id):
    job, name, exp, sig = _split(signer.url(job_id, "ela"))
    tampered = ("0" if sig[0] != "0" else "1") + sig[1:]
    assert not verify_signature(KEY, job, name, exp, tampered)


def test_a_signature_does_not_transfer_to_another_job(signer, job_id):
    """The property that closes prefix-walking: replaying a valid token against a neighbouring id."""
    job, name, exp, sig = _split(signer.url(job_id, "ela"))
    assert not verify_signature(KEY, new_job_id(), name, exp, sig)


def test_a_signature_does_not_transfer_to_another_technique(signer, job_id):
    job, _, exp, sig = _split(signer.url(job_id, "ela"))
    assert not verify_signature(KEY, job, "cnn", exp, sig)


def test_re_dating_a_token_requires_re_signing(signer, job_id):
    """``exp`` is inside the signed message, so a client cannot push its own deadline out."""
    job, name, _, sig = _split(signer.url(job_id, "ela", now=NOW))
    assert not verify_signature(KEY, job, name, str(int(NOW) + 10_000), sig, now=NOW)


def test_a_url_signed_with_another_deployments_key_is_refused(signer, job_id):
    other = OverlaySigner("a-different-deployment-key", ttl_s=540)
    assert not signer.check_url(other.url(job_id, "ela"))
    assert not other.check_url(signer.url(job_id, "ela"))


def test_expired_is_refused_even_with_an_authentic_signature(signer, job_id):
    url = signer.url(job_id, "ela", now=NOW)
    assert signer.check_url(url, now=NOW + 60)
    assert not signer.check_url(url, now=NOW + 600)


def test_the_deadline_is_exclusive(signer, job_id):
    """At exactly ``exp`` the token is already dead. A boundary that admits one more second admits a
    clock-skew argument nobody wants to have."""
    job, name, exp, sig = _split(signer.url(job_id, "ela", now=NOW))
    assert signer.check(job, name, exp, sig, now=int(exp) - 1)
    assert not signer.check(job, name, exp, sig, now=int(exp))


# ------------------------------------------------------------- malformed tokens


@pytest.mark.parametrize("sig", ["", "0" * 63, "0" * 65, "z" * 64, "A" * 64])
def test_a_signature_must_be_lowercase_hex_of_exactly_64(sig, signer, job_id):
    """``now`` is pinned inside the TTL so a rejection here is about the signature, not the clock."""
    assert not verify_signature(KEY, job_id, "ela", str(int(NOW) + 60), sig, now=NOW)


@pytest.mark.parametrize("exp", ["", "abc", "12.5", None, "1e9", []])
def test_a_non_integer_expiry_is_refused_without_raising(exp, signer, job_id):
    """Route code hands these over straight from a query string; parsing them is not its job."""
    _, _, _, sig = _split(signer.url(job_id, "ela"))
    assert not verify_signature(KEY, job_id, "ela", exp, sig)


def test_exp_is_normalised_before_signing(signer, job_id):
    """``000123`` and ``123`` are the same deadline.

    The signature commits to a number, not to one spelling of it, so a proxy that pads leading zeros
    cannot be the reason a legitimate overlay 404s.
    """
    exp = int(NOW) + 600
    sig = signer.signature(job_id, "ela", exp)
    assert verify_signature(KEY, job_id, "ela", f"000{exp}", sig, now=NOW)


def test_a_url_with_no_query_does_not_verify(signer, job_id):
    assert not signer.check_url(f"/api/overlays/{job_id}/ela.png")


def test_a_path_too_short_to_name_a_job_does_not_verify(signer):
    assert not signer.check_url("/ela.png?exp=9999999999&sig=" + "0" * 64)


def test_an_empty_key_never_verifies(signer, job_id):
    _, _, exp, sig = _split(signer.url(job_id, "ela"))
    assert not verify_signature("", job_id, "ela", exp, sig)


# ------------------------------------------------------------------- internals


def test_the_digest_is_not_truncated(signer, job_id):
    """64 hex characters — the full SHA-256.

    A "URL-friendly" 12-character signature is a forgery budget, and it is the kind of shortcut that
    gets added for tidiness.
    """
    _, _, _, sig = _split(signer.url(job_id, "ela"))
    assert len(sig) == len(hmac.new(b"k", b"m", hashlib.sha256).hexdigest()) == 64


def test_the_canonical_message_is_the_documented_one(signer, job_id):
    """Pin the field order.

    Reordering the message or changing the delimiter silently invalidates every URL a client still
    holds, and it reads like a key-rotation problem when it is not.
    """
    expected = hmac.new(KEY.encode(), f"{job_id}|ela|{int(NOW) + 540}".encode(),
                        hashlib.sha256).hexdigest()
    assert signer.signature(job_id, "ela", int(NOW) + 540) == expected
    assert _split(signer.url(job_id, "ela", now=NOW))[3] == expected
