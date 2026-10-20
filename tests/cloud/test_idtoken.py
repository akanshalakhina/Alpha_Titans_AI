"""Verifying the worker's Google ID token.

A real 2048-bit RSA key is generated here, in pure Python and from a fixed
seed so every run is identical, and real RS256 tokens are signed with it.
The verifier under test is the production one; only the JWKS endpoint it
fetches is served from memory.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import random

import pytest

from cloud import idtoken as IT
from cloud.gcp import Response
from cloud.sanitize import CloudError

AUD = "https://cp.example.test"
EMAIL = "at-worker@test-project-123.iam.gserviceaccount.com"
NOW = 1_800_000_000.0


def _probable_prime(n: int, rng: random.Random) -> bool:
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for _ in range(24):
        x = pow(rng.randrange(2, n - 2), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime(bits: int, rng: random.Random) -> int:
    while True:
        c = rng.getrandbits(bits) | (3 << (bits - 2)) | 1
        if _probable_prime(c, rng):
            return c


def _keypair(seed: int):
    rng = random.Random(seed)
    e = 65537
    while True:
        p, q = _prime(1024, rng), _prime(1024, rng)
        phi = (p - 1) * (q - 1)
        if p != q and math.gcd(e, phi) == 1:
            n = p * q
            return n, e, pow(e, -1, phi)


@pytest.fixture(scope="module")
def key():
    n, e, d = _keypair(20261017)
    assert n.bit_length() == 2048
    return n, e, d


@pytest.fixture(scope="module")
def other_key():
    return _keypair(42)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64int(i: int) -> str:
    return _b64(i.to_bytes((i.bit_length() + 7) // 8, "big"))


def sign(key, claims: dict, header: dict | None = None) -> str:
    n, _e, d = key
    head = {"alg": "RS256", "kid": "k1", "typ": "JWT", **(header or {})}
    si = f"{_b64(json.dumps(head).encode())}.{_b64(json.dumps(claims).encode())}"
    k = (n.bit_length() + 7) // 8
    em = IT.expected_encoding(si.encode(), k)
    sig = pow(int.from_bytes(em, "big"), d, n).to_bytes(k, "big")
    return f"{si}.{_b64(sig)}"


def good_claims(**kw) -> dict:
    return {"iss": "https://accounts.google.com", "aud": AUD,
            "email": EMAIL, "email_verified": True, "iat": NOW - 10,
            "exp": NOW + 3000, "sub": "1234567890", **kw}


class Certs:
    """Google's JWKS endpoint, from memory."""

    def __init__(self, *keys):
        self.keys = list(keys)
        self.fetches = 0

    def request(self, method, url, headers=None, body=None, timeout=None):
        self.fetches += 1
        jwks = {"keys": [{"kty": "RSA", "alg": "RS256", "use": "sig",
                          "kid": kid, "n": _b64int(n), "e": _b64int(e)}
                         for kid, (n, e, _d) in self.keys]}
        return Response(200, {}, json.dumps(jwks).encode())


@pytest.fixture
def certs(key):
    return Certs(("k1", key))


@pytest.fixture
def verifier(certs):
    return IT.GoogleIdVerifier(AUD, EMAIL, http=certs, clock=lambda: NOW)


# --------------------------------------------------------------- accepted

def test_a_genuine_token_is_accepted(verifier, key):
    claims = verifier.verify(sign(key, good_claims()))
    assert claims["email"] == EMAIL


def test_an_audience_list_containing_us_is_accepted(verifier, key):
    verifier.verify(sign(key, good_claims(aud=["other", AUD])))


# --------------------------------------------------------------- refused

@pytest.mark.parametrize("claims", [
    good_claims(aud="https://someone-else.test"),
    good_claims(exp=NOW - 120),
    good_claims(iat=NOW + 600),
    good_claims(email="attacker@evil.iam.gserviceaccount.com"),
    good_claims(email_verified=False),
    good_claims(iss="https://evil.example"),
    {k: v for k, v in good_claims().items() if k != "exp"},
], ids=["audience", "expired", "future", "email", "unverified", "issuer",
        "no-exp"])
def test_a_wrong_claim_is_refused(verifier, key, claims):
    with pytest.raises(CloudError) as e:
        verifier.verify(sign(key, claims))
    assert e.value.category == "auth"


def test_a_token_signed_by_another_key_is_refused(verifier, other_key):
    with pytest.raises(CloudError):
        verifier.verify(sign(other_key, good_claims()))


def test_a_tampered_payload_is_refused(verifier, key):
    h, _p, s = sign(key, good_claims()).split(".")
    forged = _b64(json.dumps(good_claims(email="me@evil.test")).encode())
    with pytest.raises(CloudError):
        verifier.verify(f"{h}.{forged}.{s}")


def test_alg_none_is_refused(verifier):
    head = _b64(json.dumps({"alg": "none", "kid": "k1"}).encode())
    body = _b64(json.dumps(good_claims()).encode())
    with pytest.raises(CloudError):
        verifier.verify(f"{head}.{body}.")


def test_hs256_with_the_public_key_as_secret_is_refused(verifier, key):
    """The classic algorithm-confusion attack."""
    n, e, _d = key
    secret = json.dumps({"n": _b64int(n), "e": _b64int(e)}).encode()
    head = _b64(json.dumps({"alg": "HS256", "kid": "k1"}).encode())
    body = _b64(json.dumps(good_claims()).encode())
    sig = _b64(hmac.new(secret, f"{head}.{body}".encode(),
                        hashlib.sha256).digest())
    with pytest.raises(CloudError):
        verifier.verify(f"{head}.{body}.{sig}")


@pytest.mark.parametrize("token", ["", "abc", "a.b", "a.b.c.d",
                                   "!!.@@.##", None])
def test_garbage_is_refused_without_a_crash(verifier, token):
    with pytest.raises(CloudError):
        verifier.verify(token)


def test_an_unknown_kid_refetches_the_keys_but_not_on_every_request(
        certs, key):
    v = IT.GoogleIdVerifier(AUD, EMAIL, http=certs, clock=lambda: NOW)
    v.verify(sign(key, good_claims()))
    before = certs.fetches
    for _ in range(5):
        with pytest.raises(CloudError):
            v.verify(sign(key, good_claims(), {"kid": "rotated"}))
    assert certs.fetches - before <= 1, "forged kids are not an amplifier"


def test_a_rotated_key_is_picked_up(key, other_key):
    certs = Certs(("k1", key))
    clock = [NOW]
    v = IT.GoogleIdVerifier(AUD, EMAIL, http=certs, clock=lambda: clock[0])
    v.verify(sign(key, good_claims()))
    certs.keys.append(("k2", other_key))
    clock[0] += 120
    v.verify(sign(other_key, good_claims(), {"kid": "k2"}))


def test_a_short_key_in_the_jwks_is_ignored():
    small_n = _prime(512, random.Random(1)) * _prime(512, random.Random(2))
    certs = Certs(("k1", (small_n, 65537, 0)))
    v = IT.GoogleIdVerifier(AUD, EMAIL, http=certs, clock=lambda: NOW)
    with pytest.raises(CloudError):
        v.verify("a.b.c")


def test_the_signature_check_compares_the_whole_encoding(key):
    """A signature that decrypts to a correctly-prefixed block with
    trailing garbage after the hash is a known forgery shape for parsers.
    Comparing the full encoding leaves it nowhere to hide."""
    n, e, d = key
    k = (n.bit_length() + 7) // 8
    good = IT.expected_encoding(b"msg", k)
    bad = good[:-1] + bytes([good[-1] ^ 1])
    sig = pow(int.from_bytes(bad, "big"), d, n).to_bytes(k, "big")
    assert IT.rs256_valid(b"msg", sig, n, e) is False
    sig_ok = pow(int.from_bytes(good, "big"), d, n).to_bytes(k, "big")
    assert IT.rs256_valid(b"msg", sig_ok, n, e) is True
