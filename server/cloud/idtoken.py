"""Verifying a Google-signed ID token, stdlib only.

The worker VM proves who it is with an ID token minted by its metadata
server: a JWT signed by Google with RS256, naming the VM's service account.
No shared secret has to be provisioned, rotated or leaked.

Verification needs only the PUBLIC key, which is why it can be done safely
without a crypto library. Signing would need a private key and constant-time
arithmetic; verifying is one modular exponentiation with a public exponent,
and there is nothing secret to leak through timing.

HOW THE SIGNATURE IS CHECKED
----------------------------
By RECONSTRUCTION, as RFC 8017 section 8.2.2 specifies: compute what a valid
PKCS#1 v1.5 encoding of this token's SHA-256 digest must be, byte for byte,
and compare the whole thing with what the signature decrypts to. Nothing is
parsed out of the decrypted block. Parsing is where the well-known forgeries
live (Bleichenbacher 2006, and every "find the 0x00 and read the hash after
it" bug since); a full comparison has nowhere for them to hide.

THE CLAIMS
----------
* `alg` must be exactly RS256. No `none`, and no HS256 "verified" with the
  public key as an HMAC secret - the classic algorithm-confusion attack.
* `iss` is Google's, `aud` is THIS control plane, `exp` is in the future
  (with 60s of clock skew), `iat` not in the future.
* `email` is the one configured worker service account, and
  `email_verified` is true.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time

from .gcp import Http
from .sanitize import CloudError

GOOGLE_CERTS = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})
SKEW_S = 60.0
MIN_RSA_BITS = 2048
# DER prefix of DigestInfo for SHA-256 (RFC 8017, section 9.2, note 1)
_SHA256_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


def b64url(data: str) -> bytes:
    data = data.strip()
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _int(data: str) -> int:
    return int.from_bytes(b64url(data), "big")


def expected_encoding(message: bytes, k: int) -> bytes:
    """EMSA-PKCS1-v1_5 encoding of SHA-256(message) for a k-byte modulus."""
    t = _SHA256_PREFIX + hashlib.sha256(message).digest()
    if k < len(t) + 11:
        raise CloudError("auth", "key too small")
    return b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t


def rs256_valid(signing_input: bytes, signature: bytes, n: int, e: int
                ) -> bool:
    k = (n.bit_length() + 7) // 8
    if len(signature) != k:
        return False
    s = int.from_bytes(signature, "big")
    if not 0 < s < n:
        return False
    em = pow(s, e, n).to_bytes(k, "big")
    return hmac.compare_digest(em, expected_encoding(signing_input, k))


class GoogleIdVerifier:
    def __init__(self, audience: str, expected_email: str,
                 http: Http | None = None, clock=time.time,
                 certs_url: str = GOOGLE_CERTS, cache_s: float = 3600.0):
        if not audience or not expected_email:
            raise CloudError("invalid", "audience and email are required")
        self.audience = audience
        self.email = expected_email.lower()
        self.http = http or Http(timeout=10.0)
        self.clock = clock
        self.certs_url = certs_url
        self.cache_s = cache_s
        self._keys: dict[str, tuple[int, int]] = {}
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    def _refresh(self, force: bool = False) -> None:
        now = self.clock()
        with self._lock:
            fresh = now - self._fetched_at < self.cache_s
            # an unknown kid forces a refetch, but at most once a minute so
            # a stream of forged kids cannot turn us into an amplifier
            if (fresh and not force) or (force and now - self._fetched_at
                                         < 60 and self._keys):
                return
            r = self.http.request("GET", self.certs_url, timeout=10)
            r.raise_for_status("google certs")
            keys = {}
            for jwk in r.json().get("keys", []):
                if jwk.get("kty") != "RSA" or not jwk.get("kid"):
                    continue
                if jwk.get("alg") not in (None, "RS256"):
                    continue
                n, e = _int(jwk["n"]), _int(jwk["e"])
                if n.bit_length() >= MIN_RSA_BITS and e > 1:
                    keys[jwk["kid"]] = (n, e)
            if not keys:
                raise CloudError("auth", "no usable Google signing keys")
            self._keys, self._fetched_at = keys, now

    def _key(self, kid: str) -> tuple[int, int]:
        self._refresh()
        if kid not in self._keys:
            self._refresh(force=True)
        if kid not in self._keys:
            raise CloudError("auth", "token signed by an unknown key")
        return self._keys[kid]

    def verify(self, token: str) -> dict:
        """The token's claims if, and only if, every check passes."""
        try:
            h64, p64, s64 = token.split(".")
            header = json.loads(b64url(h64))
            claims = json.loads(b64url(p64))
            sig = b64url(s64)
        except (ValueError, AttributeError):
            raise CloudError("auth", "malformed token") from None
        if not isinstance(header, dict) or not isinstance(claims, dict):
            raise CloudError("auth", "malformed token")
        if header.get("alg") != "RS256":
            raise CloudError("auth", "token algorithm is not RS256")
        n, e = self._key(str(header.get("kid", "")))
        if not rs256_valid(f"{h64}.{p64}".encode("ascii"), sig, n, e):
            raise CloudError("auth", "bad token signature")

        now = self.clock()
        if claims.get("iss") not in ISSUERS:
            raise CloudError("auth", "token not issued by Google")
        aud = claims.get("aud")
        auds = aud if isinstance(aud, list) else [aud]
        if self.audience not in auds:
            raise CloudError("auth", "token is for another audience")
        try:
            exp, iat = float(claims["exp"]), float(claims.get("iat", 0))
        except (KeyError, TypeError, ValueError):
            raise CloudError("auth", "token has no expiry") from None
        if exp + SKEW_S < now:
            raise CloudError("auth", "token expired")
        if iat - SKEW_S > now:
            raise CloudError("auth", "token issued in the future")
        if str(claims.get("email", "")).lower() != self.email:
            raise CloudError("auth", "token names another identity")
        if claims.get("email_verified") is not True:
            raise CloudError("auth", "token email is not verified")
        return claims
