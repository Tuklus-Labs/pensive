"""Crockford base32 ULID generator (stdlib only).

A ULID is a 128-bit identifier: a 48-bit big-endian millisecond timestamp
followed by 80 bits of randomness, rendered as 26 Crockford base32 characters.
The encoding is fixed width and the alphabet is in ascending byte order, so the
lexical order of two ULID strings matches the numeric order of the underlying
128-bit values. Newer ids sort after older ones, which is why the store uses
them as primary keys instead of random UUIDs: rows land roughly in creation
order without a separate sort column.
"""
import os
import time

# Crockford base32: 0-9 and A-Z minus I, L, O, U (the letters that read as
# digits). Kept in ascending ASCII order so a fixed-width string sorts the same
# way its integer value does.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ENCODED_LEN = 26
_RANDOM_BITS = 80


def _encode(value, length):
    """Render ``value`` as ``length`` Crockford base32 chars, most significant
    first. 26 chars carry 130 bits, so a 128-bit ULID leaves the top two bits
    of the first character zero (the first char is always 0-7)."""
    out = bytearray(length)
    for i in range(length - 1, -1, -1):
        out[i] = ord(_ALPHABET[value & 0x1F])
        value >>= 5
    return out.decode("ascii")


def ulid(timestampMs=None):
    """Return a fresh 26-character Crockford base32 ULID.

    ``timestampMs`` overrides the millisecond timestamp component (for tests
    that need a deterministic time prefix); by default it is the current wall
    clock in unix milliseconds (UTC).
    """
    if timestampMs is None:
        timestampMs = int(time.time() * 1000)
    randomBits = int.from_bytes(os.urandom(_RANDOM_BITS // 8), "big")
    value = (timestampMs << _RANDOM_BITS) | randomBits
    return _encode(value, _ENCODED_LEN)
