"""A minimal PNG encoder (stdlib only).

The kymograph is naturally an image - one pixel per (nucleotide, time) - and
encoding it as a PNG lets the same bytes be embedded in an SVG figure, in the
HTML player and in a report, without pulling in an imaging library.  Only what
is needed is implemented: 8-bit RGBA, no interlacing, filter type 0.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def encode_rgba(pixels: np.ndarray) -> bytes:
    """Encode an ``(h, w, 4)`` uint8 array as PNG bytes."""
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("expected an (h, w, 4) RGBA array")
    data = np.ascontiguousarray(pixels, dtype=np.uint8)
    height, width = data.shape[:2]
    # each scanline is prefixed with its filter byte (0 = none)
    raw = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), data.reshape(height, width * 4)],
        axis=1,
    ).tobytes()
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        _SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def data_uri(pixels: np.ndarray) -> str:
    """``data:image/png;base64,...`` for an RGBA array."""
    return "data:image/png;base64," + base64.b64encode(encode_rgba(pixels)).decode(
        "ascii"
    )
