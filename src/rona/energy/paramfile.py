"""Reader for Turner nearest-neighbour parameter files (ViennaRNA ``.par`` v2.0).

The file shipped in :mod:`rona.params` is the Turner 2004 set in the
widely-used ``RNAfold parameter file v2.0`` layout.  Parsing it - rather than
hard-coding a hand-typed subset - means :mod:`rona.energy.model` reproduces the
published nearest-neighbour model exactly, including the 1x1/2x1/2x2 interior
loop tables and the special tri-/tetra-/hexaloop bonuses.

Energies are stored in dekacalories per mole (as in the file); the energy model
converts to kcal/mol at the boundary.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

#: Sentinel used by the file format for forbidden configurations.
INF = 10_000_000
#: "Default" terminal mismatch used for entries involving unknown bases.
DEF = -50
#: Energy assigned to non-standard stacked pairs.
NST = 0

NBPAIRS = 7  # NP, CG, GC, GU, UG, AU, UA  (+ the NS slot makes 8 conceptually)
NBASES = 5  # N, A, C, G, U

_TOKEN = re.compile(r"[^\s]+")
_COMMENT = re.compile(r"/\*.*?\*/", re.S)

# Each section maps to ``(shape_in_file, padding_spec)``.
#
# The file indexes its pair dimensions from ``CG`` (there is no "no pair" slot)
# and ``int22`` additionally omits the unknown-base slot.  The padding spec says
# how to lift each axis into the uniform encoding used throughout rona:
#   'P' -> pair axis of 7 (CG..NS)  -> 8, index 0 = NP
#   'Q' -> pair axis of 6 (CG..UA)  -> 8, indices 0 and 7 unused
#   'B' -> base axis of 5 (N,A,C,G,U), already aligned
#   'b' -> base axis of 4 (A,C,G,U) -> 5, index 0 = N
#   'n' -> plain axis, left alone
_ARRAY_SECTIONS: dict[str, tuple[tuple[int, ...], str]] = {}


def _register(section: str, shape: tuple[int, ...], pad: str, *, enthalpy: bool = True) -> None:
    _ARRAY_SECTIONS[section] = (shape, pad)
    if enthalpy:
        _ARRAY_SECTIONS[section + "_enthalpies"] = (shape, pad)


_register("stack", (7, 7), "PP")
for _mm in (
    "mismatch_hairpin",
    "mismatch_internal",
    "mismatch_internal_1n",
    "mismatch_internal_23",
    "mismatch_multi",
    "mismatch_exterior",
):
    _register(_mm, (7, 5, 5), "PBB")
_register("dangle5", (7, 5), "PB")
_register("dangle3", (7, 5), "PB")
_register("int11", (7, 7, 5, 5), "PPBB")
_register("int21", (7, 7, 5, 5, 5), "PPBBB")
_register("int22", (6, 6, 4, 4, 4, 4), "QQbbbb")
_register("hairpin", (31,), "n")
_register("bulge", (31,), "n")
_register("internal", (31,), "n")
_ARRAY_SECTIONS["ML_params"] = ((6,), "n")
_ARRAY_SECTIONS["NINIO"] = ((3,), "n")
_ARRAY_SECTIONS["Misc"] = ((6,), "n")

#: Sizes of each padded axis, keyed by the padding-spec character.
_PADDED_SIZE = {"P": NBPAIRS + 1, "Q": NBPAIRS + 1, "B": NBASES, "b": NBASES}
#: Offset at which the file data is placed along each padded axis.
_PAD_OFFSET = {"P": 1, "Q": 1, "B": 0, "b": 1}


def _pad(raw: np.ndarray, pad: str) -> np.ndarray:
    """Lift a raw table onto rona's uniform pair-type / base encoding."""
    if pad == "n" or all(c not in _PADDED_SIZE for c in pad):
        return raw
    shape = tuple(_PADDED_SIZE.get(c, n) for c, n in zip(pad, raw.shape))
    out = np.full(shape, float(INF))
    index = tuple(
        slice(_PAD_OFFSET.get(c, 0), _PAD_OFFSET.get(c, 0) + n)
        for c, n in zip(pad, raw.shape)
    )
    out[index] = raw
    return out


_LOOP_SECTIONS = {"Hexaloops": 8, "Tetraloops": 6, "Triloops": 5}


class ParameterError(ValueError):
    """Raised when a parameter file cannot be interpreted."""


def _value(token: str) -> float:
    if token.startswith("INF"):
        return float(INF)
    if token.startswith("DEF"):
        return float(DEF)
    if token.startswith("NST"):
        return float(NST)
    try:
        return float(token)
    except ValueError as exc:  # pragma: no cover - malformed file
        raise ParameterError(f"cannot parse parameter token {token!r}") from exc


def _split_sections(text: str) -> dict[str, str]:
    body = _COMMENT.sub(" ", text)
    sections: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            continue
        if stripped.startswith("#"):
            if name is not None:
                sections[name] = "\n".join(buf)
            name = stripped[1:].strip()
            buf = []
            if name == "END":
                break
            continue
        if name is not None:
            buf.append(line)
    if name is not None and name != "END":
        sections[name] = "\n".join(buf)
    return sections


@dataclass(slots=True)
class TurnerParams:
    """Raw Turner tables in dekacal/mol, plus the matching enthalpy tables.

    Attributes mirror the section names of the parameter file.  ``int22`` is
    stored padded to the full ``[7][7][5][5][5][5]`` shape so every table can be
    indexed with the same pair-type and base encodings.
    """

    name: str = "turner2004"
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    tetraloops: dict[str, tuple[float, float]] = field(default_factory=dict)
    triloops: dict[str, tuple[float, float]] = field(default_factory=dict)
    hexaloops: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    # -- scalars -----------------------------------------------------------
    @property
    def ml_base(self) -> float:
        return float(self.arrays["ML_params"][0])

    @property
    def ml_base_dh(self) -> float:
        return float(self.arrays["ML_params"][1])

    @property
    def ml_closing(self) -> float:
        return float(self.arrays["ML_params"][2])

    @property
    def ml_closing_dh(self) -> float:
        return float(self.arrays["ML_params"][3])

    @property
    def ml_intern(self) -> float:
        return float(self.arrays["ML_params"][4])

    @property
    def ml_intern_dh(self) -> float:
        return float(self.arrays["ML_params"][5])

    @property
    def ninio(self) -> float:
        return float(self.arrays["NINIO"][0])

    @property
    def ninio_dh(self) -> float:
        return float(self.arrays["NINIO"][1])

    @property
    def ninio_max(self) -> float:
        return float(self.arrays["NINIO"][2])

    @property
    def duplex_init(self) -> float:
        return float(self.arrays["Misc"][0])

    @property
    def terminal_au(self) -> float:
        return float(self.arrays["Misc"][2])

    @property
    def terminal_au_dh(self) -> float:
        return float(self.arrays["Misc"][3])

    @property
    def lxc(self) -> float:
        return float(self.arrays["Misc"][4])


def parse(text: str, *, name: str = "turner2004") -> TurnerParams:
    """Parse the text of a ViennaRNA-format v2.0 parameter file."""
    sections = _split_sections(text)
    params = TurnerParams(name=name)

    for section, (shape, pad) in _ARRAY_SECTIONS.items():
        if section not in sections:
            raise ParameterError(f"missing section '{section}' in parameter file")
        tokens = _TOKEN.findall(sections[section])
        values = [_value(t) for t in tokens]
        want = math.prod(shape)
        if len(values) != want:
            raise ParameterError(
                f"section '{section}': expected {want} values, found {len(values)}"
            )
        raw = np.asarray(values, dtype=np.float64).reshape(shape)
        params.arrays[section] = _pad(raw, pad)

    for section, length in _LOOP_SECTIONS.items():
        table: dict[str, tuple[float, float]] = {}
        for line in sections.get(section, "").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            motif = parts[0].upper().replace("T", "U")
            if len(motif) != length:
                raise ParameterError(
                    f"{section}: motif {motif!r} should have length {length}"
                )
            table[motif] = (_value(parts[1]), _value(parts[2]))
        setattr(params, section.lower(), table)

    return params


def load(path: str | Path | None = None) -> TurnerParams:
    """Load a parameter file, defaulting to the bundled Turner 2004 set."""
    if path is None:
        data = (
            resources.files("rona.params")
            .joinpath("rna_turner2004.par")
            .read_text(encoding="utf-8")
        )
        return parse(data, name="turner2004")
    p = Path(path)
    return parse(p.read_text(encoding="utf-8"), name=p.stem)


_CACHE: dict[str, TurnerParams] = {}


def default_params() -> TurnerParams:
    """Cached accessor for the bundled Turner 2004 parameters."""
    if "turner2004" not in _CACHE:
        _CACHE["turner2004"] = load()
    return _CACHE["turner2004"]
