"""Disk-backed storage for completed temporal-pair results.

A dense pair result contains many full-image arrays whose useful values exist
only at DIC subset centres. Keeping hundreds of those results in RAM merely to
make seeking/export deterministic is not viable. Each generated pair is packed
to one small, independently writable NPZ file; worker threads can therefore
finish out of order without sharing an HDF5 handle.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .compact_field import CompactField, CompactMask


MEASUREMENT_FIELDS = ("u", "v")
RATE_FIELDS = ("Exx_rate", "Exy_rate", "Eyy_rate", "Eeff_rate")
STRAIN_FIELDS = ("Exx_gl", "Exy_gl", "Eyy_gl", "Eeff_gl")


def _packed_common(result, names, *, fill_missing: bool = True):
    """Align a field family, padding missing NPZ fields or omitting HDF5 fields."""
    packed = []
    for name in names:
        field = getattr(result, name, None)
        if field is None:
            values = [np.zeros(0, np.float32) for _ in names] if fill_missing else []
            return np.zeros(0, np.uint32), values
        if isinstance(field, CompactField):
            indices = field.indices
            values = field.values
        else:
            dense = np.asarray(field)
            indices = np.flatnonzero(np.isfinite(dense).reshape(-1)).astype(
                np.uint32, copy=False)
            values = dense.reshape(-1)[indices].astype(np.float32, copy=False)
        packed.append((indices, values))
    common = packed[0][0]
    for indices, _ in packed[1:]:
        common = np.intersect1d(common, indices, assume_unique=True)
    values = [data[np.searchsorted(indices, common)].astype(np.float32, copy=False)
              for indices, data in packed]
    return common.astype(np.uint32, copy=False), values


def _make_temporal_result(image_path, shape, measurement, rates, strains,
                          elapsed, pair):
    """Build a packed pair and its shared aliases for every storage backend."""
    from .analysis import PairResult

    valid_idx, values = measurement
    u, v = (CompactField(shape, valid_idx, value) for value in values)
    mag = CompactField(shape, valid_idx, np.hypot(u.values, v.values).astype(np.float32))
    scale = 1.0 / max(float(elapsed), 1e-12)
    rate_idx, values = rates
    exx_r, exy_r, eyy_r, eeff_r = (
        CompactField(shape, rate_idx, value) for value in values)
    strain_idx, values = strains
    exx_g, exy_g, eyy_g, eeff_g = (
        CompactField(shape, strain_idx, value) for value in values)

    out = PairResult(
        image_path=image_path, u=u, v=v,
        Exx=exx_g, Exy=exy_g, Eyy=eyy_g, Eeff=eeff_g,
        du_dx=None, du_dy=None, dv_dx=None, dv_dy=None, corr=None,
        u_inc=u, v_inc=v, mag_inc=mag,
        Vx=u.scaled(scale), Vy=v.scaled(scale), Veff=mag.scaled(scale),
        dVx_dx=exx_r, dVx_dy=None, dVy_dx=None, dVy_dy=eyy_r,
        Exx_rate=exx_r, Exy_rate=exy_r, Gxy_rate=exy_r.scaled(2.0),
        Eyy_rate=eyy_r, Eeff_rate=eeff_r,
        valid=CompactMask(shape, valid_idx), elapsed=float(elapsed),
        Exx_gl=exx_g, Exy_gl=exy_g, Eyy_gl=eyy_g, Eeff_gl=eeff_g,
    )
    out.pair_start, out.pair_end = map(int, pair)
    return out


def compact_temporal_result(result):
    """Retain independent displayed fields and derive their cheap aliases."""
    return _make_temporal_result(
        result.image_path, tuple(int(v) for v in result.u.shape),
        _packed_common(result, MEASUREMENT_FIELDS),
        _packed_common(result, RATE_FIELDS),
        _packed_common(result, STRAIN_FIELDS), result.elapsed,
        (getattr(result, "pair_start", -1), getattr(result, "pair_end", -1)))


def save_temporal_result(path: str, result):
    """Pack and atomically save one pair, returning its compact result."""
    compact = compact_temporal_result(result)
    temporary = path + ".part"
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            shape=np.asarray(compact.u.shape, np.int64),
            pair=np.asarray((compact.pair_start, compact.pair_end), np.int64),
            elapsed=np.asarray(compact.elapsed, np.float64),
            valid_indices=compact.u.indices,
            rate_indices=compact.Exx_rate.indices,
            strain_indices=compact.Exx_gl.indices,
            **{name: getattr(compact, name).values
               for name in MEASUREMENT_FIELDS + RATE_FIELDS + STRAIN_FIELDS},
        )
    os.replace(temporary, path)
    return compact


def load_temporal_result(path: str):
    """Load one compact pair without expanding any full-image field."""
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(v) for v in data["shape"])
        a, b = (int(v) for v in data["pair"])
        return _read_temporal_fields(data, shape, a, b, float(data["elapsed"]))


def _read_temporal_fields(data, shape, a, b, elapsed):
    """Read the common NPZ/HDF5 field layout while its file is open."""
    def packed(prefix, names):
        indices = data[f"{prefix}_indices"][:].astype(np.uint32, copy=False)
        return indices, [data[name][:] for name in names]

    return _make_temporal_result(
        f"pair {a + 1}→{b + 1}", shape,
        packed("valid", MEASUREMENT_FIELDS), packed("rate", RATE_FIELDS),
        packed("strain", STRAIN_FIELDS), elapsed, (a, b))


class TemporalResultSequence(Sequence):
    """Read-only sequence facade over a directory of numbered pair files."""

    def __init__(self, directory: str, pairs) -> None:
        self.directory = os.path.abspath(directory)
        self.pairs = list(pairs)

    def path_for(self, index: int) -> str:
        return os.path.join(self.directory, f"pair_{int(index):06d}.npz")

    def has(self, index: int) -> bool:
        return 0 <= int(index) < len(self) and os.path.isfile(self.path_for(index))

    def completed_count(self) -> int:
        return sum(self.has(i) for i in range(len(self)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return load_temporal_result(self.path_for(index))


def load_hdf5_temporal_result(path: str, index: int):
    """Read one temporal result from an exported strainX HDF5 subgroup."""
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle["temporal_sequence"][f"pair_{int(index):06d}"]
        shape = tuple(int(v) for v in group.attrs["field_shape"])
        a = int(group.attrs["pair_start"])
        b = int(group.attrs["pair_end"])
        return _read_temporal_fields(
            group, shape, a, b, float(group.attrs["elapsed_s"]))


class HDF5TemporalResultSequence(Sequence):
    """On-demand temporal results from a saved session, safe across threads."""

    def __init__(self, path: str, pairs) -> None:
        self.path = os.path.abspath(path)
        self.pairs = [tuple(int(v) for v in pair) for pair in pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def has(self, index: int) -> bool:
        return 0 <= int(index) < len(self)

    def completed_count(self) -> int:
        return len(self)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return load_hdf5_temporal_result(self.path, index)
