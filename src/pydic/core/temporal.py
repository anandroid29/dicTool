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


def _packed_common(result, names):
    packed = []
    for name in names:
        field = getattr(result, name, None)
        if field is None:
            return np.zeros(0, np.uint32), [np.zeros(0, np.float32) for _ in names]
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


def _field(shape, indices, values) -> CompactField:
    return CompactField(shape, indices, values)


def compact_temporal_result(result):
    """Retain independent displayed fields and derive their cheap aliases."""
    from .analysis import PairResult

    shape = tuple(int(v) for v in result.u.shape)
    valid_idx, measurement = _packed_common(result, MEASUREMENT_FIELDS)
    rate_idx, rates = _packed_common(result, RATE_FIELDS)
    strain_idx, strains = _packed_common(result, STRAIN_FIELDS)

    u, v = (_field(shape, valid_idx, values) for values in measurement)
    mag = _field(shape, valid_idx, np.hypot(u.values, v.values).astype(np.float32))
    dt = max(float(getattr(result, "elapsed", 0.0)), 1e-12)
    vx, vy, veff = u.scaled(1.0 / dt), v.scaled(1.0 / dt), mag.scaled(1.0 / dt)
    exx_r, exy_r, eyy_r, eeff_r = (
        _field(shape, rate_idx, values) for values in rates)
    exx_g, exy_g, eyy_g, eeff_g = (
        _field(shape, strain_idx, values) for values in strains)

    out = PairResult(
        image_path=result.image_path, u=u, v=v,
        Exx=exx_g, Exy=exy_g, Eyy=eyy_g, Eeff=eeff_g,
        du_dx=None, du_dy=None, dv_dx=None, dv_dy=None, corr=None,
        u_inc=u, v_inc=v, mag_inc=mag,
        Vx=vx, Vy=vy, Veff=veff,
        dVx_dx=exx_r, dVx_dy=None, dVy_dx=None, dVy_dy=eyy_r,
        Exx_rate=exx_r, Exy_rate=exy_r, Gxy_rate=exy_r.scaled(2.0),
        Eyy_rate=eyy_r, Eeff_rate=eeff_r,
        valid=CompactMask(shape, valid_idx), elapsed=float(result.elapsed),
        Exx_gl=exx_g, Exy_gl=exy_g, Eyy_gl=eyy_g, Eeff_gl=eeff_g,
    )
    out.pair_start = int(getattr(result, "pair_start", -1))
    out.pair_end = int(getattr(result, "pair_end", -1))
    return out


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
            u=compact.u.values, v=compact.v.values,
            rate_indices=compact.Exx_rate.indices,
            Exx_rate=compact.Exx_rate.values,
            Exy_rate=compact.Exy_rate.values,
            Eyy_rate=compact.Eyy_rate.values,
            Eeff_rate=compact.Eeff_rate.values,
            strain_indices=compact.Exx_gl.indices,
            Exx_gl=compact.Exx_gl.values,
            Exy_gl=compact.Exy_gl.values,
            Eyy_gl=compact.Eyy_gl.values,
            Eeff_gl=compact.Eeff_gl.values,
        )
    os.replace(temporary, path)
    return compact


def load_temporal_result(path: str):
    """Load one compact pair without expanding any full-image field."""
    from .analysis import PairResult

    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(v) for v in data["shape"])
        a, b = (int(v) for v in data["pair"])
        elapsed = float(data["elapsed"])
        valid_idx = data["valid_indices"].astype(np.uint32, copy=False)
        rate_idx = data["rate_indices"].astype(np.uint32, copy=False)
        strain_idx = data["strain_indices"].astype(np.uint32, copy=False)
        u = _field(shape, valid_idx, data["u"])
        v = _field(shape, valid_idx, data["v"])
        mag = _field(shape, valid_idx, np.hypot(u.values, v.values).astype(np.float32))
        scale = 1.0 / max(elapsed, 1e-12)
        exx_r = _field(shape, rate_idx, data["Exx_rate"])
        exy_r = _field(shape, rate_idx, data["Exy_rate"])
        eyy_r = _field(shape, rate_idx, data["Eyy_rate"])
        eeff_r = _field(shape, rate_idx, data["Eeff_rate"])
        exx_g = _field(shape, strain_idx, data["Exx_gl"])
        exy_g = _field(shape, strain_idx, data["Exy_gl"])
        eyy_g = _field(shape, strain_idx, data["Eyy_gl"])
        eeff_g = _field(shape, strain_idx, data["Eeff_gl"])

    out = PairResult(
        image_path=f"pair {a + 1}→{b + 1}", u=u, v=v,
        Exx=exx_g, Exy=exy_g, Eyy=eyy_g, Eeff=eeff_g,
        du_dx=None, du_dy=None, dv_dx=None, dv_dy=None, corr=None,
        u_inc=u, v_inc=v, mag_inc=mag,
        Vx=u.scaled(scale), Vy=v.scaled(scale), Veff=mag.scaled(scale),
        dVx_dx=exx_r, dVx_dy=None, dVy_dx=None, dVy_dy=eyy_r,
        Exx_rate=exx_r, Exy_rate=exy_r, Gxy_rate=exy_r.scaled(2.0),
        Eyy_rate=eyy_r, Eeff_rate=eeff_r,
        valid=CompactMask(shape, valid_idx), elapsed=elapsed,
        Exx_gl=exx_g, Exy_gl=exy_g, Eyy_gl=eyy_g, Eeff_gl=eeff_g,
    )
    out.pair_start, out.pair_end = a, b
    return out


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
    """Read one temporal result from an exported PyDIC HDF5 subgroup."""
    import h5py
    from .analysis import PairResult

    with h5py.File(path, "r") as handle:
        group = handle["temporal_sequence"][f"pair_{int(index):06d}"]
        shape = tuple(int(v) for v in group.attrs["field_shape"])
        a = int(group.attrs["pair_start"])
        b = int(group.attrs["pair_end"])
        elapsed = float(group.attrs["elapsed_s"])
        valid_idx = group["valid_indices"][:].astype(np.uint32, copy=False)
        rate_idx = group["rate_indices"][:].astype(np.uint32, copy=False)
        strain_idx = group["strain_indices"][:].astype(np.uint32, copy=False)
        u = _field(shape, valid_idx, group["u"][:])
        v = _field(shape, valid_idx, group["v"][:])
        mag = _field(shape, valid_idx, np.hypot(u.values, v.values).astype(np.float32))
        exx_r = _field(shape, rate_idx, group["Exx_rate"][:])
        exy_r = _field(shape, rate_idx, group["Exy_rate"][:])
        eyy_r = _field(shape, rate_idx, group["Eyy_rate"][:])
        eeff_r = _field(shape, rate_idx, group["Eeff_rate"][:])
        exx_g = _field(shape, strain_idx, group["Exx_gl"][:])
        exy_g = _field(shape, strain_idx, group["Exy_gl"][:])
        eyy_g = _field(shape, strain_idx, group["Eyy_gl"][:])
        eeff_g = _field(shape, strain_idx, group["Eeff_gl"][:])

    scale = 1.0 / max(elapsed, 1e-12)
    out = PairResult(
        image_path=f"pair {a + 1}→{b + 1}", u=u, v=v,
        Exx=exx_g, Exy=exy_g, Eyy=eyy_g, Eeff=eeff_g,
        du_dx=None, du_dy=None, dv_dx=None, dv_dy=None, corr=None,
        u_inc=u, v_inc=v, mag_inc=mag,
        Vx=u.scaled(scale), Vy=v.scaled(scale), Veff=mag.scaled(scale),
        dVx_dx=exx_r, dVx_dy=None, dVy_dx=None, dVy_dy=eyy_r,
        Exx_rate=exx_r, Exy_rate=exy_r, Gxy_rate=exy_r.scaled(2.0),
        Eyy_rate=eyy_r, Eeff_rate=eeff_r,
        valid=CompactMask(shape, valid_idx), elapsed=elapsed,
        Exx_gl=exx_g, Exy_gl=exy_g, Eyy_gl=eyy_g, Eeff_gl=eeff_g,
    )
    out.pair_start, out.pair_end = a, b
    return out


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
