"""Compact storage for sparse full-image DIC fields.

DIC values exist only at subset centres.  Keeping those values in an H x W
array therefore spends almost all of its memory on NaNs.  These containers keep
the original image shape for API compatibility while storing only flat pixel
indices and the values that exist there.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class CompactField:
    """A sparse numeric image whose missing pixels read as NaN."""

    __slots__ = ("shape", "indices", "values")
    __array_priority__ = 1000

    def __init__(self, shape, indices, values, *, copy: bool = False) -> None:
        self.shape = tuple(int(v) for v in shape)
        if len(self.shape) != 2:
            raise ValueError("CompactField shape must be two-dimensional")
        self.indices = np.asarray(indices, dtype=np.uint32)
        self.values = np.asarray(values, dtype=np.float32)
        if self.indices.ndim != 1 or self.values.ndim != 1:
            raise ValueError("CompactField indices and values must be one-dimensional")
        if self.indices.size != self.values.size:
            raise ValueError("CompactField indices and values must have equal length")
        if copy:
            self.indices = self.indices.copy()
            self.values = self.values.copy()

    @classmethod
    def from_dense(cls, values, valid: Optional[np.ndarray] = None,
                   indices: Optional[np.ndarray] = None) -> "CompactField":
        dense = np.asarray(values)
        if dense.ndim != 2:
            raise ValueError("A compact DIC field must originate from a 2-D array")
        flat = dense.reshape(-1)
        if indices is None:
            keep = np.isfinite(flat)
            if valid is not None:
                keep &= np.asarray(valid, dtype=bool).reshape(-1)
            indices = np.flatnonzero(keep).astype(np.uint32, copy=False)
        else:
            indices = np.asarray(indices, dtype=np.uint32)
            selected = flat[indices]
            finite = np.isfinite(selected)
            if not finite.all():
                indices = indices[finite]
        return cls(dense.shape, indices, flat[indices])

    @classmethod
    def empty(cls, shape) -> "CompactField":
        return cls(shape, np.zeros(0, np.uint32), np.zeros(0, np.float32))

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return int(self.shape[0] * self.shape[1])

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def nbytes(self) -> int:
        return int(self.indices.nbytes + self.values.nbytes)

    def __len__(self) -> int:
        return self.shape[0]

    def finite_values(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]

    def to_dense(self, dtype=None) -> np.ndarray:
        dtype = self.values.dtype if dtype is None else np.dtype(dtype)
        out = np.full(self.shape, np.nan, dtype=dtype)
        if self.indices.size:
            out.reshape(-1)[self.indices] = self.values.astype(dtype, copy=False)
        return out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = self.to_dense(dtype)
        return out.copy() if copy else out

    def __getitem__(self, key):
        if (isinstance(key, tuple) and len(key) == 2 and
                all(isinstance(v, (int, np.integer)) for v in key)):
            y, x = int(key[0]), int(key[1])
            if y < 0:
                y += self.shape[0]
            if x < 0:
                x += self.shape[1]
            if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
                raise IndexError(key)
            flat = np.uint32(y * self.shape[1] + x)
            pos = int(np.searchsorted(self.indices, flat))
            if pos < self.indices.size and self.indices[pos] == flat:
                return self.values[pos]
            return np.float32(np.nan)
        return self.to_dense()[key]

    def copy(self) -> "CompactField":
        return CompactField(self.shape, self.indices, self.values, copy=True)

    def scaled(self, factor: float) -> "CompactField":
        return CompactField(self.shape, self.indices,
                            self.values * np.float32(factor))

    def __mul__(self, factor):
        if np.isscalar(factor):
            return self.scaled(float(factor))
        return self.to_dense() * factor

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __truediv__(self, divisor):
        if np.isscalar(divisor):
            return self.scaled(1.0 / float(divisor))
        return self.to_dense() / divisor


class CompactMask:
    """A boolean image represented solely by the indices that are true."""

    __slots__ = ("shape", "indices")

    def __init__(self, shape, indices, *, copy: bool = False) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.indices = np.asarray(indices, dtype=np.uint32)
        if self.indices.ndim != 1:
            raise ValueError("CompactMask indices must be one-dimensional")
        if copy:
            self.indices = self.indices.copy()

    @property
    def ndim(self) -> int:
        return 2

    @property
    def size(self) -> int:
        return int(self.shape[0] * self.shape[1])

    @property
    def dtype(self):
        return np.dtype(bool)

    @property
    def nbytes(self) -> int:
        return int(self.indices.nbytes)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = np.zeros(self.shape, dtype=bool)
        out.reshape(-1)[self.indices] = True
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out.copy() if copy else out

    def __getitem__(self, key):
        if (isinstance(key, tuple) and len(key) == 2 and
                all(isinstance(v, (int, np.integer)) for v in key)):
            y, x = int(key[0]), int(key[1])
            if y < 0:
                y += self.shape[0]
            if x < 0:
                x += self.shape[1]
            if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
                raise IndexError(key)
            flat = np.uint32(y * self.shape[1] + x)
            pos = int(np.searchsorted(self.indices, flat))
            return bool(pos < self.indices.size and self.indices[pos] == flat)
        return np.asarray(self)[key]

    def copy(self) -> "CompactMask":
        return CompactMask(self.shape, self.indices, copy=True)

    def any(self) -> bool:
        return bool(self.indices.size)

    def sum(self) -> int:
        return int(self.indices.size)

    def astype(self, dtype, copy=True):
        return np.asarray(self, dtype=dtype).copy() if copy else np.asarray(self, dtype=dtype)


def finite_values(values) -> np.ndarray:
    """Return finite stored samples without expanding a compact field."""
    if isinstance(values, CompactField):
        return values.finite_values()
    arr = np.asarray(values)
    return arr[np.isfinite(arr)]
