"""Read-only access to one MCEq HDF5 file, returning numpy arrays."""

from contextlib import contextmanager
from typing import NamedTuple

import h5py
import numpy as np
from scipy.sparse import csr_matrix


class ChannelPack(NamedTuple):
    """One channel-matrix dataset and its ``_indptrs`` sibling.

    ``data`` is ``dset[:, :]`` in the stored dtype, ``indptrs`` is
    ``<name>_indptrs[:]``, ``len_data`` and ``tuple_idcs`` are the dataset
    attributes of those names, ``description`` is the optional attribute.
    """

    data: np.ndarray
    indptrs: np.ndarray
    len_data: np.ndarray
    tuple_idcs: np.ndarray
    description: str | None


class HDF5Store:
    """Reader for one MCEq HDF5 file.

    Construction opens nothing, so the file may be absent until first read.
    Inside :meth:`session` reads share one handle; otherwise each read opens
    and closes the file.
    """

    def __init__(self, fname):
        self.fname = fname
        self._file = None

    @contextmanager
    def session(self):
        """Reuse one read handle; nested sessions and exceptions close safely."""
        if self._file is not None:
            yield self
            return
        with h5py.File(self.fname, "r") as handle:
            self._file = handle
            try:
                yield self
            finally:
                self._file = None

    @contextmanager
    def _open(self):
        if self._file is not None:
            yield self._file
        else:
            with h5py.File(self.fname, "r") as handle:
                yield handle

    def attrs(self, path="/") -> dict:
        """Attributes of ``path`` as a plain dict (``dict(f[path].attrs)``)."""
        with self._open() as f:
            return dict(f[path].attrs)

    def members(self, path="/") -> list[str]:
        """Child names of ``path``, in file order."""
        with self._open() as f:
            return list(f[path])

    def has(self, path) -> bool:
        """``path in f``."""
        with self._open() as f:
            return path in f

    def read_dataset(self, path) -> np.ndarray:
        """The dataset at ``path`` as ``np.asarray(f[path][()])``."""
        with self._open() as f:
            return np.asarray(f[path][()])

    def read_table(self, path) -> tuple:
        """``(f[path][:], dict(f[path].attrs))`` of the table at ``path``."""
        with self._open() as f:
            return f[path][:], dict(f[path].attrs)

    def read_optional_table(self, path, attrs_path="common"):
        """``(f[path][:], dict(f[path].attrs), dict(f[attrs_path].attrs))`` in
        one open, or None when ``path`` is absent."""
        with self._open() as f:
            if path not in f:
                return None
            return f[path][:], dict(f[path].attrs), dict(f[attrs_path].attrs)

    def read_channel_pack(self, group_path, name) -> ChannelPack:
        """The pack ``group_path/name`` with its ``_indptrs`` sibling."""
        with self._open() as f:
            group = f[group_path]
            attrs = dict(group[name].attrs)
            return ChannelPack(
                data=group[name][:, :],
                indptrs=group[name + "_indptrs"][:],
                len_data=attrs["len_data"],
                tuple_idcs=attrs["tuple_idcs"],
                description=attrs["description"] if "description" in attrs else None,
            )

    def read_group_datasets(self, path) -> dict:
        """``{k: np.asarray(group[k]) for k in group}`` for ``path``."""
        with self._open() as f:
            group = f[path]
            return {k: np.asarray(group[k]) for k in group}


def angular_basis(common, fname):
    """``"legendre"`` for a 2D ``common`` group (``k_dim`` set), ``None`` for 1D.

    The modes of a 2D file are the Legendre moments ``<P_l(cos theta)>`` with
    ``l = k_grid``, declared by ``angular_basis = "legendre"``. Other 2D files
    raise ``ValueError``.
    """
    if "k_dim" not in common:
        return None
    basis = common.get("angular_basis")
    basis = basis.decode() if isinstance(basis, bytes) else basis
    if basis != "legendre":
        raise ValueError(
            f"{fname} is a 2D database with angular_basis = {basis!r}. MCEq "
            "reads 2D databases booked in Legendre moments only (common "
            "attribute angular_basis = 'legendre'); Hankel-mode 2D files, such "
            "as mceq_base_air_2d_v2.h5 of v2.0.0, are not supported."
        )
    return basis


def unpack_channel(
    mat_data,
    indptr_row,
    start,
    length,
    *,
    dim_full,
    is_2d,
    n_k,
    min_idx,
    max_idx,
    cuts,
) -> np.ndarray:
    """Decode one CSR-packed channel to its dense, energy-cut matrix.

    ``mat_data[0]`` holds the CSR values and ``mat_data[1]`` the column
    indices of the whole pack; ``indptr_row`` is one row of the ``indptrs``
    dataset; ``start``/``length`` locate the channel in the flat arrays
    (``length`` already scaled by ``n_k`` in 2D). Returns ``(n_e, n_e)`` in 1D
    and ``(n_k, n_e, n_e)`` when ``is_2d``, also for ``n_k == 1``.
    """
    if is_2d:
        # The stored matrix is block-diagonal in the mode; cut each
        # per-mode block directly from the sparse matrix.
        channel_csr = csr_matrix(
            (
                mat_data[0, start : start + length],
                mat_data[1, start : start + length],
                indptr_row,
            ),
            shape=(dim_full, dim_full),
        )
        n_e_full = dim_full // n_k
        n_e = max_idx - min_idx
        tensor = np.empty((n_k, n_e, n_e), dtype=mat_data.dtype)
        for ik in range(n_k):
            lo = ik * n_e_full + min_idx
            hi = ik * n_e_full + max_idx
            tensor[ik] = channel_csr[lo:hi, lo:hi].toarray()
        return tensor
    return np.asarray(
        (
            csr_matrix(
                (
                    mat_data[0, start : start + length],
                    mat_data[1, start : start + length],
                    indptr_row,
                ),
                shape=(dim_full, dim_full),
            )[cuts, min_idx:max_idx]
        ).toarray()
    )
