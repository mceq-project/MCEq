"""Materialized reads for one MCEq HDF5 file: the IO seam of the backend.

Every method opens the file for the duration of the call and returns
numpy arrays or plain containers; no live ``h5py`` node ever escapes.
The module imports ``h5py``, ``numpy``, ``scipy.sparse`` and nothing from
``MCEq``.  Decoding policy -- the ``disabled_particles`` filter
interleaved with the read cursor, equivalence injection, the rho-stack
choice -- stays with the backend, so :meth:`HDF5Store.read_channel_pack`
returns the raw :class:`ChannelPack`, not a decoded channel index.
Design: section 13.1 of the layered-architecture refactor plan.
"""

from typing import NamedTuple

import h5py
import numpy as np
from scipy.sparse import csr_matrix


class ChannelPack(NamedTuple):
    """One channel-matrix dataset and its ``_indptrs`` sibling.

    ``data`` is ``dset[:, :]`` in the stored dtype -- no cast applied,
    the dtype cast is backend policy; ``indptrs`` is
    ``<name>_indptrs[:]``; ``len_data`` and ``tuple_idcs`` are the
    attributes of those names; ``description`` is the optional
    ``description`` attribute or ``None``.
    """

    data: np.ndarray
    indptrs: np.ndarray
    len_data: np.ndarray
    tuple_idcs: np.ndarray
    description: str | None


class HDF5Store:
    """Materialized reader for one MCEq HDF5 file.

    Construction opens nothing -- the EM database may be absent while
    ``enable_em`` is off, and the backend builds a store for it anyway.
    Every read opens and closes the file itself; nothing is cached,
    mirroring the pre-split ``HDF5Backend``, which had no caching and
    reopened whichever file(s) each public call needed.
    """

    def __init__(self, fname):
        self.fname = fname

    def attrs(self, path="/") -> dict:
        """Attributes of ``path`` as a plain dict (``dict(f[path].attrs)``)."""
        with h5py.File(self.fname, "r") as f:
            return dict(f[path].attrs)

    def members(self, path="/") -> list[str]:
        """Child names of ``path``, in file order."""
        with h5py.File(self.fname, "r") as f:
            return list(f[path])

    def has(self, path) -> bool:
        """``path in f``."""
        with h5py.File(self.fname, "r") as f:
            return path in f

    def read_dataset(self, path) -> np.ndarray:
        """The dataset at ``path`` as ``np.asarray(f[path][()])``."""
        with h5py.File(self.fname, "r") as f:
            return np.asarray(f[path][()])

    def read_table(self, path) -> tuple:
        """``(f[path][:], dict(f[path].attrs))`` of the table at ``path``."""
        with h5py.File(self.fname, "r") as f:
            return f[path][:], dict(f[path].attrs)

    def read_channel_pack(self, group_path, name) -> ChannelPack:
        """The pack ``group_path/name`` with its ``_indptrs`` sibling."""
        with h5py.File(self.fname, "r") as f:
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
        with h5py.File(self.fname, "r") as f:
            group = f[path]
            return {k: np.asarray(group[k]) for k in group}


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

    Verbatim extraction of the two unpack branches of
    ``HDF5Backend._gen_db_dictionary``.  ``mat_data[0]`` carries the CSR
    values and ``mat_data[1]`` the column indices of the whole pack;
    ``indptr_row`` is one row of the sibling ``indptrs`` dataset;
    ``start`` / ``length`` locate the channel's entries in the flat
    arrays, ``length`` being the already-expanded read length
    (``expand_len * len_data[tupidx]``, the stride that scales with
    ``n_k`` in 2D databases).  ``is_2d`` is explicit and never inferred
    from ``n_k``: a 2D database with ``n_k == 1`` still yields
    ``(1, n_e, n_e)``.  A 1D channel yields ``(n_e, n_e)``.
    """
    if is_2d:
        # The stored channel matrix is block-diagonal in kappa
        # (modes are decoupled). Slice each per-mode block out of
        # the sparse matrix and energy-cut it straight into a
        # preallocated ``(n_k, n_e, n_e)`` tensor -- the dense
        # ``(dim_full, dim_full)`` intermediate would be n_k times
        # larger than the data it carries.
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
