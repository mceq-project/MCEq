"""Sparse-matrix checks the tests share.

``canonical_csr_problems`` is the one property a checksum cannot see:
MKL and cuSPARSE both require sorted, duplicate-free CSR indices.
"""

from __future__ import annotations

import hashlib

import numpy as np


def array_digest(arr) -> str:
    """sha256 over dtype, shape and the C-contiguous bytes of `arr`.

    dtype is part of the digest, so a `float32` run is distinguishable from a
    `float64` one rather than colliding with it.
    """
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sparse_digest(mat) -> dict:
    """Digest a scipy sparse matrix without disturbing it.

    Copies before any canonicalisation: `sort_indices()` mutates in place and
    MKL sparse handles hold raw pointers into the live `data`/`indices`
    buffers, so sorting a matrix that a backend has bound corrupts that handle.

    The copy is also why the digest says nothing about canonical form — the
    matrix it hashes is canonical whatever came in. That is the job of
    :func:`canonical_csr_problems`.
    """
    m = mat.tocsr(copy=True)
    m.sort_indices()
    return {
        "shape": list(m.shape),
        "nnz": int(m.nnz),
        "dtype": m.dtype.str,
        "data": array_digest(m.data),
        "indices": array_digest(m.indices),
        "indptr": array_digest(m.indptr),
    }


def canonical_csr_problems(mat, label="matrix") -> list:
    """Every way `mat` is not a canonical CSR, as sentences. Empty when it is.

    Canonical is what MKL and cuSPARSE assume of the buffers they bind: one
    entry per `(row, column)`, columns ascending within a row. Unsorted indices
    are *invisible* to every digest in this harness — :func:`sparse_digest`
    sorts a copy before hashing, so a matrix that came back with two entries of
    a row transposed hashes identically to the sorted one and compares green
    against the stored digest. Duplicate entries do move the hash, but only as
    an unexplained digest change that regenerating the section absorbs, after
    which the section pins a non-canonical operator forever. Either way the
    live buffers a backend has bound are wrong and nothing here says so.

    The buffers are checked directly, not through `has_sorted_indices` /
    `has_canonical_format`: scipy caches both flags on first read, so an
    in-place edit of `indices` leaves them answering True over unsorted data.
    Each flag is then checked *against* the buffers, sorting for the first and
    sorting-plus-uniqueness for the second, because a backend that reads a flag
    and skips its own sort is misled by a stale one exactly as badly as by
    unsorted data.
    """
    problems = []
    if mat.format != "csr":
        return [f"{label}: format {mat.format!r}, not 'csr'"]

    n_rows, n_cols = mat.shape
    indptr, indices, data = mat.indptr, mat.indices, mat.data

    if len(indptr) != n_rows + 1:
        return [f"{label}: indptr has {len(indptr)} entries, expected {n_rows + 1}"]
    if indptr[0] != 0:
        problems.append(f"{label}: indptr[0] is {indptr[0]}, not 0")
    if len(data) != len(indices):
        problems.append(
            f"{label}: {len(data)} values against {len(indices)} column indices"
        )
    if indptr[-1] != len(indices):
        problems.append(
            f"{label}: indptr[-1] is {indptr[-1]}, {len(indices)} column indices"
        )
    if np.any(np.diff(indptr) < 0):
        problems.append(f"{label}: indptr is not non-decreasing")
    if problems:
        return problems

    nnz = len(indices)
    if nnz and (indices.min() < 0 or indices.max() >= n_cols):
        problems.append(
            f"{label}: column index out of range [0, {n_cols}): "
            f"[{indices.min()}, {indices.max()}]"
        )

    descending = duplicate = 0
    if nnz > 1:
        # Strictly increasing within every row is sorted *and* duplicate-free.
        # A step counts only when it stays inside a row: position p ends its
        # row exactly when p + 1 starts one, which `indptr` marks directly and
        # correctly for empty rows too.
        starts = np.zeros(nnz + 1, dtype=bool)
        starts[indptr] = True
        inside = ~starts[1:nnz]
        steps = np.diff(indices)
        descending = int(np.count_nonzero((steps < 0) & inside))
        duplicate = int(np.count_nonzero((steps == 0) & inside))
        if descending:
            problems.append(f"{label}: {descending} descending column step(s)")
        if duplicate:
            problems.append(f"{label}: {duplicate} duplicate (row, column) entry/ies")

    for flag, truth in (
        ("has_sorted_indices", not descending),
        ("has_canonical_format", not (descending or duplicate)),
    ):
        claim = bool(getattr(mat, flag))
        if claim != truth:
            problems.append(
                f"{label}: {flag} is {claim} and the buffers say {truth}; "
                f"a backend that trusts the flag skips work it needs"
            )
    return problems
