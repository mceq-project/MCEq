"""Build tiny, valid MCEq HDF5 databases for the decode tests.

These are *synthetic* databases, small enough to build in a fraction of a
second, so they are made into a tmp directory by the session-scoped
``hdf5_fixture_dbs`` fixture in ``tests/conftest.py`` and never committed as
``.h5``. What they buy is the two decode branches no shipped database
reaches: a 2-column ``tuple_idcs`` and the malformed width that trips the
``"Failed decoding parent-child relation."`` raise in
``MCEq.data.HDF5Backend._gen_db_dictionary``.

Layout
------
Written from what the reader actually requires (``HDF5Backend.__init__`` and
``_gen_db_dictionary``), cross-checked against
``src/MCEq/data/mceq_db_v140reduced_compact.h5`` (1D, v1.4.0) and
``mceq_db_v2_fluka2d_rc7.h5`` (2D, 48 modes) with h5py:

* ``/common`` carries the grid entirely in **attributes**, not datasets:
  ``e_grid`` (n_e,), ``e_bins`` (n_e+1,), ``widths`` (n_e,), scalar
  ``e_dim``; a 2D file adds scalar ``k_dim`` and ``k_grid`` (n_k,).
  ``e_grid`` is the geometric centre of ``e_bins`` and ``widths`` is
  ``diff(e_bins)`` (verified on the reduced database).
* A channel pack is a ``(2, N)`` float64 dataset -- row 0 CSR ``data``, row 1
  CSR ``indices`` **stored as float64** -- with attrs ``tuple_idcs``
  (n_ch, width) int64, ``len_data`` (n_ch,) int64 and an optional
  ``description``; the sibling dataset ``<NAME>_indptrs`` has shape
  (n_ch, dim_full+1).
* ``len_data[i]`` is the nnz of **one Hankel-mode block**, not of the whole
  channel: on the rc7 2D file ``data.shape[1] == len_data.sum() * n_k``
  (``decays/polarized``: 13721232 == 285859 * 48; ``decays/unpolarized``:
  11822448 == 246301 * 48) and each channel's indptr rises by exactly
  ``len_data[i]`` across every one of the 48 mode blocks. Both invariants
  are asserted below, so a malformed fixture fails here rather than
  decoding to a wrong-but-finite matrix.
* ``/cross_sections/<medium>/<MODEL>`` is ``(n_e, n_parents)`` with a
  ``parents`` attribute. ``/continuous_losses/<medium>/{ionization,total}``
  holds one ``(n_e,)`` dataset per PDG-as-string plus a ``(2, M)``
  ``hadron`` dataset of ``(beta*gamma, dE/dX)``. Sign convention copied from
  the reduced database: the per-lepton curves are **negative**, the two
  ``hadron`` rows are **positive** (``Losses.load`` takes ``np.log`` of both
  hadron rows, so a negative one produces a RuntimeWarning and a nan spline).

Facts verified while building this, worth not rediscovering
-----------------------------------------------------------
* A decay pack that stops at ``pi/K -> mu nu`` cannot even reach particle
  setup. ``MCEqRun`` on such a file raises ``Exception('Unstable particle
  without decay distribution:', (-13, np.int64(0)), 'mu+')`` from
  ``particlemanager.MCEqParticle.set_decay_channels``. Hence
  ``mu+- -> e nu nu`` in :data:`DECAY_CHANNELS`. These fixtures are still not
  ``MCEqRun``-able -- with the muon channels present the same probe gets one
  step further and raises ``Exception('No nucleons in eqn system, primary
  flux model can not be used.')``, the fixture having no nucleon channels in
  its decay pack. Nothing here builds an ``MCEqRun``; the tests drive
  ``HDF5Backend`` directly.
* The 8-mode kappa grid is a **subsample** of the 48-value rc7 production
  grid, not its head. A head truncation (``kappa = 0..7``) loads fine but
  then fails ``MCEq.operators.secant.build_secant_kernel_ops`` with
  ``RuntimeError: secant coupling: S_P eigenvalues not positive-real (min
  real -1.384e+01, ...)`` -- whose text blames ``theta_cap_deg`` even though
  the cap was the default 75. The subsample in :data:`K_GRID` builds
  successfully (eig(S_P) in [1.00006, 1.0547] at cap 75; ~40 s, so no test
  here does it).

Usage (writes the four variants into a directory, for eyeballing with h5py)::

    .venv/bin/python tests/data/make_hdf5_fixtures.py /tmp/fixtures
"""

import pathlib

import h5py
import numpy as np
from scipy.sparse import block_diag, csr_matrix

#: Energy grid: 20 bins at 10/decade starting at 1 GeV, so 1 -- 100 GeV.
N_E = 20
E_BINS = 10.0 ** (np.arange(N_E + 1) / 10.0)
E_GRID = np.sqrt(E_BINS[:-1] * E_BINS[1:])
WIDTHS = np.diff(E_BINS)

#: The 48 kappa values of the production 2D database
#: ``mceq_db_v2_fluka2d_rc7.h5`` (``/common`` attr ``k_grid``).
# fmt: off
K_GRID_PRODUCTION = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 21, 24, 28, 33, 38, 43,
    50, 58, 67, 77, 89, 102, 118, 136, 156, 180, 208, 239, 276, 318, 366, 422,
    486, 560, 645, 743, 856, 986, 1136, 1308, 1507, 1736, 2000,
], dtype=np.int64)
# fmt: on
N_K = 8
#: Evenly spaced *subsample* of the production grid -- both endpoints kept.
K_GRID = K_GRID_PRODUCTION[
    np.linspace(0, len(K_GRID_PRODUCTION) - 1, N_K).round().astype(int)
]

MEDIUM = "air"
#: A name ``_interaction_db_single`` can map to an equivalence table. SIBYLL21
#: is chosen because its table contains ``-2212 -> 2212``, which is what the
#: pinned ``available_parents`` defect needs to become visible.
MODEL = "SIBYLL21"

#: ``(parent_pdg, child_pdg)``, in file order. The antiproton rows come first
#: so that the SIBYLL21 equivalence copy from the proton, which happens later
#: in the loop, can overwrite them on the width-2 file.
#:
#: The trailing ``(-211, 3112)`` row exists as a positive control for the
#: equivalence machinery. SIBYLL21 maps ``3112 -> 2212``; naming 3112 as a
#: *child* puts it in ``model_particles`` (which is built from columns 0 and 2)
#: without putting it in ``available_parents``, so it is the one equivalence
#: replacement that actually fires on this fixture. Every other SIBYLL21
#: parent mapping onto 2212 (3222, 3312, the anti-hyperons) is filtered out by
#: the ``eqv_parent[0] not in model_particles`` guard, and 111 -> 211 likewise;
#: without this row the only replacement candidate would be the antiproton,
#: and a decode that never ran the copy loop at all would be indistinguishable
#: from one whose ``available_parents`` guard worked.
#: It is a ``-211`` parent on purpose: the proton's child list -- which the
#: width-2 defect aliases onto the antiproton -- must stay unchanged.
INTERACTION_CHANNELS = [
    (-2212, -2212),
    (-2212, 211),
    (-2212, -211),
    (2212, 2212),
    (2212, 211),
    (2212, -211),
    (211, 211),
    (-211, -211),
    (-211, 3112),
]

#: Channel 4 and channel 7 have an ``e+-`` child, so the default
#: ``disabled_particles = [11, -11]`` skips them -- which is what the
#: flat-cursor tests use.
DECAY_CHANNELS = [
    (211, -13),
    (211, 14),
    (-211, 13),
    (-211, -14),
    (-13, -11),
    (-13, 12),
    (-13, -14),
    (13, 11),
    (13, -12),
    (13, 14),
]

CS_PARENTS = [-2212, -211, 211, 2212]
LOSS_PDGS = ["-15", "-13", "-11", "11", "13", "15"]

#: Half-bandwidth of the nonzero pattern of a channel block.
BAND = 3
_ROWS, _COLS = np.nonzero(
    np.triu(np.ones((N_E, N_E)), 0) - np.triu(np.ones((N_E, N_E)), BAND + 1)
)

#: name -> file name of the four variants :func:`build_all` writes.
VARIANTS = {
    "one_d_width4": "fixture_1d_w4.h5",
    "one_d_width2": "fixture_1d_w2.h5",
    "one_d_width3": "fixture_1d_w3.h5",
    "two_d_width4": "fixture_2d_w4.h5",
}


def channel_block(channel_index, mode_index=0):
    """The ``(N_E, N_E)`` matrix stored for one channel and Hankel mode.

    Analytic rather than random so a test can assert the decoded matrix
    element by element. Every entry of the band is strictly positive, so the
    CSR pattern is the band and nnz does not depend on the indices.
    """
    values = (
        (1.0 + channel_index)
        * np.exp(-(_COLS - _ROWS) / 2.0)
        / (1.0 + _ROWS)
        / (1.0 + mode_index) ** 2
    )
    block = np.zeros((N_E, N_E))
    block[_ROWS, _COLS] = values
    return block


def pack_channels(channels, n_k, tuple_width):
    """Pack channels the way a database stores them.

    Returns ``(data, indptrs, len_data, tuple_idcs)``. In 2D each channel is
    the block-diagonal matrix of its ``n_k`` mode blocks, exactly as the
    production 2D database stores it.
    """
    dim_full = N_E * n_k
    data_parts, index_parts, indptrs, len_data, tuples = [], [], [], [], []

    for ich, (parent, child) in enumerate(channels):
        blocks = [csr_matrix(channel_block(ich, ik)) for ik in range(n_k)]
        nnz = {b.nnz for b in blocks}
        assert len(nnz) == 1, (
            f"channel {ich}: mode blocks have differing nnz {sorted(nnz)}; the "
            "stored indptr encodes one stride per channel, so every mode block "
            "must share the sparsity pattern"
        )
        full = csr_matrix(block_diag(blocks, format="csr"))
        assert full.shape == (dim_full, dim_full)

        len_data.append(nnz.pop())
        data_parts.append(full.data)
        # Row 1 of the pack is the CSR index array, stored as float64.
        index_parts.append(full.indices.astype(np.float64))
        indptrs.append(full.indptr.astype(np.int64))

        if tuple_width == 4:
            tuples.append((parent, 0, child, 0))
        elif tuple_width == 2:
            tuples.append((parent, child))
        elif tuple_width == 3:
            # Deliberately malformed: no decode branch accepts width 3.
            tuples.append((parent, 0, child))
        else:
            raise ValueError(f"unsupported tuple_idcs width {tuple_width}")

    data = np.vstack([np.concatenate(data_parts), np.concatenate(index_parts)])
    indptrs = np.array(indptrs, dtype=np.int64)
    len_data = np.array(len_data, dtype=np.int64)
    tuple_idcs = np.array(tuples, dtype=np.int64)

    _check_pack(data, indptrs, len_data, tuple_idcs, n_k)
    return data, indptrs, len_data, tuple_idcs


def _check_pack(data, indptrs, len_data, tuple_idcs, n_k):
    """Fail loudly here rather than decode to a wrong-but-finite matrix."""
    dim_full = N_E * n_k
    n_ch = len(len_data)

    assert data.shape == (2, int(len_data.sum()) * n_k), (
        f"pack is {data.shape}, expected (2, {int(len_data.sum()) * n_k}): the "
        "flat read length of a channel is n_k * len_data[i]"
    )
    assert indptrs.shape == (n_ch, dim_full + 1), (
        f"indptrs are {indptrs.shape}, expected ({n_ch}, {dim_full + 1})"
    )
    assert tuple_idcs.shape[0] == n_ch
    assert data[1].min() >= 0.0 and data[1].max() < dim_full, (
        "CSR column indices leave the (dim_full, dim_full) matrix"
    )

    for i in range(n_ch):
        indptr = indptrs[i]
        assert indptr[0] == 0, f"channel {i}: indptr does not start at 0"
        assert indptr[-1] == n_k * len_data[i], (
            f"channel {i}: indptr ends at {indptr[-1]}, expected "
            f"n_k * len_data[i] = {n_k * len_data[i]}"
        )
        for ik in range(n_k):
            stride = indptr[(ik + 1) * N_E] - indptr[ik * N_E]
            assert stride == len_data[i], (
                f"channel {i}, mode {ik}: indptr stride {stride} disagrees "
                f"with len_data[i] = {len_data[i]}"
            )


def _write_pack(group, name, channels, n_k, tuple_width, description, layout=None):
    data, indptrs, len_data, tuple_idcs = pack_channels(channels, n_k, tuple_width)
    dset = group.create_dataset(name, data=data)
    dset.attrs["len_data"] = len_data
    dset.attrs["tuple_idcs"] = tuple_idcs
    if description is not None:
        dset.attrs["description"] = description
    if layout is not None:
        dset.attrs["layout"] = layout
    group.create_dataset(name + "_indptrs", data=indptrs)


def build_database(path, *, n_k=1, tuple_width=4):
    """Write one synthetic database and return its path."""
    path = pathlib.Path(path)
    is_2d = n_k > 1
    # The malformed-width file is also the one that omits the optional
    # ``description`` attribute, so it reaches the reader's ``description =
    # None`` fallback on its way to the decode raise.
    described = tuple_width != 3

    with h5py.File(path, "w") as db:
        db.attrs["version"] = "0.0.0-fixture"

        common = db.create_group("common")
        common.attrs["e_grid"] = E_GRID
        common.attrs["e_bins"] = E_BINS
        common.attrs["widths"] = WIDTHS
        common.attrs["e_dim"] = N_E
        if is_2d:
            # The only 2D marker the backend looks at.
            common.attrs["k_dim"] = n_k
            common.attrs["k_grid"] = K_GRID[:n_k]

        _write_pack(
            db.create_group(f"hadronic_interactions/{MEDIUM}"),
            MODEL,
            INTERACTION_CHANNELS,
            n_k,
            tuple_width,
            f"synthetic {MODEL} yields" if described else None,
        )

        decays = db.create_group("decays")
        for dset_name in ("polarized", "unpolarized"):
            _write_pack(
                decays,
                dset_name,
                DECAY_CHANNELS,
                n_k,
                tuple_width,
                f"synthetic {dset_name} decays" if described else None,
                # decay_db() refuses a 2D 'polarized' set without this.
                layout="superset" if (is_2d and dset_name == "polarized") else None,
            )

        cross_sections = db.create_group(f"cross_sections/{MEDIUM}")
        table = 1e-3 * np.outer(1.0 + np.log10(E_GRID), 1 + np.arange(len(CS_PARENTS)))
        cs_dset = cross_sections.create_dataset(MODEL, data=table)
        cs_dset.attrs["parents"] = np.array(CS_PARENTS, dtype=np.int64)

        for loss_case in ("ionization", "total"):
            group = db.create_group(f"continuous_losses/{MEDIUM}/{loss_case}")
            for pdg in LOSS_PDGS:
                # Negative, as in the shipped database.
                group.create_dataset(pdg, data=-2e-3 * (1.0 + 0.1 * np.log(E_GRID)))
            boost = np.logspace(-3, 1, 30)
            group.create_dataset(
                # Both rows positive: Losses.load takes np.log of each.
                "hadron",
                data=np.vstack([boost, 2e-3 * (1.0 + 1.0 / boost)]),
            )

    return path


def build_all(directory):
    """Write every variant into ``directory``; return ``{name: path}``."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "one_d_width4": build_database(
            directory / VARIANTS["one_d_width4"], n_k=1, tuple_width=4
        ),
        "one_d_width2": build_database(
            directory / VARIANTS["one_d_width2"], n_k=1, tuple_width=2
        ),
        "one_d_width3": build_database(
            directory / VARIANTS["one_d_width3"], n_k=1, tuple_width=3
        ),
        "two_d_width4": build_database(
            directory / VARIANTS["two_d_width4"], n_k=N_K, tuple_width=4
        ),
    }


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, written in build_all(target).items():
        print(f"{name}: {written}")
