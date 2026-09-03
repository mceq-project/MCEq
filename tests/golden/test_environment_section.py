"""Structure of the ``environment`` section, read off the stored file.

The generator asserts its own invariants while it builds, which only happens
when the section is regenerated or compared — and comparing it needs
``--run-golden-host``, so a platform job skips it entirely. These tests read
the committed ``.npz`` instead, so the coverage the sweep claims and the
behaviours it pins are checked on every platform and in every job.

What they state: every concrete atmosphere class is either covered or named,
with a reason, in ``meta/uncovered``; no two cells stored the same numbers;
``set_h_obs`` resplines at zenith 60 and — because ``if self.theta_deg:`` is
false at ``0.0`` — does not at zenith 0, leaving that cell's splines bitwise
equal to the pre-move ones; ``max_theta`` is overwritten on a plain atmosphere
and preserved on a detector-centred one; the detector-centred zenith guard
raises ``ValueError`` where the base classes raise a bare ``Exception``; the
two ``geom.h`` call forms disagree at some of the geometries the sweep visits;
``gtracr_cutoff`` recognises IceCube and ORCA by coordinate and misses ARCA;
and the provenance stanza describes the file it was written beside.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import HOST_SECTIONS, SECTIONS, SLOW_SECTIONS
from ._harness import load_section, section_path
from .gen_environment import (
    ATMOSPHERES,
    CENTERED,
    H_OBS_CELLS,
    PROBE_HEIGHTS_CM,
    SECTION,
    X_FRACTIONS,
)

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def section():
    if not section_path(SECTION).exists():
        pytest.skip(f"golden section {SECTION!r} has not been generated")
    return load_section(SECTION)


def test_the_section_is_registered_host_pinned_and_not_slow():
    """`environment` is in `SECTIONS`, has a generator, and is `golden_host`.

    Not `golden_slow`: it opens no database and forks nothing, so CI runs it
    wherever the reference job runs. `golden_host` because its knot and
    coefficient keys are sha256 digests, which admit no tolerance.
    """
    from .make_goldens import GENERATORS

    assert SECTION in SECTIONS
    assert SECTION in GENERATORS
    assert SECTION in HOST_SECTIONS
    assert SECTION not in SLOW_SECTIONS


def test_every_concrete_atmosphere_class_is_covered_or_explained(section):
    """The coverage table is exhaustive, and the gap carries its reason.

    ``meta/concrete_classes`` is read live off the two modules by the
    generator, so a class a later phase adds lands here and fails until it is
    either swept or written into ``UNCOVERED`` with why.
    """
    arrays, _ = section
    concrete = set(arrays["meta/concrete_classes"].tolist())
    covered = set(arrays["meta/covered_classes"].tolist())
    reasons = arrays["meta/uncovered"].tolist()

    assert covered <= concrete, sorted(covered - concrete)
    for name in sorted(concrete - covered):
        assert any(reason.startswith(f"{name}:") for reason in reasons), (
            f"{name} is a concrete EarthsAtmosphere subclass that the "
            f"environment section neither sweeps nor explains in "
            f"meta/uncovered"
        )
    assert covered, "the section covers no atmosphere at all"


def test_no_two_cells_stored_the_same_numbers(section):
    """Every sweep axis moves something.

    The generator asserts this while it builds; here it is re-derived from the
    file so a section regenerated on a machine where, say, ``nrlmsis`` was
    missing cannot ship a degenerate cell.
    """
    arrays, prov = section
    cells = arrays["meta/cells"].tolist()
    assert prov["extra"]["n_cells"] == len(cells)
    assert prov["extra"]["n_distinct_cells"] == len(cells), (
        f"{len(cells) - prov['extra']['n_distinct_cells']} cell(s) duplicate another"
    )

    signatures = {}
    for cell in cells:
        prefix = f"cells/{cell}/"
        keys = sorted(key for key in arrays if key.startswith(prefix))
        assert keys, f"cell {cell!r} is in meta/cells and stored nothing"
        signatures[cell] = tuple(
            (key, np.asarray(arrays[key]).tobytes()) for key in keys
        )
    assert len(set(signatures.values())) == len(cells)


def test_the_probe_grids_have_the_widths_the_generator_declares(section):
    """`X_FRACTIONS` and `PROBE_HEIGHTS_CM` against the stored arrays.

    A shape drift here would silently narrow what the wrappers are pinned on.
    """
    arrays, prov = section
    assert prov["extra"]["x_fractions"] == list(X_FRACTIONS)
    assert prov["extra"]["probe_heights_cm"] == list(PROBE_HEIGHTS_CM)
    for cell in arrays["meta/cells"].tolist():
        key = f"cells/{cell}/probe/X"
        if key not in arrays:
            continue
        assert arrays[key].shape == (len(X_FRACTIONS),), cell
        assert arrays[f"cells/{cell}/probe/r_X2rho"].shape == (len(X_FRACTIONS),)
        assert arrays[f"cells/{cell}/probe/get_density"].shape == (
            arrays[f"cells/{cell}/probe/rho_h"].size,
        )


def test_set_h_obs_does_not_respline_at_zenith_zero(section):
    """``if self.theta_deg:`` is false at ``0.0``, so the spline goes stale.

    ``EarthsAtmosphere.set_h_obs`` moves the geometry unconditionally and
    resplines only under that guard. At zenith 0 the observation level is 2 km
    up and every spline is still the sea-level one — the ``theta0`` cell's
    three coefficient digests equal the ``corsika_usstd/theta0`` cell's
    exactly, and its ``X2h`` still lands at 0 cm at full depth instead of at
    the new 2e5.

    Pinned, not fixed: the fix is a decision about ``set_h_obs``, and this
    section exists so that decision is visible when it is taken.
    """
    arrays, _ = section
    stale = "cells/hobs/corsika_usstd/theta0_h200000"
    fresh = "cells/hobs/corsika_usstd/theta60_h200000"
    origin = "cells/corsika_usstd/theta0"

    assert int(arrays[f"{stale}/respline_fires"]) == 0
    assert int(arrays[f"{fresh}/respline_fires"]) == 1

    # The geometry moved in both.
    assert float(arrays[f"{stale}/geom"][0]) == 2.0e5
    assert float(arrays[f"{fresh}/geom"][0]) == 2.0e5

    # The splines moved only in the second.
    for name in ("s_h2X", "s_X2rho", "s_lX2h"):
        assert str(arrays[f"{stale}/{name}/coeffs"]) == str(
            arrays[f"{origin}/{name}/coeffs"]
        ), name
        assert str(arrays[f"{fresh}/{name}/coeffs"]) != str(
            arrays[f"{origin}/{name}/coeffs"]
        ), name

    # And the observable consequence: X2h at full depth.
    assert float(arrays[f"{stale}/probe/X2h"][-1]) == 0.0
    assert float(arrays[f"{fresh}/probe/X2h"][-1]) == pytest.approx(2.0e5)

    # Every other h_obs cell runs at a non-zero zenith and does respline.
    for label, _cls, _args, _kwargs, theta, h_obs in H_OBS_CELLS:
        cell = f"cells/hobs/{label}/theta{theta:g}_h{h_obs:g}"
        assert int(arrays[f"{cell}/respline_fires"]) == int(bool(theta)), cell


def test_set_h_obs_resets_max_theta_unless_the_model_preserves_it(section):
    """``_preserve_max_theta``, in both directions.

    A plain atmosphere takes ``geom.theta_max_deg`` — 90 becomes 91.44 once the
    observation level is 2 km up — while a detector-centred model, which allows
    upgoing angles, must keep its 180.
    """
    arrays, _ = section
    for label, _cls, _args, _kwargs, theta, h_obs in H_OBS_CELLS:
        cell = f"cells/hobs/{label}/theta{theta:g}_h{h_obs:g}"
        before = float(arrays[f"{cell}/max_theta_before"])
        after = float(arrays[f"{cell}/scalars"][2])
        assert before == 90.0, cell
        assert after == pytest.approx(float(arrays[f"{cell}/geom"][6])), cell
        assert after > before, cell

    before, after, geom_max = arrays["cells/hobs/msis00_icecube/preserved_max_theta"]
    assert before == 180.0
    assert after == 180.0, (
        "set_h_obs overwrote max_theta on a model with "
        "_preserve_max_theta = True, which silently forbids the upgoing angles "
        "it was constructed for"
    )
    assert geom_max < after, (
        "the geometry's own theta_max no longer differs from the preserved "
        "value, so the flag is untested here"
    )


def test_the_detector_centred_zenith_guard_raises_valueerror(section):
    """``ValueError``, where the base classes raise a bare ``Exception``.

    ``paths`` pins the base-class side (a bare ``Exception``, which
    ``pytest.raises(ValueError)`` does not catch). This is the other half, for
    every detector-centred class, and it is the half a caller can actually
    handle.

    ``guarded`` is a subset of the table rather than equal to it: without
    ``nrlmsis`` the four MSIS21 rows are absent from the file.
    """
    arrays, _ = section
    guarded = {
        key.split("/")[1]
        for key in arrays
        if key.startswith("cells/") and key.endswith("/zenith_guard")
    }
    assert guarded <= {label for label, *_ in CENTERED}
    assert guarded, "no zenith guard was recorded"
    for label in sorted(guarded):
        assert str(arrays[f"cells/{label}/zenith_guard"]) == "ValueError", label
        assert int(arrays[f"cells/{label}/depends_on_azimuth"]) == 1, label
        assert int(arrays[f"cells/{label}/preserve_max_theta"]) == 1, label


def test_the_two_geom_h_call_forms_disagree_at_some_geometries(section):
    """``geom_h/forms``: the array call against the scalar loop.

    Four rows per geometry — ``(thrad, h_obs, n_differing, max_abs_diff_cm)``.
    The pin is that the count is not zero everywhere and that where it is not,
    the residual is the one-ULP-of-``r_E`` endpoint cancellation and nothing
    larger. ``tests/geometry/test_environment_pins.py`` states the same thing
    as behaviour; this checks the stored numbers agree with it.
    """
    arrays, prov = section
    forms = arrays["geom_h/forms"]
    assert forms.ndim == 2 and forms.shape[1] == 4
    assert forms.shape[0] == prov["extra"]["n_geometries"]

    differing = forms[forms[:, 2] > 0]
    assert differing.shape[0] == prov["extra"]["n_geometries_where_h_forms_differ"]
    assert differing.shape[0] > 0, (
        "the array and scalar geom.h forms now agree at every geometry the "
        "sweep visits; if the four spline tails were unified, say which form "
        "was kept"
    )
    # One ULP of r_E = 6.37e8 cm. Two, for headroom on another libm.
    ulp = float(np.spacing(6371.315e5))
    assert float(differing[:, 3].max()) == pytest.approx(
        prov["extra"]["max_h_form_delta_cm"]
    )
    assert float(differing[:, 3].max()) <= 2 * ulp
    assert float(forms[forms[:, 2] == 0][:, 3].max(initial=0.0)) == 0.0


def test_the_gtracr_location_table_misses_arca(section):
    """IceCube and ORCA resolve by coordinate; ARCA does not.

    ``_gtracr_location_from_atmosphere`` tests ``lat 36.264 / lon 15.4``
    against the ``_KM3NET_DETECTORS`` entry's ``36.267 / 16.1``, so the ARCA
    branch never fires and the name falls through to the coordinate stub. The
    coordinates it returns are right; only the name tag is lost, and gtracr's
    own ``Location`` is then built from them — so this is pinned as the name it
    actually produces rather than reported as broken.

    An atmosphere with no geography raises ``ValueError``, which is what makes
    the geomagnetic-cutoff feature refuse a CORSIKA or isothermal profile.
    """
    arrays, _ = section
    table = {row[0]: row[1:] for row in arrays["gtracr/locations"].tolist()}

    assert table["MSIS00IceCubeCentered"][0] == "IceCube"
    assert table["MSIS00KM3NeTCentered/ORCA"][0] == "ORCA"
    assert table["MSIS00KM3NeTCentered/ARCA"][0].startswith("coord_lat+36p267"), (
        "the ARCA coordinate branch now matches, so the tolerance test in "
        "_gtracr_location_from_atmosphere was corrected; that changes the "
        "gtracr cache key and therefore every cached cutoff map"
    )
    assert table["MSIS00Atmosphere/Karlsruhe"][0] == "Karlsruhe"
    for label in ("CorsikaAtmosphere/BK_USStd", "IsothermalAtmosphere"):
        assert table[label][0] == "<ValueError>", label


def test_the_cutoff_mask_zeroes_a_species_below_its_rigidity_threshold(section):
    """``build_phi0_with_cutoff``: ``E_nucleus > Z * R_c``, per pixel.

    The pinned array is over five cutoffs, and what it has to satisfy is
    monotonicity: raising ``R_c`` can only remove flux, and ``R_c = 0`` removes
    none. Two rows of the stub state, the proton and the neutron block, must
    both carry flux — a phi0 that wrote them into the same rows would pass an
    L2 check and fail this.
    """
    arrays, _ = section
    phi0 = arrays["gtracr/phi0/values"]
    rc = arrays["gtracr/phi0/rc_GV"]
    assert phi0.shape[1] == rc.size
    assert tuple(arrays["gtracr/phi0/shape"].tolist()) == phi0.shape

    order = np.argsort(rc)
    totals = phi0.sum(axis=0)[order]
    assert np.all(np.diff(totals) <= 0.0), (
        f"total phi0 is not non-increasing in the rigidity cutoff: {totals}"
    )
    assert totals[0] > 0.0
    assert totals[-1] < totals[0], "the largest cutoff removed nothing"

    # Proton and neutron blocks are disjoint and both populated at R_c = 0.
    rows = arrays["gtracr/phi0/nonzero_rows"]
    dim = arrays["gtracr/phi0/e_grid"].size
    blocks = sorted({int(row) // dim for row in rows})
    assert blocks == [0, 2], (
        f"phi0 populated state blocks {blocks}; the stub run puts the proton "
        f"at block 0 and the neutron at block 2, so anything else means the "
        f"lidx/uidx slicing moved"
    )


def test_the_provenance_summary_describes_the_file(section):
    """The `extra` stanza the generator wrote against the arrays it wrote."""
    arrays, prov = section
    extra = prov["extra"]

    assert extra["cells"] == arrays["meta/cells"].tolist()
    assert extra["covered_classes"] == arrays["meta/covered_classes"].tolist()
    assert extra["concrete_classes"] == arrays["meta/concrete_classes"].tolist()
    assert extra["n_steps"] == int(arrays["meta/n_steps"])
    assert extra["have_nrlmsis"] == bool(int(arrays["meta/have_nrlmsis"]))

    for kind in ("knot", "coeff"):
        keys = [key for key in arrays if key.endswith(f"/{kind}s")]
        assert extra[f"n_{kind}_digests"] == len(keys)
        assert extra[f"n_distinct_{kind}_digests"] == len(
            {str(arrays[key]) for key in keys}
        )

    # The coefficients discriminate far better than the knots, which are
    # geometry: `n_distinct_knot_digests` is well below its key count because
    # cells at one (thrad, h_obs, h_form) share s_h2X knots across every
    # density model, while the coefficients collide only where set_h_obs
    # declined to respline.
    assert extra["n_distinct_coeff_digests"] > extra["n_distinct_knot_digests"]
    assert extra["n_geometries_shared_by_several_cells"] > 0

    # Every sweep axis is present in the stanza, so a trimmed table is visible.
    assert len(extra["zeniths"]) >= 4
    assert len(extra["single_azimuth_cells"]) >= 2
    assert {row[0] for row in extra["single_azimuth_cells"]} != {
        row[1] for row in extra["single_azimuth_cells"]
    }
    assert extra["blas_threads"]["cpu_count"]
    assert len(ATMOSPHERES) >= 6
