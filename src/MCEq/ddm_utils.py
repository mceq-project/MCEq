"""DDM matrix generation, spline evaluation and knot variations."""

from copy import copy
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from jacobi import propagate
from scipy.integrate import quad
from scipy.interpolate import splev

from MCEq.misc import _eval_energy_cuts, energy_grid, info

_PROPAGATE_PARAMS = dict(maxiter=10, maxgrad=1, method=0)
_QUAD_PARAMS = dict(limit=150, epsabs=1e-5)


def fmteb(ebeam: Union[float, str]) -> str:
    """Beam energy as the string key of the spline tables ("158.0")."""
    return f"{float(ebeam):.1f}"


def _spline_min_max_at_knot(
    tck: Tuple, iknot: int, sigma: npt.NDArray
) -> Tuple[Tuple, Tuple, float]:
    """Splines ``(tck_min, tck_max, h)`` with coefficient ``iknot`` shifted by
    ``-h`` and ``+h``, ``h = sigma[iknot]``."""
    assert iknot <= len(tck[1]) - 1, f"Invalid knot {iknot} requested"
    _c_min = np.copy(tck[1])
    _c_max = np.copy(tck[1])
    h = sigma[iknot]
    _c_min[iknot] = tck[1][iknot] - h
    _c_max[iknot] = tck[1][iknot] + h
    return (tck[0], _c_min, tck[-1]), (tck[0], _c_max, tck[-1]), h


def _generate_DDM_matrix(
    entries, mceq, e_min: float = -1.0, e_max: float = -1.0
) -> npt.NDArray:
    """DDM yield matrix of one channel (its ``entries``) on the grid of ``mceq``.

    The matrix covers the bins in ``[e_min, e_max]`` GeV (negative: grid edge).
    Between two splines the bin-averaged dN/dx is interpolated linearly in log
    beam energy; outside them the nearest spline is used.
    """
    projectile = entries[0].projectile
    secondary = entries[0].secondary
    # Definitions of particles and energy grid
    e_min = mceq._energy_grid.b[0] if e_min < 0 else e_min
    e_max = mceq._energy_grid.b[-1] if e_max < 0 else e_max
    mceq_egr = mceq._energy_grid
    min_idx, max_idx, _cuts = _eval_energy_cuts(mceq_egr.c, e_min, e_max)
    _energy_grid = energy_grid(
        mceq_egr.c[_cuts],
        mceq_egr.b[min_idx : max_idx + 1],
        mceq_egr.w[_cuts],
        int(max_idx - min_idx),
    )
    dim = len(_energy_grid.c)

    projectile_mass = mceq.pman[projectile].mass
    secondary_mass = mceq.pman[secondary].mass

    # Convert from kinetic to lab energy
    elab_secondary_bins = _energy_grid.b + secondary_mass
    elab_secondary_centers = _energy_grid.c + secondary_mass
    elab_proj_centers = _energy_grid.c + projectile_mass
    e_widths = np.diff(elab_secondary_bins)
    xgrid = elab_secondary_bins / elab_secondary_centers[-1]

    # Pull a copy of the original matrix
    try:
        # If matrices are stored on GPU, copy them to CPU
        mat = copy(mceq.pman[projectile].hadr_yields[mceq.pman[secondary]].get())
    except AttributeError:
        mat = copy(mceq.pman[projectile].hadr_yields[mceq.pman[secondary]])
    mat = mat[min_idx : max_idx + 1, min_idx : max_idx + 1]

    info(5, f"DDM will be used between {e_min:.1e} GeV and {e_max:.1e} GeV")
    assert entries, "Number of splines per channel can't be zero"

    dndx = []
    for entry in entries:
        try:
            dndx.append(_gen_averaged_dndx(xgrid, entry))
        except ValueError:
            raise Exception(
                "Error averaging cross section for "
                + f"({projectile},{secondary},{entry.ebeam})"
            )
    ie_beam = [np.argmin(np.abs(e.fl_ebeam - elab_proj_centers)) for e in entries]
    for i, (entry, ie) in enumerate(zip(entries, ie_beam)):
        info(
            5,
            f"Dataset {i} ({projectile}, {secondary}): mceq_eidx={ie}, "
            + f"mceq_ebeam={elab_proj_centers[ie]:4.3f}, ebeam={entry.ebeam}",
        )

    if len(entries) == 1:
        for ie, eproj in enumerate(elab_proj_centers[:-1]):
            if (eproj < e_min) or (eproj > e_max):
                info(10, "Skipping out of range energy", eproj)
                continue
            mat[: ie + 1, ie] = (
                dndx[0][dim - ie - 1 :] / elab_proj_centers[ie] * e_widths[ie]
            )
        return mat

    # Loop through datasets in pairs of two and interpolate between them
    n_intervals = len(entries) - 1
    for interval in range(n_intervals):
        entry_0, entry_1 = entries[interval], entries[interval + 1]
        ie0, ie1 = ie_beam[interval], ie_beam[interval + 1]
        dndx_0, dndx_1 = dndx[interval], dndx[interval + 1]
        for ie, eproj in enumerate(elab_proj_centers):
            if (eproj < e_min) or (eproj > e_max):
                continue
            if ie <= ie0 and interval == 0:
                # The first interval extrapolates down to the lowest energies
                mat[: ie + 1, ie] = dndx_0[dim - ie - 1 :] / eproj * e_widths[ie]
            elif ie < ie0 and interval > 0:
                # columns filled by the previous interval
                continue
            elif ie <= ie1:
                f0 = dndx_0[dim - ie - 1 :] / eproj * e_widths[ie]
                f1 = dndx_1[dim - ie - 1 :] / eproj * e_widths[ie]
                m = (f1 - f0) / np.log(entry_1.fl_ebeam / entry_0.fl_ebeam)
                mat[: ie + 1, ie] = f0 + m * np.log(eproj / entry_0.fl_ebeam)
            elif ie > ie1 and interval != n_intervals - 1:
                # columns left to the next interval
                break
            else:
                # The last interval extrapolates up to the highest energies
                mat[: ie + 1, ie] = dndx_1[dim - ie - 1 :] / eproj * e_widths[ie]

    return mat


def _eval_spline(
    x: Union[float, npt.NDArray],
    tck: Tuple[npt.NDArray, npt.NDArray, int],
    x17: bool,
    cov: npt.NDArray,
    return_error: Optional[bool] = False,
    gamma_zfac: Optional[float] = None,
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    """dN/dx of the spline ``tck`` at ``x``, and its propagated error if ``return_error``.

    ``x17`` removes the stored ``x**1.7`` weighting; ``gamma_zfac`` multiplies by
    ``x**gamma_zfac``.
    """
    if gamma_zfac is None:
        factor = x ** (-1.7) if x17 else 1.0
    else:
        factor = x ** (gamma_zfac) if not x17 else x ** (gamma_zfac - 1.7)

    def func(tck_1: npt.NDArray) -> npt.NDArray:
        return factor * np.exp(splev(x, (tck[0], tck_1, tck[2])))

    if return_error:
        y, C = propagate(func, tck[1], cov, **_PROPAGATE_PARAMS)
        sig_y = np.squeeze(np.sqrt(np.diag(np.atleast_1d(C))))
        return y, sig_y
    res = np.atleast_1d(func(tck[1]))
    res[(res < 0) | ~np.isfinite(res)] = 0.0
    return res.squeeze()


def _gen_averaged_dndx(xbins: np.ndarray, entry) -> np.ndarray:
    """Tuned dN/dx averaged over each bin of ``xbins``, integrated within
    ``[entry.x_min, 1]``."""
    xwidths = np.diff(xbins)
    integral = np.zeros_like(xwidths)

    for ib in range(len(integral)):
        if xbins[ib + 1] > 1:
            low_lim, up_lim = xbins[ib], 1.0
        elif xbins[ib + 1] < entry.x_min:
            continue
        else:
            low_lim, up_lim = max(xbins[ib], entry.x_min), xbins[ib + 1]
        integral[ib] = entry.tv * (
            quad(
                _eval_spline,
                low_lim,
                up_lim,
                args=(entry.tck, entry.x17, entry.cov),
                **_QUAD_PARAMS,
            )[0]
            / xwidths[ib]
        )

    return integral


def gen_matrix_variations(ddm_obj, mceq):
    """Matrices with each spline coefficient shifted by one sigma.

    Returns ``matrix_variations`` ``{(proj, sec): {ispl: {iknot: [mat_max,
    mat_min, h]}}}`` and, for K+- secondaries, ``isospin_partners``
    ``{(proj, sec): ((310, 130), variations)}`` of the K0 matrices.
    """
    matrix_variations = {}
    isospin_partners = {}

    ddm_matrices = ddm_obj.ddm_matrices(mceq)
    for (projectile, secondary), entries in ddm_obj.channels.items():
        info(1, f"Generating variations for channel {projectile} -> {secondary}")
        mat_db = {}
        iso_part_db = {}

        for ispl, spl_entry in enumerate(entries):
            tck = spl_entry.tck
            sig = spl_entry.knot_sigma
            mat_db[ispl] = {}
            iso_part_db[ispl] = {}

            for iknot in range(spl_entry.n_knots):
                if np.allclose(tck[1][iknot], 0.0):
                    continue

                tck_min, tck_max, h = _spline_min_max_at_knot(tck, iknot, sig)
                spl_entry.tck = tck_max
                mat_max = _generate_DDM_matrix(
                    entries, mceq, e_min=ddm_obj.e_min, e_max=ddm_obj.e_max
                )
                spl_entry.tck = tck_min
                mat_min = _generate_DDM_matrix(
                    entries, mceq, e_min=ddm_obj.e_min, e_max=ddm_obj.e_max
                )
                spl_entry.tck = tck

                mat_db[ispl][iknot] = [mat_max, mat_min, h]

                if secondary in [321, -321] and ddm_obj.enable_K0_from_isospin:
                    other = ddm_matrices[(projectile, -secondary)]
                    iso_part_db[ispl][iknot] = [
                        0.5 * (mat_max + other),
                        0.5 * (mat_min + other),
                        h,
                    ]

        matrix_variations[(projectile, secondary)] = mat_db
        if secondary in [321, -321]:
            isospin_partners[(projectile, secondary)] = ((310, 130), iso_part_db)

    return matrix_variations, isospin_partners
