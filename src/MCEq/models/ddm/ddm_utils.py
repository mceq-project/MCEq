"""DDM matrix generation, spline evaluation and knot variations."""

from copy import copy
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from jacobi import propagate
from scipy.integrate import quad
from scipy.interpolate import splev

from MCEq.misc import info

_LIMIT_PROPAGATE = np.inf
_PROPAGATE_PARAMS = dict(maxiter=10, maxgrad=1, method=0)
_QUAD_PARAMS = dict(limit=150, epsabs=1e-5)


def fmteb(ebeam: Union[float, str]) -> str:
    """
    Format beam energy for handling in db.

    Parameters
    ----------
    ebeam : Union[float, str, int]
        The beam energy.

    Returns
    -------
    str
        The formatted beam energy.
    """
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
    _tck_min = (tck[0], _c_min, tck[-1])
    _tck_max = (tck[0], _c_max, tck[-1])
    return _tck_min, _tck_max, h


def _generate_DDM_matrix(
    channel, mceq, e_min: float = -1.0, e_max: float = -1.0, average: bool = True
) -> npt.NDArray:
    """DDM yield matrix of ``channel`` on the grid of ``mceq``.

    Columns of projectile energies in ``[e_min, e_max]`` GeV (negative: grid
    edge) are replaced by the DDM spectrum, interpolated linearly in log
    beam energy between the splines; ``average`` uses bin-averaged dN/dx.
    """
    from MCEq.data.energy_grid import EnergyGrid, _eval_energy_cuts

    projectile = channel.projectile
    secondary = channel.secondary
    # Definitions of particles and energy grid
    e_min = mceq._energy_grid.b[0] if e_min < 0 else e_min
    e_max = mceq._energy_grid.b[-1] if e_max < 0 else e_max
    mceq_egr = mceq._energy_grid
    min_idx, max_idx, _cuts = _eval_energy_cuts(mceq_egr.c, e_min, e_max)
    _energy_grid = EnergyGrid(
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
    if average:
        dndx_generator = _gen_averaged_dndx
    else:
        dndx_generator = _gen_dndx

    info(
        5,
        f"DDM will be used between {e_min:.1e} GeV and {e_max:.1e} GeV",
    )
    n_datasets = channel.n_splines
    assert n_datasets > 0, "Number of splines per channel can't be zero"

    if n_datasets == 1:
        entry = channel.get_entry(idx=0)
        ie0 = np.argmin(np.abs(entry.fl_ebeam - elab_proj_centers))
        info(
            5,
            f"Dataset 0 ({projectile}, {secondary}): mceq_eidx={ie0}, "
            + "mceq_ebeam={mceq_ebeam:4.3f}, ebeam={entry.ebeam}",
        )
        averaged_dndx = dndx_generator(xgrid, entry)

        for ie, eproj in enumerate(elab_proj_centers[:-1]):
            if (eproj < e_min) or (eproj > e_max):
                info(0, "Skipping out of range energy", eproj)
                continue

            mat[: ie + 1, ie] = (
                averaged_dndx[dim - ie - 1 :] / elab_proj_centers[ie] * e_widths[ie]
            )
    else:
        # Loop through datasets in pairs of two and interpolate between them
        for interval in range(n_datasets - 1):
            entry_0 = channel.get_entry(idx=interval)
            entry_1 = channel.get_entry(idx=interval + 1)
            ie0 = np.argmin(np.abs(entry_0.fl_ebeam - elab_proj_centers))
            ie1 = np.argmin(np.abs(entry_1.fl_ebeam - elab_proj_centers))
            mceq_ebeam_0 = elab_proj_centers[ie0]
            mceq_ebeam_1 = elab_proj_centers[ie1]
            info(
                5,
                f"Dataset {interval} ({projectile}, {secondary}): "
                + f"mceq_eidx={ie0}, mceq_ebeam={mceq_ebeam_0:4.3f}, "
                + f"ebeam={entry_0.ebeam}",
            )
            info(
                5,
                f"Dataset {interval + 1} ({projectile}, {secondary}): "
                + f"mceq_eidx={ie1}, mceq_ebeam={mceq_ebeam_1:4.3f}, "
                + f"ebeam={entry_1.ebeam}",
            )

            try:
                averaged_dndx_0 = dndx_generator(xgrid, entry_0)
            except ValueError:
                raise Exception(
                    "Error averaging cross section for "
                    + f"({projectile},{secondary},{entry_0.ebeam})"
                )

            try:
                averaged_dndx_1 = dndx_generator(xgrid, entry_1)
            except ValueError:
                raise Exception(
                    "Error averaging cross section for "
                    + f"({projectile},{secondary},{entry_1.ebeam})"
                )

            for ie, eproj in enumerate(elab_proj_centers):
                if (eproj < e_min) or (eproj > e_max):
                    continue
                if ie <= ie0 and interval == 0:
                    # If it is the first interval, extrapolate from the lowest energies
                    mat[: ie + 1, ie] = (
                        averaged_dndx_0[dim - ie - 1 :] / eproj * e_widths[ie]
                    )
                elif ie < ie0 and interval > 0:
                    # if it is not the first interval, don't touch the columns
                    # that have been previously filled
                    continue
                elif ie <= ie1:
                    # Interpolate between two datasets
                    f0 = averaged_dndx_0[dim - ie - 1 :] / eproj * e_widths[ie]
                    f1 = averaged_dndx_1[dim - ie - 1 :] / eproj * e_widths[ie]
                    m = (f1 - f0) / np.log(entry_1.fl_ebeam / entry_0.fl_ebeam)
                    mat[: ie + 1, ie] = f0 + m * np.log(eproj / entry_0.fl_ebeam)
                elif ie > ie1 and interval != n_datasets - 2:
                    # if it is not the first dataset, don't touch the columns
                    # that have been previously filled
                    break
                else:
                    mat[: ie + 1, ie] = (
                        averaged_dndx_1[dim - ie - 1 :] / eproj * e_widths[ie]
                    )

    return mat


def _eval_spline(
    x: Union[float, npt.NDArray],
    tck: Tuple[npt.NDArray, npt.NDArray, int],
    x17: bool,
    cov: npt.NDArray,
    return_error: Optional[bool] = False,
    gamma_zfac: Optional[float] = None,
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    """
    Evaluate the spline in dn/dx.

    Parameters
    ----------
    x : float or numpy.ndarray
        Evaluation points.
    tck : Tuple[numpy.ndarray, numpy.ndarray, int]
        Spline tuple ``(knots, coefficients, degree)``.
    x17 : bool
        If True, remove the stored ``x**1.7`` weighting. Otherwise, the spline
        is taken as dn/dx.
    cov : numpy.ndarray
        The covariance matrix of the spline.
    return_error : bool, optional
        If True, return the propagated error of the fit. Default is False.
    gamma_zfac : float, optional
        Supplies an additional power of x. If None, it is not used. Default is None.

    Returns
    -------
    numpy.ndarray or Tuple[numpy.ndarray, numpy.ndarray]
        The value of the fit at each knot point. If `return_error` is True,
        it also returns the propagated error of the fit.
    """
    if gamma_zfac is None:
        factor = x ** (-1.7) if x17 else 1.0
    else:
        factor = x ** (gamma_zfac) if not x17 else x ** (gamma_zfac - 1.7)

    def func(tck_1: npt.NDArray) -> npt.NDArray:
        return factor * np.exp(splev(x, (tck[0], tck_1, tck[2])))

    func_params = tck[1]

    if return_error:
        y, C = propagate(func, func_params, cov, **_PROPAGATE_PARAMS)
        sig_y = np.squeeze(np.sqrt(np.diag(np.atleast_1d(C))))
        return y, sig_y
    res = np.atleast_1d(func(func_params))
    res[(res < 0) | ~np.isfinite(res) | (res > _LIMIT_PROPAGATE)] = 0.0
    return res.squeeze()


def _gen_dndx(xbins: np.ndarray, entry) -> np.ndarray:
    """Tuned dN/dx at the geometric bin centres of ``xbins``, zero below
    ``entry.x_min``."""

    x = np.sqrt(xbins[1:] * xbins[:-1])

    res = _eval_spline(x, entry.tck, entry.x17, entry.cov, return_error=False)
    res[x < entry.x_min] = 0
    return entry.tv * res


def _gen_averaged_dndx(xbins: np.ndarray, entry) -> np.ndarray:
    """Tuned dN/dx averaged over each bin of ``xbins``, integrated within
    ``[entry.x_min, 1]``; one value per bin."""

    xwidths = np.diff(xbins)
    integral = np.zeros_like(xwidths)

    for ib in range(len(integral)):
        if xbins[ib + 1] > 1:
            integral[ib] = entry.tv * (
                quad(
                    _eval_spline,
                    xbins[ib],
                    1.0,
                    args=(entry.tck, entry.x17, entry.cov),
                    **_QUAD_PARAMS,
                )[0]
                / xwidths[ib]
            )
        else:
            if xbins[ib + 1] < entry.x_min:
                integral[ib] = 0.0
                continue
            if xbins[ib] < entry.x_min:
                low_lim = entry.x_min
            else:
                low_lim = xbins[ib]

            integral[ib] = entry.tv * (
                quad(
                    _eval_spline,
                    low_lim,
                    xbins[ib + 1],
                    args=(entry.tck, entry.x17, entry.cov),
                    **_QUAD_PARAMS,
                )[0]
                / xwidths[ib]
            )

    return integral


def gen_matrix_variations(ddm_obj, mceq):
    """
    Generate variations of DDM matrices and isospin partners.

    Parameters
    ----------
    ddm_obj : DataDrivenModel
        DataDrivenModel object.
    mceq : MCEq
        MCEq object.

    Returns
    -------
    Dict[Tuple[int, int], Dict[int, List[numpy.ndarray]]]
        Dictionary mapping (projectile, secondary) pairs to a dictionary
        of variations for each knot index. Each variation consists of a
        list of two matrices (max and min) and the variation magnitude.
    Dict[Tuple[int, int], Tuple[Tuple[int, int], Dict[int, List[numpy.ndarray]]]]
        Dictionary mapping (projectile, secondary) pairs to their isospin
        partners, along with the variations of isospin matrices.

    Examples
    --------
    >>> model = DataDrivenModel(...)
    >>> mceq = MCEq(...)
    >>> matrix_variations, isospin_partners = gen_matrix_variations(model, mceq)
    """

    matrix_variations = {}
    isospin_partners = {}

    ddm_matrices = ddm_obj.ddm_matrices(mceq)
    for channel in ddm_obj.spline_db.channels:
        info(1, f"Generating variations for channel\n{channel}")
        mat_db = {}
        iso_part_db = {}

        for ispl, spl_entry in enumerate(channel.entries):
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
                    channel, mceq, e_min=ddm_obj.e_min, e_max=ddm_obj.e_max
                )
                spl_entry.tck = tck_min
                mat_min = _generate_DDM_matrix(
                    channel, mceq, e_min=ddm_obj.e_min, e_max=ddm_obj.e_max
                )
                spl_entry.tck = tck

                mat_db[ispl][iknot] = [mat_max, mat_min, h]

                if channel.secondary in [321, -321] and ddm_obj.enable_K0_from_isospin:
                    imat_max = 0.5 * (
                        mat_max + ddm_matrices[(channel.projectile, -channel.secondary)]
                    )
                    imat_min = 0.5 * (
                        mat_min + ddm_matrices[(channel.projectile, -channel.secondary)]
                    )
                    iso_part_db[ispl][iknot] = [imat_max, imat_min, h]

        matrix_variations[(channel.projectile, channel.secondary)] = mat_db
        if channel.secondary in [321, -321]:
            isospin_partners[(channel.projectile, channel.secondary)] = (
                (
                    310,
                    130,
                ),
                iso_part_db,
            )

    return matrix_variations, isospin_partners
