from copy import copy
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from jacobi import propagate
from scipy.integrate import quad
from scipy.interpolate import splev

import mceq_config as config

from .misc import info

_LIMIT_PROPAGATE = np.inf
_PROPAGATE_PARAMS = dict(maxiter=10, maxgrad=1, method=0)  # Used to compute the
_QUAD_PARAMS = dict(limit=150, epsabs=1e-5)


def fmteb(ebeam: Union[float, str, int]) -> str:
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
    """
    Return variation of spline coefficients by 1 sigma.

    Args:
        tck : Tuple
            A tuple representing the input spline.
        iknot : int
            The knot index to calculate the variation of.
        sigma : npt.NDArray
            The 1-sigma errors of each knot.

    Returns
    -------
    Tuple[Tuple, Tuple, float]
        A tuple containing two tuples representing the minimum and maximum spline
        coefficients, respectively, and a float representing the variation of the
        spline coefficients by 1 sigma.
    """
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
    """
    Generate a Data-Driven Model matrix for the given channel and MCEq object.

    Args:
        channel : _DDMChannel
            The DDM channel.
        mceq : MCEq
            The MCEq object.
        e_min : float, optional
            The minimum energy range in GeV where DDM cross sections overwrite original MCEq matrices,
            by default -1.0.
        e_max : float, optional
            The maximum energy range in GeV where DDM cross sections overwrite original MCEq matrices,
            by default -1.0.
        average : bool, optional
            Whether to average the dndx values, by default True.

    Returns:
        npt.NDArray
            The DDM matrix.
    """
    from MCEq.misc import energy_grid, _eval_energy_cuts

    projectile = channel.projectile
    secondary = channel.secondary
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
        mceq_ebeam = elab_proj_centers[ie0]
        info(
            5,
            f"Dataset 0 ({projectile}, {secondary}): mceq_eidx={ie0}, mceq_ebeam={mceq_ebeam:4.3f}, ebeam={entry.ebeam}",
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
                        averaged_dndx_0[dim - ie - 1 :]
                        / elab_proj_centers[ie]
                        * e_widths[ie]
                    )
                elif ie < ie0 and interval > 0:
                    # if it is not the first interval, don't touch the columns
                    # that have been previously filled
                    continue
                elif ie <= ie1:
                    # Interpolate between two datasets
                    f0 = (
                        averaged_dndx_0[dim - ie - 1 :]
                        / elab_proj_centers[ie]
                        * e_widths[ie]
                    )
                    f1 = (
                        averaged_dndx_1[dim - ie - 1 :]
                        / elab_proj_centers[ie]
                        * e_widths[ie]
                    )
                    m = (f1 - f0) / np.log(entry_1.fl_ebeam / entry_0.fl_ebeam)
                    mat[: ie + 1, ie] = f0 + m * np.log(eproj / entry_0.fl_ebeam)
                elif ie > ie1 and interval != n_datasets - 2:
                    # if it is not the first dataset, don't touch the columns
                    # that have been previously filled
                    break
                else:
                    mat[: ie + 1, ie] = (
                        averaged_dndx_1[dim - ie - 1 :]
                        / elab_proj_centers[ie]
                        * e_widths[ie]
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
        The knot points.
    tck : Tuple[numpy.ndarray, numpy.ndarray, int]
        A tuple containing the knots of the spline.
    x17 : bool
        If True, scale by x^1.7 dn/dx. Otherwise, return only dn/dx.
    cov : numpy.ndarray
        The covariance matrix of the spline.
    return_error : bool, optional
        If True, return the propagated error of the fit. Default is False.
    gamma_zfac : float, optional
        The exponent for the gamma function. If None, it is not used. Default is None.

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

    def func(tck_1: npt.NDArray) -> npt.NDArray:  # type: ignore
        return factor * np.exp(splev(x, (tck[0], tck_1, tck[2])))

    func_params = tck[1]

    if return_error:
        y, C = propagate(func, func_params, cov, **_PROPAGATE_PARAMS)  # type: ignore
        sig_y = np.squeeze(np.sqrt(np.diag(np.atleast_1d(C))))
        return y, sig_y
    else:
        res = np.atleast_1d(func(func_params))
        res[(res < 0) | ~np.isfinite(res) | (res > _LIMIT_PROPAGATE)] = 0.0
        return res.squeeze()


def _eval_spline_and_correction(
    x: Union[float, npt.NDArray],
    tck_dndx: Tuple[npt.NDArray, npt.NDArray, int],
    tck_corr: Tuple[npt.NDArray, npt.NDArray, int],
    x17: bool,
    cov: npt.NDArray,
    return_error: Optional[bool] = False,
    gamma_zfac: Optional[float] = None,
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    """
    This case evaluates the product of two splines, which has been
    used in DDM to apply the MC correction from pp -> K+ to pC -> K+

    Args:
        x: The knot points.
        tck_dndx: A tuple containing the knots of the spline.
        tck_corr: Knots of the spline that is multiplied with the dndx spline.
        x17: If True, fit x^1.7 dn/dx. Otherwise, fit dn/dx.
        cov: The covariance matrix .
        tv: The tuning value.
        te: The error scale.
        return_error: If True, return the propagated error of the fit.
        gamma_zfac: The exponent for the gamma function. If None, it is not used.

    Returns:
        The value of the fit at each knot point.
    """
    if gamma_zfac is None:
        factor = x ** (-1.7) if x17 else 1.0
    else:
        factor = x ** (gamma_zfac) if not x17 else x ** (gamma_zfac - 1.7)

    func_params = np.hstack([tck_dndx[1], tck_corr[1]])

    def func(params: npt.NDArray) -> npt.NDArray:
        tck_dndx_1 = params[: len(tck_dndx[1])]
        tck_corr_1 = params[len(tck_dndx[1]) :]
        return (
            factor
            * np.exp(splev(x, (tck_dndx[0], tck_dndx_1, tck_dndx[2])))
            * splev(x, (tck_corr[0], tck_corr_1, tck_corr[2]))
        )

    if return_error:
        y, C = propagate(func, func_params, cov, **_PROPAGATE_PARAMS)  # type: ignore
        sig_y = np.squeeze(np.sqrt(np.diag(np.atleast_1d(C))))
        return y, sig_y
    else:
        res = np.atleast_1d(func(func_params))
        res[(res < 0) | ~np.isfinite(res) | (res > _LIMIT_PROPAGATE)] = 0.0
        return res.squeeze()


def _gen_dndx(xbins: np.ndarray, entry) -> np.ndarray:
    """
    Generate dN/dx values for a given energy binning using a DDM entry.

    Parameters
    ----------
    xbins : numpy.ndarray
        Binning of x values.
    entry : _DDMEntry
        The DDM entry containing the spline information.

    Returns
    -------
    numpy.ndarray
        Array of dN/dx values corresponding to the provided xbinning.

    Notes
    -----
    The dN/dx values are calculated by evaluating the spline defined by the
    DDM entry at the square root of the bin boundaries. The resulting values
    are multiplied by the tuning value of the entry.

    If the evaluated x value is less than the minimum x value of the spline,
    the corresponding dN/dx value is set to zero.

    The returned array has the same shape as the provided xbinning.

    Examples
    --------
    >>> xbins = np.linspace(0, 1, 11)
    >>> entry = _DDMEntry(...)
    >>> dndx = _gen_dndx(xbins, entry)
    """

    x = np.sqrt(xbins[1:] * xbins[:-1])

    res = _eval_spline(x, entry.tck, entry.x17, entry.cov, return_error=False)
    res[x < entry.x_min] = 0  # type: ignore
    return entry.tv * res


def _gen_averaged_dndx(xbins: np.ndarray, entry) -> np.ndarray:
    """
    Generate averaged dN/dx values for a given energy binning using a DDM entry.

    Parameters
    ----------
    xbins : numpy.ndarray
        Binning of x values.
    entry : _DDMEntry
        The DDM entry containing the spline information.

    Returns
    -------
    numpy.ndarray
        Array of averaged dN/dx values corresponding to the provided xbinning.

    Notes
    -----
    The averaged dN/dx values are calculated by integrating the spline defined
    by the DDM entry over each energy bin and dividing by the bin width. The
    resulting values are multiplied by the tuning value of the entry.

    If an energy bin extends beyond x=1, the integration is performed from the
    lower limit of the bin to x=1.

    If an energy bin is entirely below entry.x_min, the averaged dN/dx value
    for that bin is set to zero.

    The returned array has the same shape as the provided xbinning[:-1].

    Examples
    --------
    >>> xbins = np.linspace(0.1, 1, 11)
    >>> entry = _DDMEntry(...)
    >>> averaged_dndx = _gen_averaged_dndx(xbins, entry)
    """

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
            elif xbins[ib] < entry.x_min:
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


def calc_zfactor_and_error(
    ddm, projectile: int, secondary: int, ebeam: Union[str, float], gamma: float = 1.7
):
    """
    Calculate the Z-factor and its error for a given projectile,
    secondary, and beam energy.

    Parameters
    ----------
    ddm: DataDrivenModel
        Instance of DataDrivenModel.
    projectile : int
        Projectile PDG code.
    secondary : int
        Secondary PDG code.
    ebeam : float
        Beam energy in GeV.
    gamma : float, optional
        CR nucleon integral spectral index. Default is 1.7.

    Returns
    -------
    numpy.ndarray
        Z-factor values.
    numpy.ndarray
        Error of the Z-factor values.

    Notes
    -----
    The Z-factor is calculated by integrating the spline defined by the DDM entry
    over the range from entry.x_min to 1.0. The integral is multiplied by the tuning
    value of the entry and raised to the power of gamma.

    The error of the Z-factor is propagated using the covariance matrix of the spline
    and the provided _PROPAGATE_PARAMS.

    Examples
    --------
    >>> model = DataDrivenModel(...)
    >>> projectile = 2212
    >>> secondary = 211
    >>> ebeam = 10.0
    >>> z_factor, z_error = model.calc_zfactor_and_error(projectile, secondary, ebeam)
    """

    entry = ddm.spline_db.get_entry(projectile, secondary, ebeam)

    info(3, f"Calculating Z-factor for {projectile} --> {secondary} @ {ebeam} GeV.")

    def func_int(tck_1):
        res = quad(
            _eval_spline,
            entry.x_min,
            1.0,
            args=(
                (entry.tck[0], tck_1, entry.tck[2]),
                entry.x17,
                entry.cov,
                False,
                gamma,
            ),
            **_QUAD_PARAMS,
        )[0]
        res = np.atleast_1d(res)
        res[(res < 0) | ~np.isfinite(res) | (res > _LIMIT_PROPAGATE)] = 0.0
        return res.squeeze()

    y, C = propagate(func_int, entry.tck[1], entry.cov, **_PROPAGATE_PARAMS)
    sig_y = np.sqrt(C)

    return y, sig_y


def calc_zfactor_and_error2(
    ddm, projectile: int, secondary: int, ebeam: Union[str, float], gamma: float = 1.7
):
    """
    Alternative Z-factor and error calculation for a given projectile,
    secondary, and beam energy.

    Parameters
    ----------
    ddm: DataDrivenModel
        Instance of DataDrivenModel.
    projectile : int
        Projectile PDG code.
    secondary : int
        Secondary PDG code.
    ebeam : float
        Beam energy in GeV.
    gamma : float, optional
        CR nucleon integral spectral index. Default is 1.7.

    Returns
    -------
    numpy.ndarray
        Z-factor values.
    numpy.ndarray
        Error of the Z-factor values.

    Notes
    -----
    The main difference to the previous function is the error calculation, which
    is performed by integrating the error band of the evaluated splines and not
    by propagating the errors of the integral. This method is a bit more stable
    for the "problematic" splines.

    Examples
    --------
    >>> model = DataDrivenModel(...)
    >>> projectile = 2212
    >>> secondary = 211
    >>> ebeam = 158.0
    >>> z_factor, z_error = model.calc_zfactor_and_error2(projectile, secondary, ebeam)
    """

    entry = ddm.spline_db.get_entry(projectile, secondary, ebeam)

    info(3, f"Calculating Z-factor for {projectile}-->{secondary} @ {ebeam} GeV.")

    def fitfunc_center(*args, **kwargs):
        return _eval_spline(*args, **kwargs)[0]

    def fitfunc_error(*args, **kwargs):
        return _eval_spline(*args, **kwargs)[1]

    zfactor_center = quad(
        fitfunc_center,
        entry.x_min,
        1.0,
        args=(entry.tck, entry.x17, entry.cov, True, gamma),
        **_QUAD_PARAMS,
    )[0]
    zfactor_error = quad(
        fitfunc_error,
        entry.x_min,
        1.0,
        args=(entry.tck, entry.x17, entry.cov, True, gamma),
        **_QUAD_PARAMS,
    )[0]

    return (
        entry.tv * zfactor_center + entry.tv * entry.te * zfactor_error,
        entry.te * entry.tv * zfactor_error,
    )


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

            for iknot in range(0, spl_entry.n_knots):
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
                310,
                130,
            ), iso_part_db

    return matrix_variations, isospin_partners
