from copy import copy
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splev
from jacobi import propagate
import mceq_config as config
from dataclasses import dataclass
import pathlib

from .misc import info
from .particlemanager import _pdata

# isospin symmetries used in the DDM
isospin_partners = {2212: 2112, -211: 211}
isospin_symmetries = {
    2112: {
        321: 321,  # This is not exact but this is what we have
        211: -211,
        2212: 2112,
        2112: 2212,
        -2212: -2212,
        -2112: -2112,
        -211: 211,
        -321: -321,
        310: 310,
        130: 130,
    },
    211: {
        321: -321,
        211: -211,
        2212: 2112,
        2112: 2212,
        -2212: -2112,
        -2112: -2212,
        -211: 211,
        -321: 321,
        310: 310,
        130: 130,
    },
}
_LIMIT_PROPAGATE = np.inf
_PROPAGATE_PARAMS = dict(maxiter=10, maxgrad=1, method=0)  # Used to compute the
_QUAD_PARAMS = dict(limit=150, epsabs=1e-5)


def _spline_min_max_at_knot(
    tck: Tuple, iknot: int, sigma: List[float]
) -> Tuple[Tuple, Tuple, float]:
    """Return variation of spline coefficients by 1 sigma.

    Args:
        tck: A tuple representing the input spline.
        iknot: An integer representing the knot index to calculate the variation of.
        sigma: A list of floats representing the 1-sigma errors of each knot.

    Returns:
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


def _fitfunc(
    x: Union[float, np.ndarray],
    tck: Tuple,
    x17: bool,
    cov: np.ndarray,
    tv: float,
    te: float,
    return_error: Optional[bool] = False,
    gamma_zfac: Optional[float] = None,
) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    The fitting function for a spline.

    Args:
        x: The knot points.
        tck: A tuple containing the knots of the spline.
        x17: If True, fit x^1.7 dn/dx. Otherwise, fit dn/dx.
        cov: The covariance matrix of the spline.
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

    if len(tck) == 3:

        def func(tck_1: np.ndarray) -> np.ndarray:
            return factor * np.exp(splev(x, (tck[0], tck_1, tck[2])))

        func_params = tck[1]

    else:
        (tck_fit, tck_corr) = tck
        func_params = np.hstack([tck_fit[1], tck_corr[1]])

        def func(params: np.ndarray) -> np.ndarray:
            tck_fit_1 = params[: len(tck_fit[1])]
            tck_corr_1 = params[len(tck_fit[1]) :]
            return (
                factor
                * np.exp(splev(x, (tck_fit[0], tck_fit_1, tck_fit[2])))
                * splev(x, (tck_corr[0], tck_corr_1, tck_corr[2]))
            )

    if return_error:
        y, C = propagate(func, func_params, cov, **_PROPAGATE_PARAMS)
        sig_y = np.squeeze(np.sqrt(np.diag(np.atleast_1d(C))))
        return tv * y, tv * te * sig_y
    else:
        res = np.atleast_1d(tv * func(func_params))
        res[(res < 0) | ~np.isfinite(res) | (res > _LIMIT_PROPAGATE)] = 0.0
        return res.squeeze()


@dataclass
class _DDMEntry:
    """Represents information about a single DDM spline, i.e. for one
    specific beam energy and projectile, secondary combination.

    Attributes:
        ebeam (float): The beam energy in GeV.
        x17 (bool): Whether the spline fits x^1.7 dn/dx or just dn/dx.
        tck (Tuple): The knots of the spline.
        cov (np.ndarray): The covariance matrix of the spline.
        tv (float): The tuning value.
        te (float): The error scale.
        spl_idx (int): The index of the spline."""

    ebeam: float
    projectile: int
    secondary: int
    x17: bool
    tck: Tuple
    cov: np.ndarray
    tv: float
    te: float
    spl_idx: int


class DDMEntryCollection:
    """A class for storing DDM splines.

    Attributes:
        _entries (List[_DDMEntry]): A list of _DDMEntry objects.

    Methods:
        add_entry: Adds a new DDM entry.
        entries: Returns the DDM entries.
        splines: Returns the splines.
        covariances: Returns the covariance matrices.
        tuning_values: Returns the tuning values.
        error_scales: Returns the error scales.
        spline_indices: Returns the spline indices.
        spline: Returns a spline.
        covariance: Returns a covariance matrix.
        tuning_value: Returns a tuning value.
        error_scale: Returns an error scale.
        spline_index: Returns a spline index.
        spline_at_ebeam: Returns a spline at a given beam energy."""

    def __init__(self):
        self._entries = []

    def add_entry(
        self,
        ebeam: float,
        projectile: int,
        secondary: int,
        x17: bool,
        tck: Tuple,
        cov: np.ndarray,
        tv: float,
        te: float,
        spl_idx: int,
    ):
        """Adds a new DDM entry.

        Args:
            ebeam (float): The beam energy in GeV.
            x17 (bool): Whether the spline fits x^1.7 dn/dx or just dn/dx.
            tck (Tuple): The knots of the spline.
            cov (np.ndarray): The covariance matrix of the spline.
            tv (float): The tuning value.
            te (float): The error scale.
            spl_idx (int): The index of the spline."""

        self._entries.append(
            _DDMEntry(ebeam, projectile, secondary, x17, tck, cov, tv, te, spl_idx)
        )
        self._entries.sort(key=lambda entry: entry.ebeam)

    @property
    def entries(self) -> List[_DDMEntry]:
        """Returns the DDM entries.

        Returns:
            A list of _DDMEntry objects."""

        return self._entries

    @property
    def spline_indices(self) -> List[int]:
        """Returns the spline indices.

        Returns:
            A list of spline indices."""

        return [entry.spl_idx for entry in self._entries]

    # def spline(self, idx: int) -> Tuple:
    #     """Returns a spline.

    #     Args:
    #         idx (int): The index of the spline.

    #     Returns:
    #         The spline."""

    #     return self._entries[idx].tck

    def spline_index_at_ebeam(self, ebeam: float) -> int:
        """Returns a spline index at a given beam energy.

        Args:
            ebeam (float): The beam energy in GeV.

        Returns:
            The spline index."""

        return self._entries[self.spline_index_at_ebeam(ebeam)].spl_idx

    @property
    def data_combinations(self) -> List[Tuple[int, int]]:
        """Returns a list of projectile, secondary combinations.

        Returns:
            A list of projectile, secondary combinations."""

        return [(entry.projectile, entry.secondary) for entry in self._entries]

    def get_entry(
        self,
        projectile: int,
        secondary: int,
        ebeam: Optional[float] = None,
        idx: Optional[int] = None,
    ) -> _DDMEntry:
        """Returns a DDM entry.

        Args:
            idx (int): The index of the spline.

        Returns:
            The DDM entry."""

        assert (ebeam is not None) != (
            idx is not None
        ), "Define either ebeam or spl_idx"
        assert (
            projectile,
            secondary,
        ) in self.data_combinations, (
            f"({projectile},{secondary}) not in valid combinations"
        )

        return self._entries[idx]


class DataDrivenModel:
    """A class for data-driven models.

    Attributes:
        e_min: A float representing the minimum energy range where DDM cross
               sections overwrite original MCEq matrices.
        e_max: A float representing the maximum energy range where DDM cross
               sections overwrite original MCEq matrices.
        enable_K0_from_isospin: A bool representing whether to use isospin
                                relation for K0S/L.
        data_combinations: A dictionary representing the data points used to
                           build the model.
        data_energy_map: A dictionary representing the mapping between energy
                         points and data points.
        channels: A list of strings representing the channels used to build the model.
        _ddm_matrices: A numpy array representing the DDM matrices.

    Methods:
        ddm_matrices(mceq): Returns the DDM matrices.
        clone_datapoint(prim, sec, original_ebeam, new_ebeam): Clones a data point
                                                               with a new energy value.
        apply_tuning(prim, sec, ebeam=None, tv=1.0, te=1.0, spl_idx=None): Applies
            tuning parameters to a data point.
        _fitfunc(x, tck, x17, cov, tv, te, return_error=False, gamma_zfac=None): A
            private method that evaluates the model fit.

    """

    def __init__(
        self,
        filename: str = str(pathlib.Path(config.data_dir) / "DDM_1.0.npy"),
        e_min: float = 0,
        e_max: float = np.inf,
        enable_channels: List[Tuple[int, int]] = [],
        exclude_projectiles: List[int] = [],
        enable_K0_from_isospin: bool = True,
    ):
        """Initializes a DataDrivenModel object.

        Args:
            filename: A string representing the file name containing the data
                points used to build the model.
            e_min: A float representing the minimum energy range where DDM
                cross sections overwrite original MCEq matrices.
                Default is 0.
            e_max: A float representing the maximum energy range where DDM
                cross sections overwrite original MCEq matrices.
                Default is infinity.
            enable_channels: A list of strings representing the channels to
                use in building the model.
                Default is an empty list.
            exclude_projectiles: A list of integers representing the projectiles
                to exclude in building the model.
                Default is an empty list.
            enable_K0_from_isospin: A bool representing whether to use isospin
                relation for K0S/L.
                Default is True.
        """
        self.e_min = e_min
        self.e_max = e_max
        self.enable_K0_from_isospin = enable_K0_from_isospin
        self.data_combinations = self._load_file(
            filename, enable_channels, exclude_projectiles
        )
        self.data_energy_map, self.channels = {}, []
        self._sort_datasets()
        self._ddm_mat = None

    def _generate_ddm_matrices(self, mceq) -> Dict[Tuple[int, int], np.ndarray]:
        """Generates a dictionary of DDM matrices for the given MCEq object.

        Args:
            mceq: An MCEq object.

        Returns:
             A dictionary with keys as tuples (projectile PDG, secondary PDG) and
        values as numpy arrays representing the corresponding DDM matrices.
        """

        _ddm_mat = dict()
        for (prim, sec) in self.channels:
            info(2, f"Generating {prim} -> {sec} DDM matrix")
            _ddm_mat[(prim, sec)] = self._generate_DDM_matrix(
                prim=prim, sec=sec, mceq=mceq
            )
        if self.enable_K0_from_isospin:
            info(3, "Generating DDM K0 matrices from isospin symm.")
            K0SL_mat = 0.5 * (_ddm_mat[(2212, 321)] + _ddm_mat[(2212, -321)])
            _ddm_mat[(2212, 310)] = K0SL_mat
            _ddm_mat[(2212, 130)] = K0SL_mat
            if (-211, 321) in _ddm_mat and (-211, -321) in _ddm_mat:
                K0SL_mat = 0.5 * (_ddm_mat[(-211, 321)] + _ddm_mat[(-211, -321)])
                _ddm_mat[(-211, 310)] = K0SL_mat
                _ddm_mat[(-211, 130)] = K0SL_mat

        return _ddm_mat

    def clone_datapoint(
        self, prim: int, sec: int, original_ebeam: float, new_ebeam: float
    ) -> None:
        """Clone a data point to a new energy.

        Args:
            prim (int): Projectile ID.
            sec (int): Secondary ID.
            original_ebeam (float): The energy of the original data point.
            new_ebeam (float): The energy of the new data point.

        Raises:
            Exception: If the original energy is not in the available energies.

        """
        assert (prim, sec) in self.data_combinations
        available_energies = [s.ebeam for s in self.data_combinations[(prim, sec)]]
        if original_ebeam not in available_energies:
            raise Exception(f"Energy {original_ebeam} not one of {available_energies}")

        # Copy the data point to the new energy
        for dc in self.data_combinations[(prim, sec)]:
            # Do not copy tuning values to new data points
            if np.allclose(dc.ebeam, dc.original_ebeam):
                self.data_combinations[(prim, sec)].append(
                    _DDMEntry(new_ebeam, dc.x17, dc.coeffs, dc.cov, 1.0, 1.0, -1)
                )

        self._sort_datasets()

    def apply_tuning(
        self,
        prim: int,
        sec: int,
        ebeam: Optional[float] = None,
        tv: float = 1.0,
        te: float = 1.0,
        spl_idx: Optional[int] = None,
    ) -> None:
        """
        Modify the tuning values and trim errors in the data_combinations dictionary.

        Args:
            prim: Primary particle code
            sec: Secondary particle code
            ebeam: Lab frame beam energy in GeV (optional)
            tv: Scaling factor for the central value
            te: Scaling factor for the errors
            spl_idx: Index of the spline to modify (optional)

        Raises:
            AssertionError: If neither `spl_idx` nor `ebeam` is set.
        """
        assert (
            ebeam is not None or spl_idx is not None
        ), "Either spl_idx or ebeam have to be set."
        if spl_idx is None:
            spl_idx = self._unpack_coeff(prim, sec, ebeam, return_spl_idx=True)[1]

        self.data_combinations[(prim, sec)][spl_idx][4] = tv
        self.data_combinations[(prim, sec)][spl_idx][5] = te
        self._sort_datasets()

    def dn_dxl(self, x, prim, sec, ebeam, return_error=True):
        """Returns dN/dxL and error."""

        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        res = _fitfunc(x, tck, x17, cov, tv, te, return_error=return_error)
        mask = np.where(x < _pdata.mass(sec) / ebeam)
        if isinstance(res, tuple):
            v, e = res
            v[mask] = 0.0
            e[mask] = 0.0
            return v, e
        elif isinstance(res, np.ndarray):
            res[mask] = 0

        return res

    def calc_zfactor_and_error(self, prim, sec, ebeam, gamma=1.7):
        """The parameter `gamma` is the CR nucleon integral spectral index."""

        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        info(3, f"Calculating Z-factor for {prim}-->{sec} @ {ebeam} GeV.")

        def func_int(tck_1):
            res = quad(
                _fitfunc,
                _pdata.mass(sec) / ebeam,
                1.0,
                args=((tck[0], tck_1, tck[2]), x17, cov, tv, te, False, gamma),
                **_QUAD_PARAMS,
            )[0]
            res = np.atleast_1d(res)
            res[(res < 0) | ~np.isfinite(res) | (res > _LIMIT_PROPAGATE)] = 0.0
            return res.squeeze()

        y, C = propagate(func_int, tck[1], cov, **_PROPAGATE_PARAMS)
        sig_y = np.sqrt(C)

        return y, sig_y

    def calc_zfactor_and_error2(self, prim, sec, ebeam, fv, fe, gamma=1.7):
        """The parameter `gamma` is the CR nucleon integral spectral index.

        fv and fe are the fitted correction and its uncertainty.
        """

        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        info(3, f"Calculating Z-factor for {prim}-->{sec} @ {ebeam} GeV.")

        def fitfunc_center(*args, **kwargs):
            return _fitfunc(*args, **kwargs)[0]

        def fitfunc_error(*args, **kwargs):
            return _fitfunc(*args, **kwargs)[1]

        zfactor_center = quad(
            fitfunc_center,
            _pdata.mass(sec) / ebeam,
            1.0,
            args=(tck, x17, cov, 1.0, 1.0, True, gamma),
            **_QUAD_PARAMS,
        )[0]
        zfactor_error = quad(
            fitfunc_error,
            _pdata.mass(sec) / ebeam,
            1.0,
            args=(tck, x17, cov, 1.0, 1.0, True, gamma),
            **_QUAD_PARAMS,
        )[0]

        return zfactor_center + fv * zfactor_error, fe * zfactor_error

    def _gen_dndx(self, xbins, prim, sec, ebeam):
        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        x = np.sqrt(xbins[1:] * xbins[:-1])

        res = _fitfunc(x, tck, x17, cov, tv, te, return_error=False)
        res[x < _pdata.mass(sec) / ebeam] = 0
        return res

    def _gen_averaged_dndx(self, xbins, prim, sec, ebeam):

        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        xwidths = np.diff(xbins)
        integral = np.zeros_like(xwidths)

        for ib in range(len(integral)):
            if xbins[ib + 1] > 1:
                integral[ib] = (
                    quad(
                        _fitfunc,
                        xbins[ib],
                        1.0,
                        args=(tck, x17, cov, tv, te),
                        **_QUAD_PARAMS,
                    )[0]
                    / xwidths[ib]
                )
            else:
                if xbins[ib + 1] < _pdata.mass(sec) / ebeam:
                    integral[ib] = 0.0
                    continue
                elif xbins[ib] < _pdata.mass(sec) / ebeam:
                    low_lim = _pdata.mass(sec) / ebeam
                else:
                    low_lim = xbins[ib]

                integral[ib] = (
                    quad(
                        _fitfunc,
                        low_lim,
                        xbins[ib + 1],
                        args=(tck, x17, cov, tv, te),
                        **_QUAD_PARAMS,
                    )[0]
                    / xwidths[ib]
                )

        return integral

    def _generate_DDM_matrix(
        self,
        prim,
        sec,
        mceq,
        average=True,
    ):
        # Definitions of particles and energy grid
        dim = mceq.dim
        prim_mass = mceq.pman[prim].mass
        sec_mass = mceq.pman[sec].mass
        # Convert from kinetic to lab energy
        elab_sec_bins = mceq._energy_grid.b + sec_mass
        elab_sec_centers = mceq._energy_grid.c + sec_mass
        elab_proj_centers = mceq._energy_grid.c + prim_mass
        e_widths = np.diff(elab_sec_bins)
        xgrid = elab_sec_bins / elab_sec_centers[-1]

        # Pull a copy of the original matrix
        try:
            mat = copy(mceq.pman[prim].hadr_yields[mceq.pman[sec]].get())
        except AttributeError:
            mat = copy(mceq.pman[prim].hadr_yields[mceq.pman[sec]])

        if average:
            dndx_generator = self._gen_averaged_dndx
        else:
            dndx_generator = self._gen_dndx

        info(
            5,
            f"DDM will be used between {self.e_min:.1e} GeV "
            + f"and {self.e_max:.1e} GeV",
        )
        n_datasets = len(self.data_combinations[(prim, sec)])
        assert n_datasets > 0, "Entry data_combinations can't be empty"

        if n_datasets == 1:
            ebeam = self._unpack_coeff(prim, sec, spl_idx=0)[0]
            ie0 = np.argmin(np.abs(ebeam - elab_proj_centers))
            mceq_ebeam = elab_proj_centers[ie0]
            info(
                5,
                f"Dataset 0 ({prim}, {sec}): mceq_eidx={ie0}, "
                + f"mceq_ebeam={mceq_ebeam:4.3f}, ebeam={ebeam}",
            )
            averaged_dndx = dndx_generator(xgrid, prim, sec, ebeam)

            for ie, eproj in enumerate(elab_proj_centers[:-1]):

                if (eproj < config.e_min) or (eproj > config.e_max):
                    info(0, "Skipping out of range energy", eproj)
                    continue

                mat[: ie + 1, ie] = (
                    averaged_dndx[dim - ie - 1 :] / elab_proj_centers[ie] * e_widths[ie]
                )
        else:
            # Loop through datasets in pairs of two and interpolate
            # between them
            for interval in range(n_datasets - 1):
                ebeam_0 = self._unpack_coeff(prim, sec, spl_idx=interval)[0]
                ebeam_1 = self._unpack_coeff(prim, sec, spl_idx=interval + 1)[0]

                ie0 = np.argmin(np.abs(ebeam_0 - elab_proj_centers))
                ie1 = np.argmin(np.abs(ebeam_1 - elab_proj_centers))
                mceq_ebeam_0 = elab_proj_centers[ie0]
                mceq_ebeam_1 = elab_proj_centers[ie1]
                info(
                    5,
                    f"Dataset {interval} ({prim}, {sec}): mceq_eidx={ie0}, "
                    + f"mceq_ebeam={mceq_ebeam_0:4.3f}, ebeam={ebeam_0}",
                )
                info(
                    5,
                    f"Dataset {interval + 1} ({prim}, {sec}): mceq_eidx={ie1}, "
                    + f"mceq_ebeam={mceq_ebeam_1:4.3f}, ebeam={ebeam_1}",
                )

                try:
                    averaged_dndx_0 = dndx_generator(xgrid, prim, sec, ebeam_0)
                except ValueError:
                    raise Exception(
                        "Error averaging cross section for "
                        + f"({prim},{sec},{ebeam_0})"
                    )
                try:
                    averaged_dndx_1 = dndx_generator(xgrid, prim, sec, ebeam_1)
                except ValueError:
                    raise Exception(
                        "Error averaging cross section for "
                        + f"({prim},{sec},{ebeam_1})"
                    )

                for ie, eproj in enumerate(elab_proj_centers):
                    if (eproj < config.e_min) or (eproj > config.e_max):
                        continue
                    if ie <= ie0 and interval == 0:
                        # If it is the first interval, extrapolate to
                        # from the lowest energies
                        mat[: ie + 1, ie] = (
                            averaged_dndx_0[dim - ie - 1 :]
                            / elab_proj_centers[ie]
                            * e_widths[ie]
                        )
                    elif ie < ie0 and interval > 0:
                        # if it is not the first interval, don't touch the
                        # columns that have been previously filled
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
                        m = (f1 - f0) / np.log(ebeam_1 / ebeam_0)
                        mat[: ie + 1, ie] = f0 + m * np.log(eproj / ebeam_0)
                    elif ie > ie1 and interval != n_datasets - 2:
                        # if it is not the first dataset, don't touch the
                        # columns that have been previously filled
                        break
                    else:
                        mat[: ie + 1, ie] = (
                            averaged_dndx_1[dim - ie - 1 :]
                            / elab_proj_centers[ie]
                            * e_widths[ie]
                        )

        return mat

    def _load_file(
        self,
        filename: str,
        enable_channels: List[Tuple[int, int]] = [],
        exclude_projectiles: List[int] = [],
    ) -> Dict[Tuple[int, int], List[_DDMEntry]]:
        """
        Load the file containing the precomputed splines for all (primary, secondary)
        particle pairs, and energy points, and convert it into a dictionary of lists
        where each list corresponds to a unique (primary, secondary) pair and contains
        the available energy points and corresponding spline coefficients and
        covariances for each energy point.

        Args:
            filename: A string representing the filename to be loaded.
            enable_channels: A list of tuples containing particle pair
                            (primary, secondary) representing the desired channels
                            to be included. If empty, all channels will be included.
            exclude_projectiles: A list of integers representing the projectile
                                 particle types to be excluded.

        Returns:
            A dictionary of lists, where each dictionary key is a tuple of two
            integers representing the (primary, secondary) particle pair, and the
            corresponding value is a list of energy points and their corresponding
            spline coefficients and covariances. Each entry of the list has the
            format [ebeam, x17, coeffs, cov, tv, te], where ebeam is the lab frame
            beam energy in GeV, x17 is a boolean indicating if the spline fit
            x^1.7 dn/dx or just dn/dx, coeffs is the list of spline coefficients,
            cov is the corresponding covariance matrix, and tv and te
            are the scale of the central value and the scale of errors, respectively.
        """
        splines = np.load(filename, allow_pickle=True, encoding="latin1").item()
        data_combinations = {}
        for (prim, sec, ebeam, x17) in splines:
            if (enable_channels and (prim, sec) not in enable_channels) or (
                abs(prim) in exclude_projectiles
            ):
                continue
            if (prim, sec) not in data_combinations:
                data_combinations[(prim, sec)] = []

            data_combinations[(prim, sec)].append(
                _DDMEntry(
                    ebeam,
                    x17,
                    splines[(prim, sec, ebeam, x17)][0],
                    splines[(prim, sec, ebeam, x17)][1],
                    1.0,
                    1.0,
                    -1,
                )
            )
        return data_combinations

    def _load_knot_sigmas(self):
        """
        Calculate the error for each knot in the spline.

        Returns:
            dict: The error at each knot for each spline.
        """
        _knot_sigmas = {}
        for prim, sec in self.data_combinations:
            _knot_sigmas[(prim, sec)] = []
            for dc in self.data_combinations[(prim, sec)]:
                info(5, f"Calculating knot error for {prim},{sec},{dc.ebeam}.")
                _knot_sigmas[(prim, sec)].append(np.sqrt(np.diag(dc.cov)))
        return _knot_sigmas

    # def _init_scenario(self, mceq):

    def _dim_spl(
        self, prim: int, sec: int, ret_detailed: bool = False
    ) -> Union[int, Tuple[List, List]]:
        """Return the number of splines or the total number of knots.

        Args:
        prim (int): projectile ID
        sec (int): secondary particle ID
        ret_detailed (bool): whether to return a detailed tuple with
            the splines and their knot counts (default False)

        Returns:
        int: if `ret_detailed` is False, returns the number of total knots
        tuple: if `ret_detailed` is True, returns a tuple with two lists,
            the first with the splines, and the second with the count
            of knots for each spline
        """
        assert (
            prim,
            sec,
        ) in self.data_combinations, f"Unknown ({prim},{sec}) combination supplied"
        channel_cross_sections = self.data_combinations[(prim, sec)]

        # Count number of knots in each spline
        tcks = [li[2] for li in channel_cross_sections]
        dim_spls = [len(tck[0]) for tck in tcks]
        ntot = sum(dim_spls)

        if ret_detailed:
            return tcks, dim_spls
        else:
            return ntot

    def gen_matrix_variations(self, mceq):
        matrix_variations = {}
        isospin_partners = {}
        sigmas = self._load_knot_sigmas()
        # generate a default set of matrices and save it for isospin cooking
        if self._ddm_matrices is None:
            _ = self.ddm_matrices(mceq)
        for (prim, sec) in self.data_combinations:
            info(1, f"Generating vatiations for {prim} -> {sec}")
            tcks, dim_spls = self._dim_spl(prim, sec, ret_detailed=True)
            sigs = sigmas[(prim, sec)]
            mat_db = []
            iso_part_db = []

            # Loop through the knots of each spline
            for ispl, n in enumerate(dim_spls):
                tck, sig = tcks[ispl], sigs[ispl]
                mat_db.append({})
                iso_part_db.append({})
                for iknot in range(0, n):
                    if np.allclose(tck[1][iknot], 0.0):
                        continue
                    tck_min, tck_max, h = _spline_min_max_at_knot(tck, iknot, sig)
                    # Replace coefficients by the varied ones in data_combinations
                    self.data_combinations[(prim, sec)][ispl][2] = tck_max
                    mat_max = self._generate_DDM_matrix(prim, sec, mceq)
                    self.data_combinations[(prim, sec)][ispl][2] = tck_min
                    mat_min = self._generate_DDM_matrix(prim, sec, mceq)
                    # Restore original tck
                    self.data_combinations[(prim, sec)][ispl][2] = tck
                    mat_db[-1][iknot] = [mat_max, mat_min, h]
                    # Create also the varied isospin matrices
                    if sec in [321, -321] and self.enable_K0_from_isospin:
                        imat_max = 0.5 * (mat_max + self._ddm_matrices[(prim, -sec)])
                        imat_min = 0.5 * (mat_min + self._ddm_matrices[(prim, -sec)])
                        iso_part_db[-1][iknot] = [imat_max, imat_min, h]
            matrix_variations[(prim, sec)] = mat_db
            if sec in [321, -321]:
                isospin_partners[(prim, sec)] = (310, 130), iso_part_db

        return matrix_variations, isospin_partners

    # def _unpack_coeff(
    #     self,
    #     prim: int,
    #     sec: int,
    #     ebeam: Optional[float] = None,
    #     spl_idx: Optional[int] = None,
    # ) ->
    # ]:
    #     """Extracts the spline coefficients, errors, and other information for a given
    #     combination of primary and secondary particle and lab energy.
    #     If `return_spl_idx` is set to True, the function also returns the spline index
    #     used to extract the information.

    #     Args:
    #         prim (int): The primary particle type.
    #         sec (int): The secondary particle type.
    #         ebeam (float, optional): The lab energy of the data point. Either `ebeam` or
    #             `spl_idx` should be defined.
    #         spl_idx (int, optional): The index of the spline in the `data_combinations`
    #             dictionary. Either `ebeam` or `spl_idx` should be defined.
    #         return_spl_idx (bool, optional): Whether to return the spline index.

    #     Returns:
    #         tuple: A tuple with the following elements:
    #             * ebeam (float): The lab energy of the data point.
    #             * x17 (bool): Whether the spline was fit to x^1.7 * dn/dx.
    #             * tck (Tuple): The spline coefficients.
    #             * cov (np.ndarray): The covariance matrix for the spline.
    #             * tv (float): The tuning value for the spline.
    #             * te (float): The trimmed error for the spline.
    #         If `return_spl_idx` is True, the tuple also includes the spline index.
    #     """
    #     assert (ebeam is not None) != (
    #         spl_idx is not None
    #     ), "Define either ebeam or spl_idx"
    #     assert (
    #         prim,
    #         sec,
    #     ) in self.data_combinations, f"({prim},{sec}) not in valid combinations"
    #     try:
    #         if ebeam:
    #             spl_idx = self.data_energy_map[(prim, sec, ebeam)]
    #         else:
    #             if spl_idx >= len(self.data_combinations[(prim, sec)]):
    #                 raise KeyError
    #     except KeyError:
    #         raise Exception(
    #             f"Combination ({prim}, {sec}, {ebeam}) not found in\n"
    #             + "\n".join(
    #                 [
    #                     "{0}, {1}, {2}".format(*a)
    #                     for a in sorted(self.data_energy_map.keys())
    #                 ]
    #             )
    #         )
    #     ebeam, x17, tck, cov, tv, te = self.data_combinations[(prim, sec)][spl_idx]
    #     assert len(tck) in [2, 3], "Unknown number of spline coeffs"
    #     if return_spl_idx:
    #         return (ebeam, x17, tck, cov, tv, te), spl_idx
    #     return (ebeam, x17, tck, cov, tv, te)

    def _sort_datasets(self) -> None:
        """
        Sorts the datasets by beam energy, and creates a mapping of data point indices
        to their corresponding beam energies. Also resets the DDM matrices to ensure
        that MCEq uses the most recent ones.
        """
        # Sort datasets by beam energy
        self.data_energy_map = {}

        for (prim, sec) in self.data_combinations:
            self.data_combinations[(prim, sec)] = sorted(
                self.data_combinations[(prim, sec)], key=lambda x: x.ebeam
            )
            for iset, dc in enumerate(self.data_combinations[(prim, sec)]):
                self.data_energy_map[(prim, sec, dc.ebeam)] = iset
                dc.spl_idx = iset
        self.channels = sorted(list(self.data_combinations.keys()))
        # Reset the matrices to make sure MCEq takes the most recent ones
        self._ddm_matrices = None

    def __repr__(self) -> str:
        s = "DDM channels:\n"
        for prim, sec in self.channels:
            s += f"\t{prim} -> {sec}:\n"
            for iset, dc in enumerate(self.data_combinations[(prim, sec)]):
                s += f"\t\t{iset}: ebeam = {dc.ebeam} GeV, x17={dc.x17}, "
                s += f"tune v|e={dc.tv:4.3f}|{dc.te:4.3f}\n"
        s += "\n"
        return s
