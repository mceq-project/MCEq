from copy import copy
from typing import Dict, List, Tuple, Union, Optional, Generator
import numpy as np
import numpy.typing as npt
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


def fmteb(ebeam: Union[float, str, int]) -> str:
    """Format beam energy for handling in db."""
    return f"{float(ebeam):.1f}"


def _spline_min_max_at_knot(
    tck: Tuple, iknot: int, sigma: npt.NDArray
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


@dataclass
class _DDMEntry:
    """
    Represents information about a single DDM spline, i.e. for one
    specific beam energy and projectile, secondary combination.

    Attributes
    ----------
    ebeam : float
        The beam energy in GeV.
    projectile : int
        The PDG code of the projectile.
    secondary : int
        The PDG code of the secondary.
    x17 : bool
        Whether the spline fits x^1.7 dn/dx or just dn/dx.
    tck : Tuple
        The knots of the spline.
    cov : npt.NDArray
        The covariance matrix of the spline.
    tv : float
        The tuning value.
    te : float
        The error scale.
    spl_idx : int
        The index of the spline.
    """

    ebeam: str
    projectile: int
    secondary: int
    x17: bool
    tck: Tuple
    cov: npt.NDArray[np.float64]
    tv: float
    te: float
    spl_idx: int

    @property
    def knot_sigma(self) -> npt.NDArray:
        """
        Compute the knot standard deviation for the spline.

        Returns
        -------
        npt.NDArray
            The standard deviation of the knots.
        """
        return self.te * np.sqrt(np.diag(self.cov))

    @property
    def fl_ebeam(self) -> float:
        """
        Get the beam energy as a float.

        Returns
        -------
        float
            The beam energy.
        """
        return float(self.ebeam)

    @property
    def n_knots(self) -> int:
        """
        Get the number of knots in the spline.

        Returns
        -------
        int
            The number of knots.
        """
        return len(self.tck[1])

    @property
    def x_min(self) -> float:
        """
        Calculate the minimum x value for the spline.

        Returns
        -------
        float
            The minimum x value.
        """
        return _pdata.mass(self.secondary) / float(self.ebeam)


class _DDMChannel:
    """
    A class for storing DDM splines for a specific projectile, secondary.

    Attributes
    ----------
    _entries : List[_DDMEntry]
        A list of _DDMEntry objects.
    """

    def __init__(
        self,
        projectile: int,
        secondary: int,
    ):
        self._entries = []
        self.projectile = projectile
        self.secondary = secondary

    def add_entry(
        self,
        ebeam: Union[float, str],
        projectile: int,
        secondary: int,
        x17: bool,
        tck: Tuple,
        cov: npt.NDArray,
        tv: float = 1.0,
        te: float = 1.0,
    ):
        """
        Adds a new DDM entry.

        Parameters
        ----------
        ebeam : float
            The beam energy in GeV.
        projectile : int
            The PDG code of the projectile.
        secondary : int
            The PDG code of the secondary.
        x17 : bool
            Whether the spline fits x^1.7 dn/dx or just dn/dx.
        tck : Tuple
            The knots of the spline.
        cov : npt.NDArray
            The covariance matrix of the spline.
        tv : float, optional
            The tuning value, by default 1.0.
        te : float, optional
            The error scale, by default 1.0.
        """

        assert projectile == self.projectile
        assert secondary == self.secondary

        for entry in self._entries:
            if fmteb(entry.ebeam) == fmteb(ebeam):
                raise ValueError(f"Duplicate entry for ebeam = {ebeam} GeV.")
        info(10, f"Adding DDM spline for {projectile} -> {secondary} at {ebeam} GeV")
        self._entries.append(
            _DDMEntry(
                fmteb(ebeam),
                projectile,
                secondary,
                x17,
                tck,
                cov,
                tv,
                te,
                len(self._entries),
            )
        )

        self._entries.sort(key=lambda entry: float(entry.ebeam))

        # Reassign the spline indices
        for i, entry in enumerate(self._entries):
            entry.spl_idx = i

    @property
    def entries(self) -> List[_DDMEntry]:
        """
        Returns the DDM entries.

        Returns
        -------
        List[_DDMEntry]
            A list of _DDMEntry objects.
        """

        return self._entries

    @property
    def total_n_knots(self) -> int:
        """
        Returns the total number of knots.

        Returns
        -------
        int
            The total number of knots.
        """

        return sum([len(entry.tck[0]) for entry in self._entries])

    @property
    def n_splines(self) -> int:
        """
        Returns the number of splines.

        Returns
        -------
        int
            The number of splines.
        """

        return len(self._entries)

    @property
    def spline_indices(self) -> List[int]:
        """
        Returns the spline indices.

        Returns
        -------
        List[int]
            A list of spline indices.
        """

        return [entry.spl_idx for entry in self._entries]

    def get_entry(
        self,
        ebeam: Optional[Union[float, int, str]] = None,
        idx: Optional[int] = None,
    ) -> _DDMEntry:
        """
        Returns a DDM entry.

        Parameters
        ----------
        ebeam : Optional[Union[float, int, str]], optional
            The beam energy in GeV, by default None.
        idx : Optional[int], optional
            The index of the spline, by default None.

        Returns
        -------
        _DDMEntry
            The DDM entry.

        Raises
        ------
        ValueError
            If no entry is found for the given ebeam or spline index.
        """

        assert (ebeam is not None) != (
            idx is not None
        ), "Define either ebeam or spl_idx"

        if ebeam is not None:
            ebeam = fmteb(ebeam)
            for entry in self._entries:
                if entry.ebeam == ebeam:
                    return entry
            raise ValueError(f"No entry for ebeam = {ebeam} GeV.")
        else:
            for entry in self._entries:
                if entry.spl_idx == idx:
                    return entry
            raise ValueError(f"No entry for spl_idx = {idx}.")

    def __str__(self) -> str:
        s = f"\t{self.projectile} -> {self.secondary}:\n"
        for iset, dc in enumerate(self.entries):
            s += f"\t\t{iset}: ebeam = {dc.ebeam} GeV, x17={dc.x17}, "
            s += f"tune v|e={dc.tv:4.3f}|{dc.te:4.3f}\n"
        return s


class DDMSplineDB:
    """A class for maintaing DDMEntryCollections for different
    projectile and secondary particle combinations."""

    _ddm_splines: Dict[str, _DDMChannel] = {}

    def __init__(
        self,
        filename: str = str(pathlib.Path(config.data_dir) / "DDM_1.0.npy"),
        enable_channels: List[Tuple[int, int]] = [],
        exclude_projectiles: List[int] = [],
    ):
        self._load_from_file(filename, enable_channels, exclude_projectiles)

    def _load_from_file(
        self,
        filename: str,
        enable_channels: List[Tuple[int, int]],
        exclude_projectiles: List[int],
    ):
        """Loads DDM splines from a file.

        Args:
            filename (str): The filename."""

        spl_file = np.load(filename, allow_pickle=True, encoding="latin1").item()
        for (projectile, secondary, ebeam, x17) in spl_file:
            if (enable_channels and (projectile, secondary) not in enable_channels) or (
                abs(projectile) in exclude_projectiles
            ):
                continue
            else:
                self.add_entry(
                    ebeam,
                    projectile,
                    secondary,
                    x17,
                    spl_file[(projectile, secondary, ebeam, x17)][0],
                    spl_file[(projectile, secondary, ebeam, x17)][1],
                    1.0,
                    1.0,
                )

    def add_entry(
        self,
        ebeam: Union[float, str],
        projectile: int,
        secondary: int,
        x17: bool,
        tck: Tuple,
        cov: npt.NDArray,
        tv: float = 1.0,
        te: float = 1.0,
    ):
        """Adds a new DDM spline.

        Args:
            ebeam (float): The beam energy in GeV.
            projectile (int): The projectile.
            secondary (int): The secondary particle.
            x17 (bool): Whether the spline fits x^1.7 dn/dx or just dn/dx.
            tck (Tuple): The knots of the spline.
            cov (npt.NDArray): The covariance matrix of the spline.
            tv (float, Optional): The tuning value.
            te (float, Optional): The error scale.
            spl_idx (int, Optional): The index of the spline."""

        info(5, f"Adding DDM spline for {projectile} -> {secondary} at {ebeam} GeV")
        channel = self._mk_channel(projectile, secondary)
        if channel not in self._ddm_splines:
            self._ddm_splines[channel] = _DDMChannel(projectile, secondary)

        self._ddm_splines[channel].add_entry(
            fmteb(ebeam), projectile, secondary, x17, tck, cov, tv, te
        )

    def get_spline_indices(
        self, projectile: int, secondary: int, ebeam: Union[float, str, int]
    ) -> List[int]:
        """Returns the spline indices.

        Args:
            channel (str): The channel.

        Returns:
            A list of spline indices."""

        channel = self._mk_channel(projectile, secondary)
        return [ent.spl_idx for ent in self._ddm_splines[channel].entries]

    @property
    def channels(self) -> Generator[_DDMChannel, None, None]:
        """Returns the DDM entries.

        Returns:
            A list of _DDMEntry objects."""

        for channel in self._ddm_splines:
            yield self._ddm_splines[channel]

    def clone_entry(
        self, projectile: int, secondary: int, original_ebeam: float, new_ebeam: float
    ) -> None:
        """Clones a DDM entry.

        Args:
            entry (_DDMEntry): The entry to clone."""

        entry = self.get_entry(projectile, secondary, original_ebeam)
        self.add_entry(
            fmteb(new_ebeam),
            entry.projectile,
            entry.secondary,
            entry.x17,
            entry.tck,
            entry.cov,
            1.0,  # Don't copy tuning values to cloned entry
            1.0,
        )

    def get_entry(
        self,
        projectile: int,
        secondary: int,
        ebeam: Optional[float] = None,
        idx: Optional[int] = None,
    ) -> _DDMEntry:
        """Returns a DDM entry.

        Args:
            channel (str): The channel.
            idx (int): The index of the spline.

        Returns:
            The DDM entry."""

        return self._ddm_splines[self._mk_channel(projectile, secondary)].get_entry(
            ebeam, idx
        )

    def _mk_channel(self, projectile: int, secondary: int) -> str:
        return f"{projectile}-{secondary}"


class DataDrivenModel:
    """This class implmenets the Data-Driven Model."""

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
        self.spline_db = DDMSplineDB(filename, enable_channels, exclude_projectiles)

    def ddm_matrices(self, mceq) -> Dict[Tuple[int, int], npt.NDArray]:
        """Generates a dictionary of DDM matrices for the given MCEq object.

        Args:
            mceq: An MCEq object.

        Returns:
             A dictionary with keys as tuples (projectile PDG, secondary PDG) and
        values as numpy arrays representing the corresponding DDM matrices.
        """

        _ddm_mat = dict()
        for channel in self.spline_db.channels:
            info(
                0, f"Generating {channel.projectile} -> {channel.secondary} DDM matrix"
            )
            _ddm_mat[(channel.projectile, channel.secondary)] = _generate_DDM_matrix(
                channel=channel, mceq=mceq, e_min=self.e_min, e_max=self.e_max
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

    def apply_tuning(
        self,
        projectile: int,
        secondary: int,
        ebeam: Optional[float] = None,
        spl_idx: Optional[int] = None,
        tv: float = 1.0,
        te: float = 1.0,
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

        entry = self.spline_db.get_entry(projectile, secondary, ebeam, spl_idx)
        entry.tv = tv
        entry.te = te

    def dn_dxl(self, x, projectile, secondary, ebeam, return_error=True):
        """Returns dN/dxL and error."""

        entry = self.spline_db.get_entry(projectile, secondary, ebeam)

        res = _eval_spline(
            x, entry.tck, entry.x17, entry.cov, return_error=return_error
        )
        mask = np.where(x < _pdata.mass(secondary) / ebeam)
        if isinstance(res, tuple):
            v, e = res
            v[mask] = 0.0
            e[mask] = 0.0
            return v, e
        elif isinstance(res, np.ndarray):
            res[mask] = 0

        return res

    def calc_zfactor_and_error(self, projectile, secondary, ebeam, gamma=1.7):
        """The parameter `gamma` is the CR nucleon integral spectral index."""

        entry = self.spline_db.get_entry(projectile, secondary, ebeam)

        info(3, f"Calculating Z-factor for {projectile}-->{secondary} @ {ebeam} GeV.")

        def func_int(tck_1):
            res = quad(
                _eval_spline,
                _pdata.mass(secondary) / ebeam,
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

    def calc_zfactor_and_error2(self, projectile, secondary, ebeam, fv, fe, gamma=1.7):
        """The parameter `gamma` is the CR nucleon integral spectral index.

        fv and fe are the fitted correction and its uncertainty.
        """

        entry = self.spline_db.get_entry(projectile, secondary, ebeam)

        info(3, f"Calculating Z-factor for {projectile}-->{secondary} @ {ebeam} GeV.")

        def fitfunc_center(*args, **kwargs):
            return _eval_spline(*args, **kwargs)[0]

        def fitfunc_error(*args, **kwargs):
            return _eval_spline(*args, **kwargs)[1]

        zfactor_center = quad(
            fitfunc_center,
            _pdata.mass(secondary) / ebeam,
            1.0,
            args=(entry.tck, entry.x17, entry.cov, True, gamma),
            **_QUAD_PARAMS,
        )[0]
        zfactor_error = quad(
            fitfunc_error,
            _pdata.mass(secondary) / ebeam,
            1.0,
            args=(entry.tck, entry.x17, entry.cov, True, gamma),
            **_QUAD_PARAMS,
        )[0]

        return (
            entry.tv * zfactor_center + fv * entry.te * zfactor_error,
            fe * entry.tv * zfactor_error,
        )

    def gen_matrix_variations(self, mceq):
        matrix_variations = {}
        isospin_partners = {}
        # sigmas = self._load_knot_sigmas()
        # generate a default set of matrices and save it for isospin cooking
        ddm_matrices = self.ddm_matrices(mceq)
        for channel in self.spline_db.channels:
            # for (prim, sec) in self.data_combinations:
            info(1, f"Generating vatiations for channel\n{channel}")
            channel.total_n_knots
            # tcks, dim_spls = self._dim_spl(projectile, secondary, ret_detailed=True)
            # sigs = sigmas[(prim, sec)]
            mat_db = []
            iso_part_db = []

            # Loop through the knots of each spline
            for ispl, spl_entry in enumerate(channel.entries):
                # for ispl, n in enumerate(dim_spls):
                tck = spl_entry.tck
                sig = spl_entry.knot_sigma
                # tck, sig = tcks[ispl], sigs[ispl]
                mat_db.append({})
                iso_part_db.append({})
                for iknot in range(0, spl_entry.n_knots):
                    if np.allclose(tck[1][iknot], 0.0):
                        continue
                    tck_min, tck_max, h = _spline_min_max_at_knot(tck, iknot, sig)
                    # Replace coefficients by the varied ones in data_combinations
                    spl_entry.tck = tck_max
                    mat_max = _generate_DDM_matrix(
                        channel, mceq, e_min=self.e_min, e_max=self.e_max
                    )
                    spl_entry.tck = tck_min
                    mat_min = _generate_DDM_matrix(
                        channel, mceq, e_min=self.e_min, e_max=self.e_max
                    )
                    # Restore original tck
                    spl_entry.tck = tck
                    mat_db[-1][iknot] = [mat_max, mat_min, h]
                    # Create also the varied isospin matrices
                    if channel.secondary in [321, -321] and self.enable_K0_from_isospin:
                        imat_max = 0.5 * (
                            mat_max
                            + ddm_matrices[(channel.projectile, -channel.secondary)]
                        )
                        imat_min = 0.5 * (
                            mat_min
                            + ddm_matrices[(channel.projectile, -channel.secondary)]
                        )
                        iso_part_db[-1][iknot] = [imat_max, imat_min, h]
            matrix_variations[(channel.projectile, channel.secondary)] = mat_db
            if channel.secondary in [321, -321]:
                isospin_partners[(channel.projectile, channel.secondary)] = (
                    310,
                    130,
                ), iso_part_db

        return matrix_variations, isospin_partners

    def __repr__(self) -> str:
        s = "DDM channels:\n"
        for channel in self.spline_db.channels:
            s += str(channel)
        s += "\n"
        return s


def _eval_spline(
    x: Union[float, npt.NDArray],
    tck,
    x17,
    cov,
    return_error: Optional[bool] = False,
    gamma_zfac: Optional[float] = None,
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    """
    Evaluate the spline in dn/dx.

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

    def func(tck_1: npt.NDArray) -> npt.NDArray:  # type: ignore
        return factor * np.exp(splev(x, (tck[0], tck_1, tck[2])))

    func_params = tck[1]

    if return_error:
        y, C = propagate(func, func_params, cov, **_PROPAGATE_PARAMS)
        sig_y = np.squeeze(np.sqrt(np.diag(np.atleast_1d(C))))
        return y, sig_y
    else:
        res = np.atleast_1d(func(func_params))
        res[(res < 0) | ~np.isfinite(res) | (res > _LIMIT_PROPAGATE)] = 0.0
        return res.squeeze()


def _eval_spline_and_correction(
    x: Union[float, npt.NDArray],
    tck_dndx,
    tck_corr,
    x17,
    cov,
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
        y, C = propagate(func, func_params, cov, **_PROPAGATE_PARAMS)
        sig_y = np.squeeze(np.sqrt(np.diag(np.atleast_1d(C))))
        return y, sig_y
    else:
        res = np.atleast_1d(func(func_params))
        res[(res < 0) | ~np.isfinite(res) | (res > _LIMIT_PROPAGATE)] = 0.0
        return res.squeeze()


def _gen_dndx(xbins, entry):

    x = np.sqrt(xbins[1:] * xbins[:-1])

    res = _eval_spline(x, entry.tck, entry.x17, entry.cov, return_error=False)
    res[x < entry.x_min] = 0  # type: ignore
    return entry.tv * res


def _gen_averaged_dndx(xbins, entry):

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


def _generate_DDM_matrix(
    channel: _DDMChannel,
    mceq,
    e_min: float = -1.0,
    e_max: float = -1.0,
    average=True,
) -> npt.NDArray:
    projectile = channel.projectile
    secondary = channel.secondary
    # Definitions of particles and energy grid
    e_min = mceq._energy_grid.b[0] if e_min < 0 else e_min
    e_max = mceq._energy_grid.b[-1] if e_max < 0 else e_max
    dim = mceq.dim
    projectile_mass = mceq.pman[projectile].mass
    secondary_mass = mceq.pman[secondary].mass

    # Convert from kinetic to lab energy
    elab_secondary_bins = mceq._energy_grid.b + secondary_mass
    elab_secondary_centers = mceq._energy_grid.c + secondary_mass
    elab_proj_centers = mceq._energy_grid.c + projectile_mass
    e_widths = np.diff(elab_secondary_bins)
    xgrid = elab_secondary_bins / elab_secondary_centers[-1]

    # Pull a copy of the original matrix
    try:
        # If matrices are stored on GPU, copy them to CPU
        mat = copy(mceq.pman[projectile].hadr_yields[mceq.pman[secondary]].get())
    except AttributeError:
        mat = copy(mceq.pman[projectile].hadr_yields[mceq.pman[secondary]])

    if average:
        dndx_generator = _gen_averaged_dndx
    else:
        dndx_generator = _gen_dndx

    info(
        5,
        f"DDM will be used between {e_min:.1e} GeV " + f"and {e_max:.1e} GeV",
    )
    n_datasets = channel.n_splines
    assert n_datasets > 0, "Number of splines per channel can't be zero"

    if n_datasets == 1:
        entry = channel.get_entry(idx=0)
        ie0 = np.argmin(np.abs(entry.fl_ebeam - elab_proj_centers))
        mceq_ebeam = elab_proj_centers[ie0]
        info(
            5,
            f"Dataset 0 ({projectile}, {secondary}): mceq_eidx={ie0}, "
            + f"mceq_ebeam={mceq_ebeam:4.3f}, ebeam={entry.ebeam}",
        )
        averaged_dndx = dndx_generator(xgrid, entry)

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
            entry_0 = channel.get_entry(idx=interval)
            entry_1 = channel.get_entry(idx=interval + 1)

            ie0 = np.argmin(np.abs(entry_0.fl_ebeam - elab_proj_centers))
            ie1 = np.argmin(np.abs(entry_1.fl_ebeam - elab_proj_centers))
            mceq_ebeam_0 = elab_proj_centers[ie0]
            mceq_ebeam_1 = elab_proj_centers[ie1]
            info(
                5,
                f"Dataset {interval} ({projectile}, {secondary}): mceq_eidx={ie0}, "
                + f"mceq_ebeam={mceq_ebeam_0:4.3f}, ebeam={entry_0.ebeam}",
            )
            info(
                5,
                f"Dataset {interval + 1} ({projectile}, {secondary}): "
                + f"mceq_eidx={ie1}, "
                + f"mceq_ebeam={mceq_ebeam_1:4.3f}, ebeam={entry_1.ebeam}",
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
                    m = (f1 - f0) / np.log(entry_1.fl_ebeam / entry_0.fl_ebeam)
                    mat[: ie + 1, ie] = f0 + m * np.log(eproj / entry_0.fl_ebeam)
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
