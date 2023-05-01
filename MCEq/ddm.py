import pathlib
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

import mceq_config as config

from .misc import info
from .particlemanager import _pdata
from .ddm_utils import fmteb, _eval_spline, _generate_DDM_matrix

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
        self._entries: List[_DDMEntry] = []
        self.projectile: int = projectile
        self.secondary: int = secondary

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
    """A class for maintaining DDMEntryCollections for different
    projectile and secondary particle combinations."""

    _ddm_splines: Dict[str, _DDMChannel] = {}

    def __init__(
        self,
        filename: str = str(pathlib.Path(config.data_dir) / "DDM_1.0.npy"),
        enable_channels: List[Tuple[int, int]] = [],
        exclude_projectiles: List[int] = [],
    ):
        """
        Initializes the DDMSplineDB.

        Parameters
        ----------
        filename : str, optional
            The filename of the DDM spline data file, by default "DDM_1.0.npy".
        enable_channels : List[Tuple[int, int]], optional
            List of projectile and secondary combinations to enable, by default an empty list.
        exclude_projectiles : List[int], optional
            List of projectile codes to exclude, by default an empty list.
        """
        self._load_from_file(filename, enable_channels, exclude_projectiles)

    def _load_from_file(
        self,
        filename: str,
        enable_channels: List[Tuple[int, int]],
        exclude_projectiles: List[int],
    ):
        """
        Loads DDM splines from a file.

        Parameters
        ----------
        filename : str
            The filename of the DDM spline data file.
        enable_channels : List[Tuple[int, int]]
            List of projectile and secondary combinations to enable.
        exclude_projectiles : List[int]
            List of projectile codes to exclude.
        """

        # Clean up if _load is called multiple times
        if self._ddm_splines:
            self._ddm_splines = {}

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
        """
        Adds a new DDM spline.

        Parameters
        ----------
        ebeam : Union[float, str]
            The beam energy in GeV.
        projectile : int
            The projectile.
        secondary : int
            The secondary particle.
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
        info(5, f"Adding DDM spline for {projectile} -> {secondary} at {ebeam} GeV")

        channel = self._mk_channel(projectile, secondary)
        if channel not in self._ddm_splines:
            self._ddm_splines[channel] = _DDMChannel(projectile, secondary)

        self._ddm_splines[channel].add_entry(
            fmteb(ebeam), projectile, secondary, x17, tck, cov, tv, te
        )

    def get_spline_indices(self, projectile: int, secondary: int) -> List[int]:
        """
        Returns the spline indices.

        Parameters
        ----------
        projectile : int
            The projectile.
        secondary : int
            The secondary particle.
        ebeam : Union[float, str, int]
            The beam energy in GeV.

        Returns
        -------
        List[int]
            A list of spline indices.
        """
        channel = self._mk_channel(projectile, secondary)
        return [ent.spl_idx for ent in self._ddm_splines[channel].entries]

    @property
    def channels(self) -> Generator[_DDMChannel, None, None]:
        """
        Returns the DDM channels.

        Yields
        ------
        _DDMChannel
            A generator of _DDMChannel objects.
        """
        for channel in self._ddm_splines:
            yield self._ddm_splines[channel]

    def clone_entry(
        self, projectile: int, secondary: int, original_ebeam: float, new_ebeam: float
    ) -> None:
        """
        Clones a DDM entry.

        Parameters
        ----------
        projectile : int
            The projectile.
        secondary : int
            The secondary particle.
        original_ebeam : float
            The original beam energy in GeV.
        new_ebeam : float
            The new beam energy in GeV.
        """
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
        """
        Returns a DDM entry.

        Parameters
        ----------
        projectile : int
            The projectile.
        secondary : int
            The secondary particle.
        ebeam : Optional[float], optional
            The beam energy in GeV, by default None.
        idx : Optional[int], optional
            The index of the spline, by default None.

        Returns
        -------
        _DDMEntry
            The DDM entry.
        """
        return self._ddm_splines[self._mk_channel(projectile, secondary)].get_entry(
            ebeam, idx
        )

    def _mk_channel(self, projectile: int, secondary: int) -> str:
        """
        Creates a channel key.

        Parameters
        ----------
        projectile : int
            The projectile.
        secondary : int
            The secondary particle.

        Returns
        -------
        str
            The channel key.
        """
        return f"{projectile}-{secondary}"


# class DataDrivenModel:
#     """This class implmenets the Data-Driven Model."""

#     def __init__(
#         self,
#         filename: str = str(pathlib.Path(config.data_dir) / "DDM_1.0.npy"),
#         e_min: float = 0,
#         e_max: float = np.inf,
#         enable_channels: List[Tuple[int, int]] = [],
#         exclude_projectiles: List[int] = [],
#         enable_K0_from_isospin: bool = True,
#     ):
#         """Initializes a DataDrivenModel object.

#         Args:
#             filename: A string representing the file name containing the data
#                 points used to build the model.
#             e_min: A float representing the minimum energy range where DDM
#                 cross sections overwrite original MCEq matrices.
#                 Default is 0.
#             e_max: A float representing the maximum energy range where DDM
#                 cross sections overwrite original MCEq matrices.
#                 Default is infinity.
#             enable_channels: A list of strings representing the channels to
#                 use in building the model.
#                 Default is an empty list.
#             exclude_projectiles: A list of integers representing the projectiles
#                 to exclude in building the model.
#                 Default is an empty list.
#             enable_K0_from_isospin: A bool representing whether to use isospin
#                 relation for K0S/L.
#                 Default is True.
#         """
#         self.e_min = e_min
#         self.e_max = e_max
#         self.enable_K0_from_isospin = enable_K0_from_isospin
#         self.spline_db = DDMSplineDB(filename, enable_channels, exclude_projectiles)

#     def ddm_matrices(self, mceq) -> Dict[Tuple[int, int], npt.NDArray]:
#         """Generates a dictionary of DDM matrices for the given MCEq object.

#         Args:
#             mceq: An MCEq object.

#         Returns:
#              A dictionary with keys as tuples (projectile PDG, secondary PDG) and
#         values as numpy arrays representing the corresponding DDM matrices.
#         """

#         _ddm_mat = dict()
#         for channel in self.spline_db.channels:
#             info(
#                 0, f"Generating {channel.projectile} -> {channel.secondary} DDM matrix"
#             )
#             _ddm_mat[(channel.projectile, channel.secondary)] = _generate_DDM_matrix(
#                 channel=channel, mceq=mceq, e_min=self.e_min, e_max=self.e_max
#             )

#         if self.enable_K0_from_isospin:
#             info(3, "Generating DDM K0 matrices from isospin symm.")
#             K0SL_mat = 0.5 * (_ddm_mat[(2212, 321)] + _ddm_mat[(2212, -321)])
#             _ddm_mat[(2212, 310)] = K0SL_mat
#             _ddm_mat[(2212, 130)] = K0SL_mat
#             if (-211, 321) in _ddm_mat and (-211, -321) in _ddm_mat:
#                 K0SL_mat = 0.5 * (_ddm_mat[(-211, 321)] + _ddm_mat[(-211, -321)])
#                 _ddm_mat[(-211, 310)] = K0SL_mat
#                 _ddm_mat[(-211, 130)] = K0SL_mat

#         return _ddm_mat

#     def apply_tuning(
#         self,
#         projectile: int,
#         secondary: int,
#         ebeam: Optional[float] = None,
#         spl_idx: Optional[int] = None,
#         tv: float = 1.0,
#         te: float = 1.0,
#     ) -> None:
#         """
#         Modify the tuning values and trim errors in the data_combinations dictionary.

#         Args:
#             prim: Primary particle code
#             sec: Secondary particle code
#             ebeam: Lab frame beam energy in GeV (optional)
#             tv: Scaling factor for the central value
#             te: Scaling factor for the errors
#             spl_idx: Index of the spline to modify (optional)

#         Raises:
#             AssertionError: If neither `spl_idx` nor `ebeam` is set.
#         """

#         entry = self.spline_db.get_entry(projectile, secondary, ebeam, spl_idx)
#         entry.tv = tv
#         entry.te = te

#     def dn_dxl(self, x, projectile, secondary, ebeam, return_error=True):
#         """Returns dN/dxL and error."""

#         entry = self.spline_db.get_entry(projectile, secondary, ebeam)

#         res = _eval_spline(
#             x, entry.tck, entry.x17, entry.cov, return_error=return_error
#         )
#         mask = np.where(x < _pdata.mass(secondary) / ebeam)
#         if isinstance(res, tuple):
#             v, e = res
#             v[mask] = 0.0
#             e[mask] = 0.0
#             return v, e
#         elif isinstance(res, np.ndarray):
#             res[mask] = 0

#         return res

#     def __repr__(self) -> str:
#         s = "DDM channels:\n"
#         for channel in self.spline_db.channels:
#             s += str(channel)
#         s += "\n"
#         return s


class DataDrivenModel:
    """This class implements the Data-Driven Model."""

    def __init__(
        self,
        filename: str = str(pathlib.Path(config.data_dir) / "DDM_1.0.npy"),
        e_min: float = 0,
        e_max: float = np.inf,
        enable_channels: List[Tuple[int, int]] = [],
        exclude_projectiles: List[int] = [],
        enable_K0_from_isospin: bool = True,
    ):
        """
        Initializes a DataDrivenModel object.

        Parameters
        ----------
        filename : str, optional
            The filename of the data file used to build the model, by default "DDM_1.0.npy".
        e_min : float, optional
            The minimum energy range where DDM cross sections overwrite original MCEq matrices, by default 0.
        e_max : float, optional
            The maximum energy range where DDM cross sections overwrite original MCEq matrices, by default np.inf.
        enable_channels : List[Tuple[int, int]], optional
            List of projectile and secondary combinations to enable, by default an empty list.
        exclude_projectiles : List[int], optional
            List of projectile codes to exclude, by default an empty list.
        enable_K0_from_isospin : bool, optional
            Whether to use isospin relation for K0S/L, by default True.
        """
        self.e_min = e_min
        self.e_max = e_max
        self.enable_K0_from_isospin = enable_K0_from_isospin
        self.spline_db = DDMSplineDB(filename, enable_channels, exclude_projectiles)

    def ddm_matrices(self, mceq) -> Dict[Tuple[int, int], npt.NDArray]:
        """
        Generates a dictionary of DDM matrices for the given MCEq object.

        Parameters
        ----------
        mceq : object
            An MCEq object.

        Returns
        -------
        Dict[Tuple[int, int], npt.NDArray]
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

        Parameters
        ----------
        projectile : int
            Primary particle code.
        secondary : int
            Secondary particle code.
        ebeam : Optional[float], optional
            Lab frame beam energy in GeV, by default None.
        spl_idx : Optional[int], optional
            Index of the spline to modify, by default None.
        tv : float, optional
            Scaling factor for the central value, by default 1.0.
        te : float, optional
            Scaling factor for the errors, by default 1.0.

        Raises
        ------
        AssertionError
            If neither `spl_idx` nor `ebeam` is set.
        """

        entry = self.spline_db.get_entry(projectile, secondary, ebeam, spl_idx)
        entry.tv = tv
        entry.te = te

    def dn_dxl(
        self,
        x: np.ndarray,
        projectile: int,
        secondary: int,
        ebeam: float,
        return_error: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns dN/dxL and error.

        Parameters
        ----------
        x : np.ndarray
            Array of x values.
        projectile : int
            The projectile.
        secondary : int
            The secondary particle.
        ebeam : float
            Lab frame beam energy in GeV.
        return_error : bool, optional
            Whether to return the error, by default True.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            The dN/dxL values and optional error.

        """
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

    def __repr__(self) -> str:
        s = "DDM channels:\n"
        for channel in self.spline_db.channels:
            s += str(channel)
        s += "\n"
        return s
