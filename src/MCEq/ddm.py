"""Data-Driven Model (DDM) of hadron production, from splines fitted to fixed-target data."""

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from jacobi import propagate
from scipy.integrate import quad

from MCEq import config
from MCEq.ddm_utils import (
    _PROPAGATE_PARAMS,
    _QUAD_PARAMS,
    _eval_spline,
    _generate_DDM_matrix,
    fmteb,
)
from MCEq.misc import info
from MCEq.particlemanager import _pdata

#: Spline file read when no filename is given, looked up in ``config.data_dir``.
_DEFAULT_DDM_FILE = "DDM_1.0.npy"

# isospin symmetries used in the DDM
isospin_partners = {2212: 2112, -211: 211}
isospin_symmetries = {
    2112: {
        321: 321,  # Approximate neutron K+ production with the proton K+ yield
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
    """One DDM spline: dN/dx (or x^1.7 dN/dx if ``x17``) of ``projectile ->
    secondary`` at beam energy ``ebeam`` (GeV, as :func:`fmteb` string), with
    the coefficient covariance ``cov``, tuning value ``tv`` and error scale ``te``.
    """

    ebeam: str
    projectile: int
    secondary: int
    x17: bool
    tck: Tuple
    cov: npt.NDArray[np.float64]
    tv: float = 1.0
    te: float = 1.0

    @property
    def knot_sigma(self) -> npt.NDArray:
        """Standard deviation of the spline coefficients, scaled by ``te``."""
        return self.te * np.sqrt(np.diag(self.cov))

    @property
    def fl_ebeam(self) -> float:
        return float(self.ebeam)

    @property
    def n_knots(self) -> int:
        return len(self.tck[1])

    @property
    def x_min(self) -> float:
        """Kinematic lower limit of x, secondary mass over beam energy."""
        return _pdata.mass(self.secondary) / float(self.ebeam)

    def calc_zfactor_and_error(self, gamma: float = 1.7):
        """Z-factor, the integral of ``x**gamma dN/dx`` over ``[x_min, 1]``, and
        its error propagated through the integral; ``tv`` and ``te`` are not applied.
        """
        info(
            3,
            f"Calculating Z-factor for {self.projectile} --> "
            f"{self.secondary} @ {self.ebeam} GeV.",
        )

        def func_int(tck_1):
            res = quad(
                _eval_spline,
                self.x_min,
                1.0,
                args=(
                    (self.tck[0], tck_1, self.tck[2]),
                    self.x17,
                    self.cov,
                    False,
                    gamma,
                ),
                **_QUAD_PARAMS,
            )[0]
            res = np.atleast_1d(res)
            res[(res < 0) | ~np.isfinite(res)] = 0.0
            return res.squeeze()

        y, C = propagate(func_int, self.tck[1], self.cov, **_PROPAGATE_PARAMS)
        return y, np.sqrt(C)

    def calc_zfactor_and_error2(self, gamma: float = 1.7):
        """Tuned Z-factor shifted by ``te`` errors, and its error, from
        integrating the central spectrum and its pointwise error separately.
        """
        info(
            3,
            f"Calculating Z-factor for {self.projectile} --> "
            + f"{self.secondary} @ {self.ebeam} GeV.",
        )

        def fitfunc_center(*args, **kwargs):
            return _eval_spline(*args, **kwargs)[0]

        def fitfunc_error(*args, **kwargs):
            return _eval_spline(*args, **kwargs)[1]

        args = (self.tck, self.x17, self.cov, True, gamma)
        zfactor_center = quad(
            fitfunc_center, self.x_min, 1.0, args=args, **_QUAD_PARAMS
        )[0]
        zfactor_error = quad(fitfunc_error, self.x_min, 1.0, args=args, **_QUAD_PARAMS)[
            0
        ]

        return (
            self.tv * zfactor_center + self.tv * self.te * zfactor_error,
            self.te * self.tv * zfactor_error,
        )


class DataDrivenModel:
    """The Data-Driven Model: DDM splines and the yield matrices built from them.

    Args:
      filename (str): spline file, by default ``DDM_1.0.npy`` in ``config.data_dir``.
      e_min, e_max (float): total projectile energy range (GeV) in which DDM
        matrices replace the MCEq ones; negative means the grid edge.
      enable_channels (list): ``(projectile, secondary)`` pairs to load; empty
        loads all.
      exclude_projectiles (list): projectile codes (absolute) to skip.
      enable_K0_from_isospin (bool): derive K0S/L from the K+- matrices.
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        e_min: float = -1.0,
        e_max: float = -1.0,
        enable_channels: List[Tuple[int, int]] = [],
        exclude_projectiles: List[int] = [],
        enable_K0_from_isospin: bool = True,
    ):
        self.e_min = e_min
        self.e_max = e_max
        self.enable_K0_from_isospin = enable_K0_from_isospin
        if filename is None:
            filename = str(pathlib.Path(config.data_dir) / _DEFAULT_DDM_FILE)

        #: ``{(projectile, secondary): [_DDMEntry, ...]}``, sorted by beam energy
        self.channels: Dict[Tuple[int, int], List[_DDMEntry]] = {}
        spl_file = np.load(filename, allow_pickle=True, encoding="latin1").item()
        for (projectile, secondary, ebeam, x17), spline in spl_file.items():
            if (enable_channels and (projectile, secondary) not in enable_channels) or (
                abs(projectile) in exclude_projectiles
            ):
                continue
            entries = self.channels.setdefault((projectile, secondary), [])
            if any(e.ebeam == fmteb(ebeam) for e in entries):
                raise ValueError(f"Duplicate entry for ebeam = {ebeam} GeV.")
            info(5, f"Adding DDM spline for {projectile} -> {secondary} at {ebeam} GeV")
            entries.append(
                _DDMEntry(
                    fmteb(ebeam), projectile, secondary, x17, spline[0], spline[1]
                )
            )
            entries.sort(key=lambda entry: entry.fl_ebeam)

    def get_entry(
        self,
        projectile: int,
        secondary: int,
        ebeam: Optional[float] = None,
        idx: Optional[int] = None,
    ) -> _DDMEntry:
        """The entry of a channel at beam energy ``ebeam`` or at position ``idx``."""
        assert (ebeam is not None) != (idx is not None), "Define either ebeam or idx"
        entries = self.channels[(projectile, secondary)]
        if idx is not None:
            return entries[idx]
        for entry in entries:
            if entry.ebeam == fmteb(ebeam):
                return entry
        raise ValueError(f"No entry for ebeam = {fmteb(ebeam)} GeV.")

    def find_channel(self, projectile: int, secondary: int) -> List[_DDMEntry]:
        """Entries of the channel ``projectile -> secondary``, sorted by beam energy."""
        return self.channels[(projectile, secondary)]

    def ddm_matrices(self, mceq) -> Dict[Tuple[int, int], npt.NDArray]:
        """DDM yield matrices ``{(projectile, secondary): matrix}`` on the grid of
        ``mceq``, including the K0 and isospin-partner channels.
        """
        _ddm_mat = {}
        for (projectile, secondary), entries in self.channels.items():
            info(2, f"Generating {projectile} -> {secondary} DDM matrix")
            mat = _generate_DDM_matrix(entries, mceq)
            # projectile energies outside [e_min, e_max] keep the MCEq columns
            proj = mceq.pman[projectile]
            e_proj = mceq.e_grid + proj.mass
            keep = np.zeros(len(e_proj), dtype=bool)
            if self.e_min >= 0:
                keep |= e_proj < self.e_min
            if self.e_max >= 0:
                keep |= e_proj > self.e_max
            if keep.any():
                orig = np.asarray(proj.hadr_yields[mceq.pman[secondary]])
                mat[:, keep] = orig[:, keep]
            _ddm_mat[(projectile, secondary)] = mat

        if self.enable_K0_from_isospin:
            info(3, "Generating DDM K0 matrices from isospin symm.")
            assert (2212, 321) in _ddm_mat, f"Missing K+ matrices in {_ddm_mat.keys()}"
            K0SL_mat = 0.5 * (_ddm_mat[(2212, 321)] + _ddm_mat[(2212, -321)])
            _ddm_mat[(2212, 310)] = K0SL_mat
            _ddm_mat[(2212, 130)] = K0SL_mat
            if (-211, 321) in _ddm_mat and (-211, -321) in _ddm_mat:
                K0SL_mat = 0.5 * (_ddm_mat[(-211, 321)] + _ddm_mat[(-211, -321)])
                _ddm_mat[(-211, 310)] = K0SL_mat
                _ddm_mat[(-211, 130)] = K0SL_mat

        # The isospin partner (n for p, pi+ for pi-) gets the same matrix.
        for (prim, sec), mat in list(_ddm_mat.items()):
            iso_prim = isospin_partners[prim]
            iso_sec = isospin_symmetries[iso_prim][sec]
            info(5, f"DDM {prim} --> {sec}, isospin: {iso_prim} --> {iso_sec}")
            _ddm_mat[(iso_prim, iso_sec)] = mat

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
        """Set the tuning value ``tv`` and error scale ``te`` of one entry,
        selected by ``ebeam`` or ``spl_idx``."""
        entry = self.get_entry(projectile, secondary, ebeam, spl_idx)
        entry.tv = tv
        entry.te = te

    def dn_dxl(
        self,
        x: np.ndarray,
        projectile: int,
        secondary: int,
        ebeam: float,
        return_error: bool = True,
    ):
        """dN/dx_L at ``x`` (and its error), zero below the kinematic limit."""
        entry = self.get_entry(projectile, secondary, ebeam)

        res = _eval_spline(
            x, entry.tck, entry.x17, entry.cov, return_error=return_error
        )
        mask = np.where(x < _pdata.mass(secondary) / ebeam)
        if isinstance(res, tuple):
            v, e = res
            v[mask] = 0.0
            e[mask] = 0.0
            return v, e
        if isinstance(res, np.ndarray):
            res[mask] = 0

        return res

    def __repr__(self) -> str:
        s = "DDM channels:\n"
        for (projectile, secondary), entries in self.channels.items():
            s += f"\t{projectile} -> {secondary}:\n"
            for iset, dc in enumerate(entries):
                s += f"\t\t{iset}: ebeam = {dc.ebeam} GeV, x17={dc.x17}, "
                s += f"tune v|e={dc.tv:4.3f}|{dc.te:4.3f}\n"
        s += "\n"
        return s
