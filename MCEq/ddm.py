from copy import copy

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splev
from jacobi import propagate

from . import config
from .misc import info

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
    },
}

propagate_params = dict(maxiter=10, maxgrad=1, method=0)
quad_params = dict(limit=150, epsabs=1e-5)


class DataDrivenModel(object):
    def __init__(
        self,
        filename,
        e_min=0,
        e_max=np.inf,
        enable_channels=[],
        exclude_projectiles=[],
    ):
        # Energy range where DDM cross sections overwrite
        # original MCEq matrices
        self.e_min = e_min
        self.e_max = e_max

        self.data_combinations = self._load_file(
            filename, enable_channels, exclude_projectiles
        )
        self.data_energy_map, self.channels = {}, []
        self._sort_datasets()
        self._ddm_matrices = None

    def ddm_matrices(self, mceq):
        if self._ddm_matrices is None:
            self._ddm_matrices = self._init_scenario(mceq)
        return self._ddm_matrices

    def clone_datapoint(self, prim, sec, original_ebeam, new_ebeam):

        assert (prim, sec) in self.data_combinations
        available_energies = [s[0] for s in self.data_combinations[(prim, sec)]]
        if original_ebeam not in available_energies:
            raise Exception("Energy not one of", available_energies)
        for (ebeam, x17, coeffs, cov, _, _) in self.data_combinations[(prim, sec)]:
            # Do not copy tuning values to new data points
            if ebeam == original_ebeam:
                self.data_combinations[(prim, sec)].append(
                    [new_ebeam, x17, coeffs, cov, 1.0, 1.0]
                )
        self._sort_datasets()

    def apply_tuning(self, prim, sec, ebeam, tv, te=1.0):
        set_idx = self._unpack_coeff(prim, sec, ebeam, return_set_idx=True)[1]

        self.data_combinations[(prim, sec)][set_idx][4] = tv
        self.data_combinations[(prim, sec)][set_idx][5] = te
        self._sort_datasets()

    def _fitfunc(self, x, tck, x17, cov, tv, te, return_error=False, gamma_zfac=None):

        # assert not ((abs(tv) > 0.0) and return_error), (
        #     f"Simultaneous error + tuning not implemented tv={tv}, "
        #     + f"return_err={return_error}"
        # )

        if gamma_zfac is None:
            factor = x ** (-1.7) if x17 else 1.0
        else:
            factor = x ** (gamma_zfac) if not x17 else x ** (gamma_zfac - 1.7)

        if len(tck) == 3:

            def func(tck_1):
                return factor * np.exp(splev(x, (tck[0], tck_1, tck[2])))

            func_params = tck[1]

        else:
            (tck_fit, tck_corr) = tck
            func_params = np.hstack([tck_fit[1], tck_corr[1]])

            def func(params):
                tck_fit_1 = params[: len(tck_fit[1])]
                tck_corr_1 = params[len(tck_fit[1]) :]
                return (
                    factor
                    * np.exp(splev(x, (tck_fit[0], tck_fit_1, tck_fit[2])))
                    * splev(x, (tck_corr[0], tck_corr_1, tck_corr[2]))
                )

        if return_error:
            y, C = propagate(func, func_params, cov, **propagate_params)
            # Handle cases where y and C are scalars
            sig_y = np.squeeze(np.sqrt(np.diag(np.atleast_1d(C))))
            return tv * y, tv * te * sig_y
        else:
            return tv * func(func_params)

    def dn_dxl(self, x, prim, sec, ebeam, return_error=True):
        """Returns dN/dxL and error."""

        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        return self._fitfunc(x, tck, x17, cov, tv, te, return_error=return_error)

    def calc_zfactor_and_error(self, prim, sec, ebeam, gamma=1.7):

        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        info(1, f"Calculating Z-factor for {prim}-->{sec} @ {ebeam} GeV.")

        def func_int(tck_1):
            return quad(
                self._fitfunc,
                0.0,
                1.0,
                args=((tck[0], tck_1, tck[2]), x17, cov, tv, te, False, gamma),
                **quad_params,
            )[0]

        y, C = propagate(func_int, tck[1], cov, **propagate_params)
        sig_y = np.sqrt(C)

        return y, sig_y

    def _gen_dndx(self, xbins, prim, sec, ebeam):
        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        x = np.sqrt(xbins[1:] * xbins[:-1])

        return self._fitfunc(x, tck, x17, cov, tv, te, return_error=True)

    def _gen_averaged_dndx(self, xbins, prim, sec, ebeam):

        (ebeam, x17, tck, cov, tv, te) = self._unpack_coeff(prim, sec, ebeam)

        xwidths = np.diff(xbins)
        integral = np.zeros_like(xwidths)

        for ib in range(len(integral)):
            if xbins[ib + 1] > 1:
                integral[ib] = (
                    quad(
                        self._fitfunc,
                        xbins[ib],
                        1.0,
                        args=(tck, x17, cov, tv, te),
                        **quad_params,
                    )[0]
                    / xwidths[ib]
                )
            else:
                integral[ib] = (
                    quad(
                        self._fitfunc,
                        xbins[ib],
                        xbins[ib + 1],
                        args=(tck, x17, cov, tv, te),
                        **quad_params,
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
            ebeam = self._unpack_coeff(prim, sec, set_idx=0)[0]
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
                ebeam_0 = self._unpack_coeff(prim, sec, set_idx=interval)[0]
                ebeam_1 = self._unpack_coeff(prim, sec, set_idx=interval + 1)[0]

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

    def _load_file(self, filename, enable_channels, exclude_projectiles):
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
                [
                    ebeam,  # Lab frame beam energy in GeV
                    x17,  # If splines fit x^1.7 dn/dx or just dn/dx
                    splines[(prim, sec, ebeam, x17)][0],  # Coefficients
                    splines[(prim, sec, ebeam, x17)][1],  # Covariance matrix
                    1.0,  # scale of central value
                    1.0,  # scale of errors
                ]
            )
        return data_combinations

    def _load_knot_sigmas(self):
        _knot_sigmas = {}
        for prim, sec in self.data_combinations:
            _knot_sigmas[(prim, sec)] = []
            for iset, (ebeam, _, tck, cov, _, _) in enumerate(
                self.data_combinations[(prim, sec)]
            ):
                info(5, f"Calculating knot error for {prim},{sec},{ebeam}.")
                _knot_sigmas[(prim, sec)].append(np.sqrt(np.diag(cov)))
        return _knot_sigmas

    def _init_scenario(self, mceq):
        ddm_mat = dict()
        for (prim, sec) in self.channels:
            info(2, f"Generating {prim} -> {sec} DDM matrix")
            ddm_mat[(prim, sec)] = self._generate_DDM_matrix(
                prim=prim, sec=sec, mceq=mceq
            )

        return ddm_mat

    def _dim_spl(self, prim, sec, ret_detailed=False):
        assert (
            prim,
            sec,
        ) in self.data_combinations, f"Unknown ({prim},{sec}) combination supplied"
        channel_cross_sections = self.data_combinations[(prim, sec)]

        # Count number of knots in each spline
        tcks = [li[2] for li in channel_cross_sections]
        # Note: There was a "-1" in len(tck0[0]) - 1 for the single spline case
        dim_spls = [len(tck[0]) for tck in tcks]
        ntot = sum(dim_spls)

        if ret_detailed:
            return tcks, dim_spls
        else:
            return ntot

    def _spline_min_max_at_knot(self, tck, iknot, sigma):
        """Return variation of spline coefficients by 1 sigma."""

        assert iknot <= len(tck[1]) - 1, f"Invalid knot {iknot} requested"

        _c_min = np.copy(tck[1])
        _c_max = np.copy(tck[1])

        h = sigma[iknot]

        _c_min[iknot] = tck[1][iknot] - h
        _c_max[iknot] = tck[1][iknot] + h

        _tck_min = (tck[0], _c_min, tck[-1])
        _tck_max = (tck[0], _c_max, tck[-1])

        return _tck_min, _tck_max, h

    def gen_matrix_variations(self, mceq, **kwargs):
        matrix_variations = {}
        sigmas = self._load_knot_sigmas()
        for (prim, sec) in self.data_combinations:
            info(1, f"Generating vatiations for {prim} -> {sec}")
            tcks, dim_spls = self._dim_spl(prim, sec, ret_detailed=True)
            sigs = sigmas[(prim, sec)]
            mat_db = []

            # Loop through the knots of each spline
            for ispl, n in enumerate(dim_spls):
                tck, sig = tcks[ispl], sigs[ispl]
                mat_db.append({})
                for iknot in range(0, n):
                    if np.allclose(tck[1][iknot], 0.0):
                        continue
                    tck_min, tck_max, h = self._spline_min_max_at_knot(tck, iknot, sig)
                    # Replace coefficients by the varied ones in data_combinations
                    self.data_combinations[(prim, sec)][ispl][2] = tck_max
                    mat_max = self._generate_DDM_matrix(prim, sec, mceq)
                    self.data_combinations[(prim, sec)][ispl][2] = tck_min
                    mat_min = self._generate_DDM_matrix(prim, sec, mceq)
                    # Restore original tck
                    self.data_combinations[(prim, sec)][ispl][2] = tck
                    mat_db[-1][iknot] = [mat_max, mat_min, h]

            matrix_variations[(prim, sec)] = mat_db

        return matrix_variations

    def _unpack_coeff(self, prim, sec, ebeam=None, set_idx=None, return_set_idx=False):
        assert (ebeam is not None) != (
            set_idx is not None
        ), "Define either ebeam or set_idx"
        assert (
            prim,
            sec,
        ) in self.data_combinations, f"({prim},{sec}) not in valid combinations"
        try:
            if ebeam:
                set_idx = self.data_energy_map[(prim, sec, ebeam)]
            else:
                if set_idx >= len(self.data_combinations[(prim, sec)]):
                    raise KeyError
        except KeyError:
            raise Exception(
                f"Combination ({prim}, {sec}, {ebeam}) not found in\n"
                + "\n".join(
                    [
                        "{0}, {1}, {2}".format(*a)
                        for a in sorted(self.data_energy_map.keys())
                    ]
                )
            )
        ebeam, x17, tck, cov, tv, te = self.data_combinations[(prim, sec)][set_idx]
        assert len(tck) in [2, 3], "Unknown number of spline coeffs"
        if return_set_idx:
            return (ebeam, x17, tck, cov, tv, te), set_idx
        return (ebeam, x17, tck, cov, tv, te)

    def _sort_datasets(self):
        # Sort datasets by beam energy
        self.data_energy_map = {}

        for (prim, sec) in self.data_combinations:
            self.data_combinations[(prim, sec)] = sorted(
                self.data_combinations[(prim, sec)]
            )
            for iset, (ebeam, _, _, _, _, _) in enumerate(
                self.data_combinations[(prim, sec)]
            ):
                self.data_energy_map[(prim, sec, ebeam)] = iset
        self.channels = sorted(list(self.data_combinations.keys()))
        # Reset the matrices to make sure MCEq takes the most recent ones
        self._ddm_matrices = None

    def __repr__(self) -> str:
        s = "DDM channels:\n"
        for prim, sec in self.channels:
            s += f"\t{prim} -> {sec}:\n"
            for iset, (ebeam, x17, coeffs, _, tv, te) in enumerate(
                self.data_combinations[(prim, sec)]
            ):
                s += f"\t\t{iset}: ebeam = {ebeam} GeV, x17={x17}, "
                s += f"tune v|e={tv:4.3f}|{te:4.3f}\n"
        s += "\n"
        return s
