from copy import copy

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splev

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


class DataDrivenModel(object):
    def __init__(self, filename, mceq, e_min=0, e_max=np.inf, enable_channels=[]):
        # Energy range where DDM cross sections overwrite
        # original MCEq matrices
        self.e_min = e_min
        self.e_max = e_max

        self.data_combinations = self._load_file(filename, enable_channels)
        self.channels = sorted(list(self.data_combinations.keys()))

        self.ddm_matrices = self._init_scenario(mceq)

    def __repr__(self) -> str:
        s = "DDM channels:\n"
        for prim, sec in self.channels:
            s += f"\t{prim} -> {sec}:\n"
            for iset, (ebeam, x17, coeffs, cov) in enumerate(
                self.data_combinations[(prim, sec)]
            ):
                s += f"\t\t{iset}: ebeam = {ebeam} GeV, x17={x17}\n"
        s += "\n"
        return s

    def clone_datapoint(self, prim, sec, original_ebeam, new_ebeam):

        assert (prim, sec) in self.data_combinations
        available_energies = [s[0] for s in self.data_combinations[(prim, sec)]]
        if original_ebeam not in available_energies:
            raise Exception("Energy not one of", available_energies)
        for (ebeam, x17, coeffs, cov) in self.data_combinations[(prim, sec)]:
            if ebeam == original_ebeam:
                self.data_combinations[(prim, sec)].append(
                    (new_ebeam, x17, coeffs, cov)
                )
        self.data_combinations[(prim, sec)] = sorted(
            self.data_combinations[(prim, sec)]
        )

    # def apply_tuning(self, **kwargs):



    def _load_file(self, filename, enable_channels):
        splines = np.load(filename, allow_pickle=True, encoding="latin1").item()
        data_combinations = {}
        for (prim, sec, ebeam, x17) in splines:
            if enable_channels and (prim, sec) not in enable_channels:
                continue
            if (prim, sec) not in data_combinations:
                data_combinations[(prim, sec)] = []

            data_combinations[(prim, sec)].append(
                (
                    ebeam,
                    x17,
                    splines[(prim, sec, ebeam, x17)][0],  # Coefficients
                    splines[(prim, sec, ebeam, x17)][1],  # Covariance matrix
                )
            )
        # Sort datasets by beam energy
        for (prim, sec) in data_combinations:
            data_combinations[(prim, sec)] = sorted(data_combinations[(prim, sec)])

        return data_combinations

    def _init_scenario(self, mceq):
        ddm_mat = dict()
        for (prim, sec) in self.channels:
            info(2, f"Generating {prim} -> {sec} DDM matrix")
            ddm_mat[(prim, sec)] = self._generate_DDM_matrix(
                prim=prim, sec=sec, mceq=mceq
            )

        return ddm_mat

    def _fitfunc(self, x, tck, x17):
        factor = x ** (-1.7) if x17 else 1.0

        if len(tck) == 3:
            return factor * np.exp(splev(x, tck))

        (tck_fit, tck_corr) = tck
        return factor * np.exp(splev(x, tck_fit)) * splev(x, tck_corr)

    def _gen_dndx(self, xbins, tck, x17):
        x = 0.5 * (xbins[1:] + xbins[:-1])
        assert len(tck) in [2, 3], "Unknown number of spline coeffs"

        return self._fitfunc(self, x, tck, x17)

    def _gen_averaged_dndx(self, xbins, tck, x17):

        xwidths = np.diff(xbins)
        integral = np.zeros_like(xwidths)

        assert len(tck) in [2, 3], "Unknown number of spline coeffs"

        for ib in range(len(integral)):
            if xbins[ib + 1] > 1:
                integral[ib] = (
                    quad(self._fitfunc, xbins[ib], 1.0, args=(tck, x17))[0]
                    / xwidths[ib]
                )
            else:
                integral[ib] = (
                    quad(self._fitfunc, xbins[ib], xbins[ib + 1], args=(tck, x17))[0]
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
            ebeam, x17, tck0, _ = self.data_combinations[(prim, sec)][0]
            ie0 = np.argmin(np.abs(ebeam - elab_proj_centers))
            mceq_ebeam = elab_proj_centers[ie0]
            info(
                5,
                f"Dataset 0 ({prim}, {sec}): mceq_eidx={ie0}, "
                + f"mceq_ebeam={mceq_ebeam:4.3f}, ebeam={ebeam}",
            )
            averaged_dndx = dndx_generator(xgrid, tck0, x17)

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
                # tck0, tck1 = coeffs[interval], coeffs[interval + 1]
                ebeam_0, x17_0, tck0, _ = self.data_combinations[(prim, sec)][interval]
                ebeam_1, x17_1, tck1, _ = self.data_combinations[(prim, sec)][
                    interval + 1
                ]
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

                averaged_dndx_0 = dndx_generator(xgrid, tck0, x17_0)
                averaged_dndx_1 = dndx_generator(xgrid, tck1, x17_1)

                for ie, eproj in enumerate(elab_proj_centers):
                    if (eproj < config.e_min) or (eproj > config.e_max):
                        print("Skip out of range energy", eproj)
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
