# -*- coding: utf-8 -*-
"""
:mod:`MCEq.energy_solvers` --- finite difference operators and schemes for energy derivative
============================================================================================

The module contains classes which are initialized by :class:`MCEq.core.MCEqRun` and called
by time integrators in :mod:`MCEq.time_solvers`. Further description will follow later.
"""
import numpy as np
from mceq_config import config, dbg


class ChangCooper(object):
    def __init__(self, ebins, dEdX, mu_lidx_nsp, 
                 mu_selector, B_fake=1e-5):

        self.mu_dEdX = dEdX  #dlnE/dX
        self.dE = ebins[1:] - ebins[:-1]
        self.ebins = ebins
        self.ecenters = np.sqrt(ebins[1:] + ebins[:-1])
        self.de = self.dE.size
        self.muloss_min_step = config['muon_energy_loss_min_step']
        self.lidx, self.nmuspec = mu_lidx_nsp
        self.mu_selector = mu_selector
        # Energy loss term 1 order
        self.A = dEdX  #np.log(-dEdX)
        # Energy loss second order
        self.B = B_fake * self.ebins
        # self._set_B(B_fake)

        self._setup_solver(config['muon_energy_loss_min_step'])

    def _set_B(self, B_fake):
        self.B = B_fake * np.ones(self.de + 1)

    def _compute_delta(self):
        bscale = 1
        wpl = -self.A[1:] / (bscale * self.B[1:]) * self.dE
        self.delta_pl = 1 / wpl - 1 / (np.exp(wpl) - 1)
        wmi = -self.A[:-1] / (bscale * self.B[:-1]) * self.dE
        self.delta_mi = 1 / wmi - 1 / (np.exp(wmi) - 1)

    def _setup_trigiag(self, dX, force_delta=None, force_B=None):

        if force_B is not None:
            self.B = force_B * np.ones_like(self.A)

        if force_delta is not None:
            self.delta_pl = self.delta_mi = force_delta * np.ones_like(
                self.delta_pl)
        else:
            self._compute_delta()

        A, B, dpl, dmi, dE = (self.A, self.B, self.delta_pl, self.delta_mi,
                              self.dE)

        #Chang-Cooper
        dl = -dmi * (A[:-1] + B[:-1] / dE)
        du = (1 - dpl) * (A[1:] - B[1:] / dE)
        dc_lhs = (2 * dE / dX + A[1:] * dpl + B[1:] / dE *
                          (1 - dpl) - A[:-1] * (1 - dmi) - B[:-1] / dE * dmi)
        dc_rhs = (2 * dE / dX - A[1:] * dpl - B[1:] / dE *
                          (1 - dpl) + A[:-1] * (1 - dmi) - B[:-1] / dE * dmi)

        # Crank-Nicholson
        # du = A[1:]
        # dl = -A[:-1]
        # dc_lhs = 4*dE/dX + A[1:] - A[:-1]
        # dc_rhs = 4*dE/dX - A[1:] + A[:-1]

        return dl, du, dc_lhs, dc_rhs

    def _setup_solver(self, dX=1, **kwargs):
        from scipy.sparse.linalg import factorized
        from scipy.sparse import dia_matrix, block_diag
        dl, du, dc_lhs, dc_rhs = self._setup_trigiag(dX=dX, **kwargs)

        data = np.vstack([dl, dc_lhs, du])
        offsets = np.array([-1, 0, 1])
        lhs_mat = dia_matrix(
            (data, offsets),
            shape=(self.de, self.de)).tocsc()
        data = np.vstack([-dl, dc_rhs, -du])
        self.rhs_mat = dia_matrix(
            (data, offsets),
            shape=(self.de, self.de))
        self.rhs_mat = block_diag(self.nmuspec*[self.rhs_mat]).tocsr()
        self.solver = factorized(block_diag(self.nmuspec*[lhs_mat]))

    def solve_step(self, phc, dX):
        # print dX
        self._setup_solver(dX)
        phc[self.mu_selector] = self.solver(
            self.rhs_mat.dot(phc[self.mu_selector]))

    def solve_ext(self, phc):
        return self.solver(self.rhs_mat.dot(phc))


class SemiLagrangian(object):
    def __init__(self, ebins, mu_dEdX, mu_lidx_nsp, mu_selector):
        self.ebins = ebins
        self.dim_e = ebins.size - 1
        self.mu_dEdX = mu_dEdX
        self.mu_lidx, self.nmuspec = mu_lidx_nsp
        self.mu_selector = mu_selector

    def solve_step(self, state, dX):

        newbins = self.ebins + self.mu_dEdX*dX
        newgrid = np.sqrt(newbins[1:]*newbins[:-1])
        oldgrid = np.sqrt(self.ebins[1:]*self.ebins[:-1])
        lidx = self.mu_lidx
        dim_e = self.dim_e
        newgrid_log = np.log(newgrid)
        oldgrid_log = np.log(oldgrid)
        dEprime_dE = (newbins[1:] - newbins[:-1]) / (self.ebins[1:] - self.ebins[:-1])
        # print np.gradient(newgrid_log, oldgrid_log)

        for nsp in xrange(self.nmuspec):
            newstate = state[lidx + dim_e * nsp:lidx + dim_e * (nsp + 1)]/dEprime_dE
            newstate = np.where(newstate > 1e-200, newstate, 1e-200)
            newstate = np.exp(np.interp(oldgrid_log, newgrid_log, np.log(newstate)))

            state[lidx + dim_e * nsp:lidx + dim_e * (nsp + 1)] = np.where(
                newstate > 1e-200, newstate, 0.)


class DifferentialOperator(object):
    def __init__(self, ebins, mu_dEdX, mu_lidx_nsp, mu_selector):
        self.ebins = ebins
        self.egrid = np.sqrt(ebins[1:] * ebins[:-1])
        self.log_h = np.log(ebins[1] / ebins[0])
        self.dim_e = ebins.size - 1
        self.mu_selector = mu_selector
        self.mu_lidx, self.nmuspec = mu_lidx_nsp
        self.dEdX = mu_dEdX
        self.op = self.construct_differential_operator()

    def construct_differential_operator(self):
        from scipy.sparse import coo_matrix, block_diag
        # Construct a 
        # First rows of operator matrix
        diags_leftmost = [0, 1, 2, 3]
        coeffs_leftmost = [-11, 18, -9, 2]
        denom_leftmost = 6
        diags_left_1 = [-1, 0, 1, 2, 3]
        coeffs_left_1 = [-3, -10, 18, -6, 1]
        denom_left_1 = 12
        diags_left_2 = [-2, -1, 0, 1, 2, 3]
        coeffs_left_2 = [3, -30, -20, 60, -15, 2]
        denom_left_2 = 60

        # Centered diagonals
        # diags = [-3, -2, -1, 1, 2, 3]
        # coeffs = [-1, 9, -45, 45, -9, 1]
        # denom = 60.
        diags = diags_left_2
        coeffs = coeffs_left_2
        denom = 60.

        # Last rows at the right of operator matrix
        diags_right_2 = [-d for d in diags_left_2[::-1]]
        coeffs_right_2 = [-d for d in coeffs_left_2[::-1]]
        denom_right_2 = denom_left_2
        diags_right_1 = [-d for d in diags_left_1[::-1]]
        coeffs_right_1 = [-d for d in coeffs_left_1[::-1]]
        denom_right_1 = denom_left_1
        diags_rightmost = [-d for d in diags_leftmost[::-1]]
        coeffs_rightmost = [-d for d in coeffs_leftmost[::-1]]
        denom_rightmost = denom_leftmost

        h = self.log_h
        dim_e = self.dim_e
        last = dim_e - 1

        op_matrix = np.zeros((dim_e, dim_e))
        op_matrix[0, np.asarray(diags_leftmost)] = np.asarray(
            coeffs_leftmost) / (denom_leftmost * h)
        op_matrix[1, 1 + np.asarray(diags_left_1)] = np.asarray(
            coeffs_left_1) / (denom_left_1 * h)
        op_matrix[2, 2 + np.asarray(diags_left_2)] = np.asarray(
            coeffs_left_2) / (denom_left_2 * h)
        op_matrix[last,
                  last + np.asarray(diags_rightmost)] = np.asarray(
                      coeffs_rightmost) / (denom_rightmost * h)
        op_matrix[last - 1, last - 1 + np.asarray(
            diags_right_1)] = np.asarray(coeffs_right_1) / (denom_right_1 * h)
        op_matrix[last - 2, last - 2 + np.asarray(
            diags_right_2)] = np.asarray(coeffs_right_2) / (denom_right_2 * h)
        for row in range(3, dim_e - 3):
            op_matrix[row, row + np.asarray(diags)] = np.asarray(coeffs) / (
                denom * h)
        # Construct and operator by left multiplication of the back-substitution
        # dlnE to dE and right multiplication of the constant energy loss  
        single_op = coo_matrix(
            -np.diag(1 / self.egrid).dot(
                op_matrix.dot(np.diag(self.dEdX)))
        )
        return block_diag(self.nmuspec*[single_op]).tocsr()

    def solve_step(self, state, dX):
        # print 'solve_step', dX
        state[self.mu_selector] += self.op.dot(
            state[self.mu_selector])*dX
