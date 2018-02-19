# -*- coding: utf-8 -*-
"""
:mod:`MCEq.data_utils` --- file operations on MCEq databases
============================================================

This module contains function to convert data files used by
MCEq.

- :func:`convert_to_compact` converts an interaction model file
  into "compact" mode
- :func:`extend_to_low_energies` extends an interaction model file
  with an low energy interaction model using interpolation
"""

import numpy as np
from mceq_config import config, dbg
import MCEq.data

def convert_to_compact(fname):
    """Converts an interaction model dictionary to "compact" mode.

    This function takes a compressed yield file, where all secondary
    particle types known to the particular model are expected to be
    final state particles (set stable in the MC), and converts it
    to a new compressed yield file which contains only production channels
    to the most important particles (for air-shower and inclusive lepton
    calculations).

    The production of short lived particles and resonances is taken into
    account by executing the convolution with their decay distributions into
    more stable particles, until only final state particles are left. The list
    of "important" particles is defined in the standard_particles variable below.
    This results in a feed-down corretion, for example the process (chain)
    :math:`p + A \\to \\rho + X \\to \\pi + \\pi + X` becomes simply
    :math:`p + A \\to \\pi + \\pi + X`.
    The new interaction yield file obtains the suffix `_compact` and it
    contains only those final state secondary particles:

    .. math::

        \pi^+, K^+, K^0_{S,L}, p, n, \\bar{p}, \\bar{n}, \\Lambda^0,
        \\bar{\Lambda^0}, \\eta, \\phi, \\omega, D^0, D^+, D^+_s +
        {\\rm c.c.} + {\\rm leptons}

    The compact mode has the advantage, that the production spectra stored in
    this dictionary are directly comparable to what accelerators consider as
    stable particles, defined by a minimal life-time requirement. Using the
    compact mode is recommended for most applications, which use
    :func:`MCEq.core.MCEqRun.set_mod_pprod` to modify the spectrum of secondary
    hadrons.

    For some interaction models, the performance advantage can be around 50\%.
    The precision loss is negligible at energies below 100 TeV, but can increase
    up to a few \% at higher energies where prompt leptons dominate. This is
    because also very short-lived charmed mesons and baryons with small branching
    ratios into leptons can interact with the atmosphere and lose energy before
    decay.

    For `QGSJET`, compact and normal mode are identical, since the model does not
    produce resonances or rare mesons by design.


    Args:
      fname (str): name of compressed yield (.bz2) file
    """

    import os
    import cPickle as pickle
    from bz2 import BZ2File
    import ParticleDataTool

    dpm_di = None
    if fname.endswith('_ledpm.bz2'):

        if dbg > 0:
            print "convert_to_compact(): Low energy extension requested", fname

        dpmpath = os.path.join(
            config['data_dir'],
            config['low_energy_extension']['le_model'].translate(
                None, "-.").upper() + '_yields_compact.bz2')

        if dbg > 2: 
            print "convert_to_compact(): looking for file", dpmpath

        if not os.path.isfile(dpmpath):
            convert_to_compact(dpmpath)
        try:
            dpm_di = pickle.load(BZ2File(dpmpath))
        except IOError:
            raise Exception(
                "convert_to_compact(): Error, low-energy model file expected but"
                + "not found.\n", dpmpath)

    # If file name is supplied as ppd or with compact including, modify to
    # expected format
    fn_he = fname.replace('.ppd', '.bz2').replace('_compact', '').replace(
        '_ledpm', '')
    if not os.path.isfile(fn_he):
        fn_he = os.path.join(config["data_dir"], fn_he)

    if dbg > 0: 
        print "convert_to_compact(): Attempting conversion of", fn_he

    # Load the yield dictionary (without multiplication with bin widths)
    mdi = pickle.load(BZ2File(fn_he))

    # Load the decay dictionary (with bin widths and index)
    try:
        ddi = pickle.load(
            open(os.path.join(config["data_dir"], config["decay_fname"]), 'rb'))
    except IOError:
        # In case the ppd file is not yet created, use the DecayYields class to
        # decompress, rotate and weight the yield files
        from MCEq.data import DecayYields
        ds = DecayYields(fname=config["decay_fname"])
        ddi = ds.decay_dict

    # Define a list of "stable" particles
    # Particles having an anti-partner
    standard_particles = [
        22, 11, 12, 13, 7113, 7213, 7313,
        14, 15, 16, 211, 321, 2212, 2112, 3122, 411, 421, 431
    ]  #for EM cascade add 11, 22
    standard_particles += [-pid for pid in standard_particles]

    # unflavored particles
    # append 221, 223, 333, if eta, omega and phi needed directly
    standard_particles += [130, 310]  #, 221, 223, 333]

    # projectiles
    allowed_projectiles = [2212, 2112, 211, 321, 130, 3122]

    import ParticleDataTool as pd
    part_d = pd.PYTHIAParticleData()
    ctau_pr = part_d.ctau(310)

    # Create new dictionary for the compact model version
    compact_di = {}

    def create_secondary_dict(yield_dict):
        """This is a replica of function
        :func:`MCEq.data.InteractionYields._gen_index`."""
        dbgstr = 'convert_to_compact::create_secondary_dict(): '
        if dbg > 2: 
            print dbgstr + 'entering...'

        secondary_dict = {}
        for key, mat in sorted(yield_dict.iteritems()):
            try:
                proj, sec = key
            except ValueError:
                if dbg > 3:
                    print(dbgstr + 'Skip additional info', key)
                continue

            if proj not in secondary_dict:
                secondary_dict[proj] = []
                if dbg > 3: 
                    print dbgstr, proj, 'added.'

            if np.sum(mat) > 0:
                assert (sec not in secondary_dict[proj]), (
                    dbgstr +
                    "Error in construction of index array: {0} -> {1}".format(
                        proj, sec))
                secondary_dict[proj].append(sec)
            else:
                if dbg > 3:
                    print dbgstr + 'Zeros for', proj, sec

        return secondary_dict

    def follow_chained_decay(real_mother, mat, interm_mothers, reclev):
        """Follows the chain of decays down to the list of stable particles.

        The function follows the list of daughters of an unstable parrticle
        and computes the feed-down contribution by evaluating the convolution
        of production spectrum and decay spectrum using the formula (24)

        ..math::
            \mathbf{C}^M^p = D^\rho \cdot \ 
        :func:`MCEq.data.InteractionYields._gen_index`."""

        dbgstr = 'convert_to_compact::follow_chained_decay(): '
        tab = 3 * reclev * '--' + '> '

        if dbg > 5 and reclev == 0:
            print dbgstr, 'start recursion with', real_mother, interm_mothers, np.sum(
                mat)
        elif dbg > 10:
            print tab, 'enter with', real_mother, interm_mothers, np.sum(mat)

        if np.sum(mat) < 1e-30:
            if dbg > 10:
                print tab, 'zero matrix for', real_mother, interm_mothers
        if interm_mothers[-1] not in dec_di or interm_mothers[-1] in standard_particles:
            if dbg > 10:
                print tab, 'no further decays of', interm_mothers
            return

        for d in dec_di[interm_mothers[-1]]:
            # Decay matrix
            dmat = ddi[(interm_mothers[-1], d)]
            # Matrix product D x C from the left (convolution)
            mprod = dmat.dot(mat)

            if np.sum(mprod) < 1e-40:
                if dbg > 10:
                    print tab, 'cancel recursion in', real_mother, interm_mothers, d, \
                    'since matrix is zero', np.sum(mat), np.sum(dmat), np.sum(mprod)
                continue

            if d not in standard_particles:
                if dbg > 5:
                    print tab, 'Recurse', real_mother, interm_mothers, d, np.sum(
                        mat)

                follow_chained_decay(real_mother, mprod, interm_mothers + [d],
                                     reclev + 1)
            else:
                # Track prompt leptons in prompt category
                if abs(d) in [12, 13, 14, 16]:
                    # is_prompt = bool(np.sum([(part_d.ctau(mo) <= ctau_pr or
                    #     4000 < abs(mo) < 7000 or 400 < abs(mo) < 500)
                    #     for mo in interm_mothers]))
                    is_prompt = sum(
                        [part_d.ctau(mo) <= ctau_pr
                         for mo in interm_mothers]) > 0
                    if is_prompt:
                        d = np.sign(d) * (7000 + abs(d))

                if dbg > 5:
                    print tab, 'contribute to', real_mother, interm_mothers, d

                if (real_mother, d) in compact_di.keys():
                    if dbg > 10:
                        print tab, '+=', (real_mother, d), np.sum(mprod)
                    compact_di[(real_mother, d)] += mprod
                else:
                    if dbg > 10:
                        print tab, 'new', (real_mother, d), interm_mothers
                    compact_di[(real_mother, d)] = mprod

        return

    # Create index of entries
    pprod_di = create_secondary_dict(mdi)
    dec_di = create_secondary_dict(ddi)

    if dbg > 5:
        print 'Int   dict:\n', sorted(pprod_di)
        print 'Decay dict:\n', sorted(dec_di)

    for proj, lsecs in sorted(pprod_di.iteritems()):

        if abs(proj) not in allowed_projectiles:
            continue

        for sec in lsecs:
            # Copy all direct production in first iteration
            if sec in standard_particles:
                compact_di[(proj, sec)] = np.copy(mdi[(proj, sec)])

                if dbg > 2: print 'copied', proj, '->', sec

        for sec in lsecs:
            #Iterate over all remaining secondaries
            if sec in standard_particles:
                continue

            if dbg > 3: proj, '->', sec, '->', dec_di[sec]
            #Enter recursion and calculate contribution from decay
            follow_chained_decay(proj, mdi[(proj, sec)], [sec], 0)

    # Copy metadata
    compact_di['ebins'] = np.copy(mdi['ebins'])
    compact_di['evec'] = np.copy(mdi['evec'])
    compact_di['mname'] = mdi['mname']

    if dpm_di:
        compact_di = extend_to_low_energies(compact_di, dpm_di)

    pickle.dump(compact_di, BZ2File(fname, 'wb'), protocol=-1)

    # Delete cached version if it exists
    if os.path.isfile(fname.replace('.bz2', '.ppd')):
        os.unlink(fname.replace('.bz2', '.ppd'))


def extend_to_low_energies(he_di=None, le_di=None, fname=None):
    """Interpolates between a high-energy and a low-energy interaction model.

    Theis function takes either two yield dictionaries or a file name
    of the high energy model and interpolates the matrices at the energy
    specified in `:mod:mceq_config` in the low_energy_extension section.
    The interpolation is linear in energy grid index.

    In 'compact' mode all particles should be supported by the low energy
    model. However if you don't use compact mode, some rare or exotic
    secondaries might be not supported by the low energy model. In this
    case the config option "use_unknown_cs" decides if only the high energy
    part is used or if to raise an excption.

    Args:
      he_di (dict,optional): yield dictionary of high-energy model
      le_di (dict,optional): yield dictionary of low-energy model
      fname (str,optional): file name of high-energy model yields
    """

    import cPickle as pickle
    from bz2 import BZ2File
    import os

    if (he_di and le_di) and fname:
        raise Exception(
            "extend_to_low_energies(): either dictionaries or a file name " +
            "should be specified, but not both.")

    if fname:
        if dbg > 0:
            print "extend_to_low_energies(): Low energy extension requested:", fname

        # Load the yield dictionary (without multiplication with bin widths ".bz2")
        he_di = pickle.load(
            BZ2File(fname.replace('_ledpm', '').replace('.ppd', '.bz2')))

        # Load low energy model yields
        le_di = pickle.load(
            BZ2File(
                os.path.join(config['data_dir'], config['low_energy_extension']
                             ['le_model'].translate(
                                 None, "-.").upper() + '_yields.bz2')))

    he_le_trasition = config['low_energy_extension']['he_le_transition']
    nbins_interp = config['low_energy_extension']['nbins_interp']

    egr = he_di['evec']  # will throw error

    # Find the index of transition in the energy grid
    transition_idx = np.count_nonzero(egr < he_le_trasition)
    if dbg > 1:
        print "extend_to_low_energies(): transition_idx={0}, transition_energy={1}".format(
            transition_idx, egr[transition_idx])

    # Indices of the transition region (+2 because 0 and 1 are included)
    intp_indices = np.arange(
        transition_idx - nbins_interp / 2 - 1,
        transition_idx + nbins_interp / 2 + 1.1,
        1,
        dtype='int32')
    intp_scales = np.linspace(0, 1, len(intp_indices))
    intp_array = np.ones((len(egr), len(intp_scales)))
    he_int_array = intp_scales * intp_array
    le_int_array = intp_scales[::-1] * intp_array
    if dbg > 2:
        print "extend_to_low_energies(): int. arrays", intp_scales, \
            intp_indices, egr[intp_indices]

    ext_di = {}

    for k in he_di.keys():
        if type(k) is not tuple:
            ext_di[k] = he_di[k]
            continue

        he_mat = np.copy(he_di[k])

        new_mat = he_mat

        if k not in le_di:
            # Use only he model cross sections if le model doesn't
            # know the process
            if config["low_energy_extension"]["use_unknown_cs"]:
                if dbg > 3:
                    print "extend_to_low_energies(): skipping particle", k
                ext_di[k] = new_mat
                continue
            else:
                raise Exception('extend_to_low_energies(): High energy model' +
                                ' contains ')
        else:
            le_mat = np.copy(le_di[k])
            try:
                new_mat[:, :intp_indices[0]] *= 0.
                new_mat[:, intp_indices] *= he_int_array
                le_mat[:, intp_indices[-1]:] *= 0
                le_mat[:, intp_indices] *= le_int_array

            except IndexError:
                print "extend_to_low_energies(): problems indexing model transition"
                print k, intp_indices,
                print he_mat
                print le_mat

            new_mat += le_mat

            ext_di[k] = new_mat

    ext_di['le_ext'] = config["low_energy_extension"]

    if fname:
        if dbg > 0: 
            print "extend_to_low_energies(): Saving", fname
        pickle.dump(ext_di, BZ2File(fname, 'wb'), protocol=-1)

    return ext_di


class LogSpacedInteractionYields(MCEq.data.InteractionYields):
    """This derived type puts MCEq on a log spaced grid"""
    
    def _load(self, interaction_model):
        """Substitute  dN/lnE during initialization"""
        import cPickle as pickle
        from os.path import join, isfile
        from MCEq.misc import normalize_hadronic_model_name
        from MCEq.data_utils import convert_to_compact, extend_to_low_energies
        if dbg > 1:
            print 'InteractionYields::_load(): entering..'

        # Remove dashes and points in the name
        iamstr = normalize_hadronic_model_name(interaction_model)

        fname = join(config['data_dir'], iamstr + '_yields.bz2')

        if config['compact_mode'] and config["low_energy_extension"]["enabled"] \
                and 'DPMJET' not in iamstr:
            fname = fname.replace('.bz2', '_compact_ledpm.bz2')
        elif not config['compact_mode'] and config["low_energy_extension"]["enabled"] \
                and ('DPMJET' not in iamstr):
            fname = fname.replace('.bz2', '_ledpm.bz2')
        elif config['compact_mode']:
            fname = fname.replace('.bz2', '_compact.bz2')

        yield_dict = None
        if dbg > 0:
            print 'InteractionYields::_load(): Looking for', fname
        if not isfile(fname):
            if config['compact_mode']:
                convert_to_compact(fname)
            elif 'ledpm' in fname:
                extend_to_low_energies(fname=fname)
            else:
                raise Exception(
                    'InteractionYields::_load(): no model file found for' +
                    interaction_model)

        if not isfile(fname.replace('.bz2', '.ppd')):
            self._decompress(fname)

        yield_dict = pickle.load(open(fname.replace('.bz2', '.ppd'), 'rb'))

        self.e_grid = yield_dict.pop('evec')
        self.e_bins = yield_dict.pop('ebins')
        self.weights = yield_dict.pop('weights')
        self.iam = normalize_hadronic_model_name(yield_dict.pop('mname'))
        self.projectiles = yield_dict.pop('projectiles')
        self.secondary_dict = yield_dict.pop('secondary_dict')
        self.nspec = yield_dict.pop('nspec')

        self.yields = yield_dict

        #  = np.diag(self.e_bins[1:] - self.e_bins[:-1])
        self.dim = self.e_grid.size
        self.no_interaction = np.zeros(self.dim**2).reshape(self.dim, self.dim)

        self.charm_model = None

        self._gen_particle_list()
