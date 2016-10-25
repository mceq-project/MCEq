# -*- coding: utf-8 -*-
"""
:mod:`MCEq.data` --- data management
====================================

This module includes code for bookkeeping, interfacing and
validating data structures:

- :class:`InteractionYields` manages particle interactions, obtained
  from sampling of various interaction models
- :class:`DecayYields` manages particle decays, obtained from
  sampling PYTHIA8 Monte Carlo
- :class:`HadAirCrossSections` keeps information about the inelastic,
  cross-section of hadrons with air. Typically obtained from Monte Carlo.
- :class:`NCEParticle` bundles different particle properties for simpler
  usage in :class:`MCEqRun`
- :class:`EdepZFactos` calculates energy-dependent spectrum weighted
  moments (Z-Factors)

"""

import numpy as np
from mceq_config import config, dbg


class NCEParticle():

    """Bundles different particle properties for simplified
    availability of particle properties in :class:`MCEq.core.MCEqRun`.

    Args:
      pdgid (int): PDG ID of the particle
      particle_db (object): handle to an instance of :class:`ParticleDataTool.SibyllParticleTable`
      pythia_db (object): handle to an instance of :class:`ParticleDataTool.PYTHIAParticleData`
      cs_db (object): handle to an instance of :class:`InteractionYields`
      d (int): dimension of the energy grid
    """

    def __init__(self, pdgid, particle_db,
                 pythia_db, cs_db, d, max_density=1.240e-03):

        #: (float) mixing energy, transition between hadron and resonance behavior
        self.E_mix = 0
        #: (int) energy grid index, where transition between hadron and resonance occurs
        self.mix_idx = 0
        #: (float) critical energy in air at the surface
        self.E_crit = 0

        #: (bool) particle is a hadron (meson or baryon)
        self.is_hadron = False
        #: (bool) particle is a meson
        self.is_meson = False
        #: (bool) particle is a baryon
        self.is_baryon = False
        #: (bool) particle is a lepton
        self.is_lepton = False
        #: (bool) particle is an alias (PDG ID encodes special scoring behavior)
        self.is_alias = False
        #: (bool) particle has both, hadron and resonance properties
        self.is_mixed = False
        #: (bool) if particle has just resonance behavior
        self.is_resonance = False
        #: (bool) particle is interacting projectile
        self.is_projectile = False

        self.pdgid = pdgid
        self.particle_db = particle_db
        self.pythia_db = pythia_db  
        if pdgid in config["vetos"]["veto_decays"]:
            pythia_db.force_stable(self.pdgid)
        self.cs = cs_db
        self.d = d
        self.max_density = max_density

        self.E_crit = self.critical_energy()
        self.name = particle_db.pdg2modname[pdgid]

        if pdgid in particle_db.mesons:
            self.is_hadron = True
            self.is_meson = True
        elif pdgid in particle_db.baryons:
            self.is_hadron = True
            self.is_baryon = True
        else:
            self.is_lepton = True
            if abs(pdgid) > 20:
                self.is_alias = True

    def hadridx(self):
        """Returns index range where particle behaves as hadron.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (self.mix_idx, self.d)

    def residx(self):
        """Returns index range where particle behaves as resonance.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (0, self.mix_idx)

    def lidx(self):
        """Returns lower index of particle range in state vector.

        Returns:
          (int): lower index in state vector :attr:`MCEqRun.phi`
        """
        return self.nceidx * self.d

    def uidx(self):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`MCEqRun.phi`
        """
        return (self.nceidx + 1) * self.d

    def inverse_decay_length(self, E, cut = True):
        """Returns inverse decay length (or infinity (np.inf), if
        particle is stable), where the air density :math:`\\rho` is
        factorized out.

        Args:
          E (float) : energy in laboratory system in GeV
          cut (bool): set to zero in 'resonance' regime
        Returns:
          (float): :math:`\\frac{\\rho}{\\lambda_{dec}}` in 1/cm
        """
        try:
            dlen = self.pythia_db.mass(self.pdgid) / \
                self.pythia_db.ctau(self.pdgid) / E
            if cut:
                dlen[0:self.mix_idx] = 0.
            return dlen
        except ZeroDivisionError:
            return np.ones(self.d) * np.inf

    def inverse_interaction_length(self, cs=None):
        """Returns inverse interaction length in Air.

        Returns:
          (float): :math:`\\frac{1}{\\lambda_{int}}` in cm**2/g
        """

        m_air = 14.5 * 1.672621 * 1e-24  # <A> * m_proton [g]
        return np.ones(self.d) * self.cs.get_cs(self.pdgid) / m_air

    def critical_energy(self):
        """Returns critical energy where decay and interaction
        are balanced.

        Approximate value in Air.

        Returns:
          (float): :math:`\\frac{m\\ 6.4 \\text{km}}{c\\tau}` in GeV
        """
        try:
            return self.pythia_db.mass(self.pdgid) * 6.4e5 / \
                self.pythia_db.ctau(self.pdgid)
        except ZeroDivisionError:
            return np.inf

    def calculate_mixing_energy(self, e_grid, no_mix=False,
                                max_density=1.240e-03):
        """Calculates interaction/decay length in Air and decides if
        the particle has resonance and/or hadron behavior.

        Class attributes :attr:`is_mixed`, :attr:`E_mix`, :attr:`mix_idx`,
        :attr:`is_resonance` are calculated here.

        Args:
          e_grid (np.array): energy grid of size :attr:`d`
          no_mix (bool): if True, mixing is disabled and all particles
                         have either hadron or resonances behavior.
          max_density (float): maximum density on the integration path (largest
                               decay length)
        """

        cross_over = config["hybrid_crossover"]

        if abs(self.pdgid) in [2212, 2112]:
            self.mix_idx = 0
            self.is_mixed = False
            return
        d = self.d

        inv_intlen = self.inverse_interaction_length()
        inv_declen = self.inverse_decay_length(e_grid)

        if not np.any(inv_declen > 0.) or not np.any(inv_intlen > 0.):
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False
            return

        lint = np.ones(d) / inv_intlen
        d_tilde = 1 / self.inverse_decay_length(e_grid)

        # multiply with typical air density at the surface
        ldec = d_tilde * max_density

        criterium = ldec / lint

        if np.max(criterium) < cross_over:
            self.mix_idx = d - 1
            self.E_mix = e_grid[self.mix_idx]
            self.is_mixed = False
            self.is_resonance = True

        elif np.min(criterium) > cross_over or no_mix:
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False
        else:
            self.mix_idx = np.where(ldec / lint > cross_over)[0][0]
            self.E_mix = e_grid[self.mix_idx]
            self.is_mixed = True
            self.is_resonance = False

    def __repr__(self):
        a_string = (
            """
        {0}:
        is_hadron   : {1}
        is_mixed    : {2}
        is_resonance: {3}
        is_lepton   : {4}
        is_alias    : {5}
        E_mix       : {6:2.1e}\n""").format(
            self.name, self.is_hadron, self.is_mixed,
            self.is_resonance, self.is_lepton,
            self.is_alias, self.E_mix)
        return a_string

#         i(Ei0)->j(Ej0)    ...     i(EiN)->j(Ej0)
#         i(Ei0)->j(Ej1)    ...     i(EiN)->j(Ej1)
#             ...                        ...
#             ...                        ...
#             ...                        ...
#         i(Ei0)->j(EjN)   .....    i(EiN)->j(EjN)


class InteractionYields():

    """Class for managing the dictionary of interaction yield matrices.

    The class unpickles a dictionary, which contains the energy grid
    and :math:`x` spectra, sampled from hadronic interaction models.



    A list of available interaction model keys can be printed by::

        $ print yield_obj

    Args:
      interaction_model (str): name of the interaction model
      charm_model (str, optional): name of the charm model

    """

    def __init__(self, interaction_model, charm_model=None):
        #: (str) InterAction Model name
        self.iam = None
        #: (str) charm model name
        self.charm_model = None
        #: (numpy.array) energy grid bin centers
        self.e_grid = None
        #: (numpy.array) energy grid bin endges
        self.e_bins = None
        #: (numpy.array) energy grid bin widths
        self.weights = None
        #: (int) dimension of grid
        self.dim = 0
        #: (tuple) selection of a band of coeffictients (in xf)
        self.band = None
        #: (tuple) modified particle combination for error prop.
        self.mod_pprod = None 

        # If parameters are provided during object creation,
        # load the tables during object creation.
        if interaction_model != None:
            self._load(interaction_model)
        else:
            print (self.__class__.__name__ + 
                '__init__(): Loading SIBYLL 2.1 by default.')
            self._load('SIBYLL2.1')

        if charm_model and interaction_model:
            self.inject_custom_charm_model(charm_model)


    def _load(self, interaction_model):
        """Un-pickles the yields dictionary using the path specified as
        ``yield_fname`` in :mod:`mceq_config`.

        Class attributes :attr:`e_grid`, :attr:`e_bins`, :attr:`weights`,
        :attr:`dim` are set here.

        Raises:
          IOError: if file not found
        """
        import cPickle as pickle
        from os.path import join
        if dbg > 1: print 'InteractionYields::_load(): entering..'
        #Remove dashes and points in the name
        interaction_model = interaction_model.replace('-','')
        interaction_model = interaction_model.replace('.','')
        fname = join(config['data_dir'], interaction_model + '_yields.ppd')
        try:
            yield_dict = pickle.load(open(fname, 'rb'))
        except IOError:
            self._decompress(fname)
            yield_dict = pickle.load(open(fname, 'rb'))
            
        self.e_grid = yield_dict.pop('evec')
        self.e_bins = yield_dict.pop('ebins')
        self.weights = yield_dict.pop('weights')
        self.iam = yield_dict.pop('mname')

        self.projectiles = yield_dict.pop('projectiles')
        self.secondary_dict = yield_dict.pop('secondary_dict')
        self.nspec = yield_dict.pop('nspec')

        self.yields = yield_dict

        #  = np.diag(self.e_bins[1:] - self.e_bins[:-1])
        self.dim = self.e_grid.size
        self.no_interaction = np.zeros(self.dim ** 2).reshape(
            self.dim, self.dim)

        self.charm_model = None

    def _gen_index(self, yield_dict):
        """Generates index of mother-daughter relationships.

        Currently this function is called each time an interaction model
        is set. In future versions this index will be part of the pickled
        dictionary.

        Args:
          yield_dict (dict): dictionary of yields for one interaction model
        """

        if dbg > 1: print 'InteractionYields::_gen_index(): entering..'
        
        ptemp = np.unique(zip(*yield_dict.keys())[0])

        # Filter out the non numerical strings from this list
        projectiles = []
        for proj in ptemp:
            try:
                projectiles.append(int(proj))
            except:
                continue

        e_bins = yield_dict['ebins']
        weights = np.diag(e_bins[1:] - e_bins[:-1])

        secondary_dict = {}

        for projectile in projectiles:
            secondary_dict[projectile] = []
        
        # New dictionary to replace yield_dict
        new_dict = {}

        for key, mat in yield_dict.iteritems():
            try:
                proj, sec = key
            except ValueError:
                if dbg > 2:
                    print '_gen_index(): Skip additional info', key
                # Copy additional items to the new dictionary
                new_dict[key] = mat
                continue

            # exclude electrons and photons
            if np.sum(mat) > 0:# and abs(sec) not in [11, 22]:
                # print sec not in secondary_dict[proj]
                assert(sec not in secondary_dict[proj]), \
                ("InteractionYields:_gen_index()::" +
                "Error in construction of index array: {0} -> {1}".format(proj, sec))
                secondary_dict[proj].append(sec)

                # Multiply by weights (energy bin widths with matrices)
                new_dict[key] = mat.dot(weights)
            else:
                if dbg > 2:
                    print '_gen_index(): Zero yield matrix for', key 

        new_dict['projectiles'] = projectiles
        new_dict['secondary_dict'] = secondary_dict
        new_dict['nspec'] = len(projectiles)
        new_dict['weights'] = weights
        
        return new_dict        

    def _decompress(self, fname):
        """Decompresses and unpickles dictionaries stored in bz2
        format.

        Args:
          fname (str): file name

        Returns:
          content of decompressed and unpickled file.

        Raises:
          IOError: if file not found

        """
        import os
        import bz2
        import cPickle as pickle
        fcompr = os.path.splitext(fname)[0] + '.bz2'

        if not os.path.isfile(fcompr):
            raise IOError(self.__class__.__name__ + 
                '::_decompress():: File {0} not found.'.format(fcompr))

        if dbg > 1:
            print 'Decompressing', fcompr

        # Generate index of primary secondary relations and 
        # multiply with yields
        new_dict = self._gen_index(pickle.load(bz2.BZ2File(fcompr)))

        # Dump the file in uncompressed form
        if dbg > 1:
            print 'Saving to', fname
        pickle.dump(new_dict, open(fname, 'wb'), protocol=-1)


    def _init_mod_matrix(self, x_func, *args):
        """Creates modification matrix using an (x,E)-dependent function.

        :math:`x = \\frac{E_{\\rm primary}}{E_{\\rm secondary}}` is the
        fraction of secondary particle energy. ``x_func`` can be an
        arbitrary function modifying the :math:`x_\\text{lab}` distribution.
        Run this method each time you change ``x_func``, or its parameters, 
        not each time you change modified particle.
        The ``args`` are passed to the function.

        Args:
          x_func (object): reference to function
        """

        if dbg > 0:
                print (self.__class__.__name__ + 
                    'init_mod_matrix():'), x_func.__name__, args
        
        # if not config['error_propagation_mode']:
        #     raise Exception(self.__class__.__name__ + 
        #             'init_mod_matrix(): enable error ' + 
        #             'propagation mode in config and re-initialize MCEqRun.')
        
        if dbg > 1:
            print (self.__class__.__name__ + 
                'mod_pprod_matrix(): creating xmat')
        self.xmat = self.no_interaction
        for eidx in range(self.dim):
            xvec = self.e_grid[:eidx+1]/self.e_grid[eidx]
            self.xmat[:eidx+1,eidx] =xvec

        #select the relevant slice of interaction matrix
        self.modmat = x_func(self.xmat, self.e_grid, *args)
        #Set lower triangular indices to 0. (should be not necessary)
        self.modmat[np.tril_indices(self.dim)] = 0.


    def _set_mod_pprod(self, prim_pdg, sec_pdg):
        """Sets combination of projectile/secondary for error propagation.

        The production spectrum of ``sec_pdg`` in interactions of 
        ``prim_pdg`` is modified according to the function passed to 
        :func:`InteractionYields.init_mod_matrix`

        Args:
          prim_pdg (int): interacting (primary) particle PDG ID
          sec_pdg (int): secondary particle PDG ID
        """
        if dbg > 0:
            print (self.__class__.__name__ + 
                'set_mod_pprod(): will modify particle production' +
                ' matrix of {0}/{1}.').format(prim_pdg, sec_pdg)
        self.mod_pprod = (prim_pdg, sec_pdg)

    def set_interaction_model(self, interaction_model, force=False):
        """Selects an interaction model and prepares all internal variables.

        Args:
          interaction_model (str): interaction model name
          force (bool): forces reloading of data from file
        Raises:
          Exception: if invalid name specified in argument ``interaction_model``
        """

        if not force and interaction_model == self.iam:
            if dbg > 0:
                print ("InteractionYields:set_interaction_model():: Model " +
                    self.iam + " already loaded.")
            return
        else:
            self._load(interaction_model)
        if interaction_model != self.iam:
            raise Exception("InteractionYields(): No coupling matrices " +
                            "available for the selected interaction " +
                            "model: {0}.".format(interaction_model))

    def set_xf_band(self, xf_low_idx, xf_up_idx):

        xf_bins = self.e_bins / self.e_bins[-1]
        self.band = (xf_low_idx, xf_up_idx)
        if dbg > 0:
            print ('InteractionYields::set_xf_band(): limiting '
            'Feynman x range to: {0:5.2f} - {1:5.2f}').format(xf_bins[self.band[0]],
                                                              xf_bins[self.band[1]])

    def is_yield(self, projectile, daughter):
        """Checks if a non-zero yield matrix exist for ``projectile``-
        ``daughter`` combination (deprecated)

        Args:
          projectile (int): PDG ID of projectile particle
          daughter (int): PDG ID of final state daughter/secondary particle
        Returns:
          bool: ``True`` if non-zero interaction matrix exists else ``False``
        """
        if projectile in self.projectiles and \
           daughter in self.secondary_dict[projectile]:
            return True
        else:
            if dbg > 1:
                print ('InteractionYields::is_yield(): no interaction matrix ' +
                       "for {0}, {1}->{2}".format(self.iam, projectile, daughter))
            return False

        return True

    def get_y_matrix(self, projectile, daughter):
        """Returns a ``DIM x DIM`` yield matrix.

        Args:
          projectile (int): PDG ID of projectile particle
          daughter (int): PDG ID of final state daughter/secondary particle
        Returns:
          numpy.array: yield matrix

        Note:
          In the current version, the matrices have to be multiplied by the
          bin widths. In later versions they will be stored with the multiplication
          carried out.
        """
        # if dbg > 1: print 'InteractionYields::get_y_matrix(): entering..'

        # TODO: modify yields to include the bin size
        # print config['vetos']['veto_charm_pprod']
        if (config['vetos']['veto_charm_pprod'] and
           ((abs(projectile) > 400 and abs(projectile) < 500) or
            (abs(projectile) > 4000 and abs(projectile) < 5000))):
           
            if dbg > 2:
                print ('InteractionYields::get_y_matrix(): disabled particle ' +
                    'production by', projectile)
            return self.no_interaction

        # The next line creates a copy, to prevent subsequent calls to modify
        # the original matrices stored in the dictionary.
        # @debug: probably performance bottleneck during init 
        m = self.yields[(projectile, daughter)] #.dot(self.weights)
        
        # For debugging purposes or plotting xlab distributions use this line instead
        # m = np.copy(self.yields[(projectile, daughter)])
        
        if config['vetos']['veto_forward_mesons'] and abs(daughter) < 2000 \
            and (projectile,-daughter) in self.yields.keys():
            manti = self.yields[(projectile, -daughter)] #.dot(self.weights)
            ie = 50
            if dbg > 2:
                print ('InteractionYields::get_y_matrix(): sum in veto_forward_mesons', 
                    (np.sum(m[:,ie - 30:ie]) - np.sum(manti[:,ie - 30:ie])))

            if (np.sum(m[:,ie - 30:ie]) - np.sum(manti[:,ie - 30:ie])) > 0:
                if dbg > 1:
                    print ('InteractionYields::get_y_matrix(): inverting meson ' +
                    'due to leading particle veto.',daughter,'->',-daughter)
                m = manti
            else:
                if dbg > 1:
                    print ('InteractionYields::get_y_matrix(): no inversion since ' +
                    'daughter not leading', daughter)
        else:
            if dbg > 2:
                print ('InteractionYields::get_y_matrix(): no meson inversion ' +
                    'in leading particle veto.',projectile, daughter)

        if self.mod_pprod == (projectile,daughter):
            if dbg > 0: print (
                'InteractionYields::get_y_matrix(): modifying particle ' +
                'production for {0}/{1}').format(projectile, daughter)
            
            m *= self.modmat

        if not self.band:
            return m
        else:
            # set all elements except those inside selected xf band to 0

            m[np.tril_indices(self.dim, -2 - self.band[1])] = 0
            if self.band[0] < 0:
                m[np.triu_indices(self.dim, -self.band[0])] = 0
            return m

    def assign_yield_idx(self, projectile, projidx,
                         daughter, dtridx, cmat):
        """Copies a subset, defined in tuples ``projidx`` and ``dtridx`` from
        the ``yield_matrix(projectile,daughter)`` into ``cmat``

        Args:
          projectile (int): PDG ID of projectile particle
          projidx (int,int): tuple containing index range relative
                             to the projectile's energy grid
          daughter (int): PDG ID of final state daughter/secondary particle
          dtridx (int,int): tuple containing index range relative
                            to the daughters's energy grid
          cmat (numpy.array): array reference to the interaction matrix
        """
        cmat[dtridx[0]:dtridx[1], projidx[0]:projidx[1]] = \
            self.get_y_matrix(projectile, daughter)[dtridx[0]:dtridx[1],
                                                    projidx[0]:projidx[1]]

    def inject_custom_charm_model(self, model='MRS'):
        """Overwrites the charm production yields of the yield
        dictionary for the current interaction model with yields from
        a custom model.

        The function walks through all (projectile, charm_daughter)
        combinations and replaces the yield matrices with those from
        the ``model``.

        Args:
          model (str): charm model name

        Raises:
          NotImplementedError: if model string unknown.
        """

        from ParticleDataTool import SibyllParticleTable
        from charm_models import MRS_charm, WHR_charm  # @UnresolvedImport

        if model == None:
            return

        if self.charm_model and self.charm_model != model:
            # reload the yields from the main dictionary
            self.set_interaction_model(self.iam, force=True)

        sib = SibyllParticleTable()
        charm_modids = [sib.modid2pdg[modid] for modid in
                        sib.mod_ids if abs(modid) >= 59]
        del sib
        
        # Remove the charm interactions from the index
        new_index = {}
        for proj, secondaries in self.secondary_dict.iteritems():
            new_index[proj] = [idx for idx in secondaries
                               if idx not in charm_modids]
        
        self.secondary_dict = new_index

        if model == 'MRS':
            # Set charm production to zero
            cs = HadAirCrossSections(self.iam)
            mrs = MRS_charm(self.e_grid, cs)
            for proj in self.projectiles:
                for chid in charm_modids:
                    self.yields[(proj, chid)] = mrs.get_yield_matrix(
                        proj, chid).dot(self.weights)
                    # Update index
                    self.secondary_dict[proj].append(chid)

        elif model == 'WHR':

            cs_h_air = HadAirCrossSections('SIBYLL2.3')
            cs_h_p = HadAirCrossSections('SIBYLL2.3_pp')
            whr = WHR_charm(self.e_grid, cs_h_air)
            for proj in self.projectiles:
                cs_scale = np.diag(cs_h_p.get_cs(proj) / cs_h_air.get_cs(proj)) * 14.5
                for chid in charm_modids:
                    self.yields[(proj, chid)] = whr.get_yield_matrix(
                        proj, chid) * 14.5.dot(self.weights)
                    # Update index
                    self.secondary_dict[proj].append(chid)

        elif model == 'sibyll23_pl':
            cs_h_air = HadAirCrossSections('SIBYLL2.3')
            cs_h_p = HadAirCrossSections('SIBYLL2.3_pp')
            for proj in self.projectiles:
                cs_scale = np.diag(cs_h_p.get_cs(proj) / cs_h_air.get_cs(proj))
                for chid in charm_modids:
                    # rescale yields with sigma_pp/sigma_air to ensure
                    # that in a later step indeed sigma_{pp,ccbar} is taken

                    self.yields[(proj, chid)] = self.yield_dict[
                        self.iam + '_pl'][(proj, chid)].dot(
                            cs_scale).dot(self.weights) * 14.5
                    # Update index
                    self.secondary_dict[proj].append(chid)

        else:
            raise NotImplementedError('InteractionYields:inject_custom_charm_model()::' +
                                      ' Unsupported model')

        self.charm_model = model

    def __repr__(self):
        a_string = 'Possible (projectile,secondary) configurations:\n'
        for key in sorted(self.yields.keys()):
            if key not in ['evec', 'ebins']:
                a_string += str(key) + '\n'
        return a_string


class DecayYields():

    """Class for managing the dictionary of decay yield matrices.

    The class un-pickles a dictionary, which contains :math:`x`
    spectra of decay products/daughters, sampled from PYTHIA 8
    Monte Carlo.

    Args:
      weights (numpy.array): bin widths of energy grid
    """

    def __init__(self):
        self._load()

        self.particle_keys = self.mothers

    def _load(self):
        """Un-pickles the yields dictionary using the path specified as
        ``decay_fname`` in :mod:`mceq_config`.

        Raises:
          IOError: if file not found
        """
        import cPickle as pickle
        from os.path import join

        fname = join(config['data_dir'], config['decay_fname'])
        try:
            self.decay_dict = pickle.load(open(fname, 'rb'))
        except IOError:
            self._decompress(fname)
            self.decay_dict = pickle.load(open(fname, 'rb'))
        
        self.daughter_dict = self.decay_dict.pop('daughter_dict')
        self.weights = self.decay_dict.pop('weights')

        for mother in config["vetos"]["veto_decays"]:
            if dbg > 1:
                print ("DecayYields:_gen_index():: switching off " +
                    "decays of {0}.").format(mother)
            self.daughter_dict.pop(mother)

        self.mothers = self.daughter_dict.keys()

    def _gen_index(self, decay_dict):
        """Generates index of mother-daughter relationships.

        This function is called once after un-pickling. In future
        versions this index will be part of the pickled dictionary.
        """
        temp = np.unique(zip(*decay_dict.keys())[0])
        # Filter out the non numerical strings from this list
        mothers = []
        for mo in temp:
            try:
                mothers.append(int(mo))
            except:
                continue

        daughter_dict = {}
        weights = decay_dict['weights']

        # New dictionary to replace yield_dict
        new_dict = {}        

        for mother in mothers:
            daughter_dict[mother] = []

        for key, mat in decay_dict.iteritems():
            try:
                mother, daughter = key
            except ValueError:
                if dbg > 2:
                    print (self.__class__.__name__ + 
                        '_gen_index(): Skip additional info', key)
                # Copy additional items to the new dictionary
                new_dict[key] = mat
                continue

            if np.sum(mat) > 0:
                if daughter not in daughter_dict[mother]:
                    daughter_dict[mother].append(daughter)
                    # Multiply by weights (energy bin widths with matrices)
                    new_dict[key] = (mat.T).dot(weights)

        # special treatment for muons, which should decay even if they
        # have an alias ID
        # the ID 7313 not included, since it's "a copy of"
        for alias in [7013, 7113, 7213]:
            if 13 not in config["vetos"]["veto_decays"]: 
                daughter_dict[alias] = daughter_dict[13]
                for d in daughter_dict[alias]:
                    new_dict[(alias, d)] = new_dict[(13, d)]
            if -13 not in config["vetos"]["veto_decays"]:
                daughter_dict[-alias] = daughter_dict[-13]
                for d in daughter_dict[-alias]:
                    new_dict[(-alias, d)] = new_dict[(-13, d)]

        new_dict['mothers'] = mothers
        new_dict['daughter_dict'] = daughter_dict

        return new_dict

    def _decompress(self, fname):
        """Decompresses and unpickles dictionaries stored in bz2
        format.

        The method calls :func:`DecayYields._gen_index` to browse
        through the file, to create an index of mother daughter relations
        and to carry out some pre-computations. In the end an uncompressed 
        file is stored including the index as a dictionary.

        Args:
          fname (str): file name

        Returns:
          content of decompressed and unpickled file.

        Raises:
          IOError: if file not found

        """
        import os
        import bz2
        import cPickle as pickle
        fcompr = os.path.splitext(fname)[0] + '.bz2'

        if not os.path.isfile(fcompr):
            raise IOError(self.__class__.__name__ + 
                '::_decompress():: File {0} not found.'.format(fcompr))

        if dbg > 1:
            print 'Decompressing', fcompr

        # Generate index of mother daughter relations and 
        # multiply with yields
        new_dict = self._gen_index(pickle.load(bz2.BZ2File(fcompr)))

        # Dump the file in uncompressed form
        if dbg > 1:
            print 'Saving to', fname

        pickle.dump(new_dict, open(fname, 'wb'), protocol=-1)

    def get_d_matrix(self, mother, daughter):
        """Returns a ``DIM x DIM`` decay matrix.

        Args:
          mother (int): PDG ID of mother particle
          daughter (int): PDG ID of final state daughter particle
        Returns:
          numpy.array: decay matrix

        Note:
          In the current version, the matrices have to be multiplied by the
          bin widths. In later versions they will be stored with the multiplication
          carried out.
        """
        if dbg > 1 and not self.is_daughter(mother, daughter):
            print ("DecayYields:get_d_matrix():: trying to get empty matrix" +
                   "{0} -> {1}").format(mother, daughter)
        # TODO: fix structure of the decay dict
        return self.decay_dict[(mother, daughter)]

    def assign_d_idx(self, mother, moidx,
                     daughter, dtridx, dmat):
        """Copies a subset, defined in tuples ``moidx`` and ``dtridx`` from
        the ``decay_matrix(mother,daughter)`` into ``dmat``

        Args:
          mother (int): PDG ID of mother particle
          moidx (int,int): tuple containing index range relative
                             to the mothers's energy grid
          daughter (int): PDG ID of final state daughter/secondary particle
          dtridx (int,int): tuple containing index range relative
                            to the daughters's energy grid
          dmat (numpy.array): array reference to the decay matrix
        """
        dmat[dtridx[0]:dtridx[1], moidx[0]:moidx[1]] = \
            self.get_d_matrix(mother, daughter)[dtridx[0]:dtridx[1],
                                                moidx[0]:moidx[1]]

    def is_daughter(self, mother, daughter):
        """Checks if ``daughter`` is a decay daughter of ``mother``.

        Args:
          mother (int): PDG ID of projectile particle
          daughter (int): PDG ID of daughter particle
        Returns:
          bool: ``True`` if ``daughter`` is daughter of ``mother``
        """
        if (mother not in self.daughter_dict.keys() or
                daughter not in self.daughter_dict[mother]):
            return False
        else:
            return True

    def daughters(self, mother):
        """Checks if ``mother`` decays and returns the list of daughter particles.

        Args:
          mother (int): PDG ID of projectile particle
        Returns:
          list: PDG IDs of daughter particles
        """
        if mother not in self.daughter_dict.keys():
            if dbg > 2:
                print "DecayYields:daughters():: requesting daughter " + \
                    "list for stable or not existing mother: " + str(mother)
            return []
        return self.daughter_dict[mother]

    def __repr__(self):
        a_string = 'Possible (mother,daughter) configurations:\n'
        for key in sorted(self.decay_dict.keys()):
            a_string += str(key) + '\n'
        return a_string


class HadAirCrossSections():

    """Class for managing the dictionary of hadron-air cross-sections.

    The class unpickles a dictionary, which contains proton-air,
    pion-air and kaon-air cross-sections tabulated on the common
    energy grid.

    Args:
      interaction_model (str): name of the interaction model
    """
    #: unit - :math:`\text{GeV} \cdot \text{fm}`
    GeVfm = 0.19732696312541853
    #: unit - :math:`\text{GeV} \cdot \text{cm}`
    GeVcm = GeVfm * 1e-13
    #: unit - :math:`\text{GeV}^2 \cdot \text{mbarn}`
    GeV2mbarn = 10.0 * GeVfm ** 2
    #: unit conversion - :math:`\text{mbarn} \to \text{cm}^2`
    mbarn2cm2 = GeVcm ** 2 / GeV2mbarn

    def __init__(self, interaction_model):
        #: current interaction model name
        self.iam = None
        #: current energy grid
        self.egrid = None

        self._load()

        if interaction_model != None:
            self.set_interaction_model(interaction_model)
        else:
            # Set some default interaction model to allow for cross-sections
            self.set_interaction_model('SIBYLL2.3')

    def _load(self):
        """Un-pickles a dictionary using the path specified as
        ``decay_fname`` in :mod:`mceq_config`.

        Raises:
          IOError: if file not found
        """
        import cPickle as pickle
        from os.path import join
        fname = join(config['data_dir'], config['cs_fname'])
        try:
            self.cs_dict = pickle.load(open(fname,'rb'))
        except IOError:
            self._decompress(fname)
            self.cs_dict = pickle.load(open(fname,'rb'))

        self.egrid = self.cs_dict['evec']

    def _decompress(self, fname):
        """Decompresses and unpickles dictionaries stored in bz2
        format.

        Args:
          fname (str): file name

        Returns:
          content of decompressed and unpickled file.

        Raises:
          IOError: if file not found

        """
        import os
        import bz2
        import cPickle as pickle
        fcompr = os.path.splitext(fname)[0] + '.bz2'

        if not os.path.isfile(fcompr):
            raise IOError(self.__class__.__name__ + 
                '::_decompress():: File {0} not found.'.format(fcompr))

        if dbg > 1:
            print 'Decompressing', fcompr
        
        new_dict = pickle.load(bz2.BZ2File(fcompr))

        # Dump the file in uncompressed form
        if dbg > 1:
            print 'Saving to', fname
        pickle.dump(new_dict, open(fname, 'wb'), protocol=-1)

    def set_interaction_model(self, interaction_model):
        """Selects an interaction model and prepares all internal variables.

        Args:
          interaction_model (str): interaction model name
        Raises:
          Exception: if invalid name specified in argument ``interaction_model``
        """

        if interaction_model == self.iam and dbg > 0:
            print ("InteractionYields:set_interaction_model():: Model " +
                   self.iam + " already loaded.")
            return
        if interaction_model in self.cs_dict.keys():
            self.iam = interaction_model

        else:
            print "Available interaction models: ", self.cs_dict.keys()
            raise Exception("HadAirCrossSections(): No cross-sections for the desired " +
                            "interaction model {0} available.".format(interaction_model))
        self.cs = self.cs_dict[self.iam]

    def get_cs(self, projectile, mbarn=False):
        """Returns inelastic ``projectile``-air cross-section
        :math:`\\sigma_{inel}^{proj-Air}(E)` as vector spanned over
        the energy grid.

        Args:
          projectile (int): PDG ID of projectile particle
          mbarn (bool,optional): if ``True``, the units of the cross-section
                                 will be :math:`mbarn`, else :math:`\\text{cm}^2`

        Returns:
          numpy.array: cross-section in :math:`mbarn` or :math:`\\text{cm}^2`
        """

        message_templ = 'HadAirCrossSections(): replacing {0} with {1} cross-section'
        scale = 1.0
        if not mbarn:
            scale = self.mbarn2cm2
        if abs(projectile) in self.cs.keys():
            return scale * self.cs[projectile]
        elif abs(projectile) in [411, 421, 431, 15]:
            if dbg > 2:
                print message_templ.format('D', 'K+-')
            return scale * self.cs[321]
        elif abs(projectile) in [4332, 4232, 4132]:
            if dbg > 2:
                print message_templ.format('charmed baryon', 'nucleon')
            return scale * self.cs[2212]
        elif abs(projectile) == 22:
            if dbg > 2:
                print message_templ.format('photon', 'pion')
            return scale * self.cs[211]
        elif abs(projectile) > 2000 and abs(projectile) < 5000:
            if dbg > 2:
                print message_templ.format(projectile, 'nucleon')
            return scale * self.cs[2212]
        elif 10 < abs(projectile) < 17 or 7000 < abs(projectile) < 7500:
            if dbg > 2:
                print 'HadAirCrossSections(): returning 0 cross-section for lepton', projectile
            return 0.
        else:
            if dbg > 2:
                print message_templ.format(projectile, 'pion')
            return scale * self.cs[211]

    def __repr__(self):
        a_string = 'HadAirCrossSections() available for the projectiles: \n'
        for key in sorted(self.cs.keys()):
            a_string += str(key) + '\n'
        return a_string
