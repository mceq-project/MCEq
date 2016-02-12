'''
:mod:`ParticleDataTool` --- collection of classes dealing with particle properties and indices
==============================================================================================

This tool gives handy Python access to the particle database from the 
`PYTHIA 8 <http://home.thep.lu.se/~torbjorn/pythia81html/Welcome.html>`_ Monte Carlo.

The general convention of my modules is to use Particle Data Group 
`PDG <http://pdg.lbl.gov>`_ particle naming/ID convention. Various interaction models
use, however, proprietary index schemes. The classes :class:`SibyllParticleTable`, 
:class:`QGSJetParticleTable` and :class:`DpmJetParticleTable` provide conversion
routines from these proprietary IDs/names into PDG indices, which in turn can
be used in the :class:`PYTHIAParticleData` to obtain properties such as
mass :func:`PYTHIAParticleData.mass`, life-time :func:`PYTHIAParticleData.ctau`,
etc.

Example:
    enter in a Python shell::
    
      $ import ParticleDataTool as pd
      $ pd.test() 
'''

from abc import ABCMeta

#===============================================================================
# PYTHIAParticleData
#===============================================================================
class PYTHIAParticleData():
    """Class wraps around the original ParticleData.xml from PYTHIA 8.
     
    It operates on in memory data after parsing the XML file or after reading
    a pickled PYTHON representation of this parsed XML.
    
    """
    def __init__(self, file_path='ParticleData.ppl', use_cache=True):
        import cPickle as pickle
        try:
            self.pytname2data, self.pdg_id2data = pickle.load(open('ParticleData.ppl', 'rb'))
        except:
            self._load_xml(file_path, use_cache)
        
        # : name aliases for backward compatibility
        self.str_alias_table = \
        {'K0L':'K_L0', 'K0S':'K_S0', 'Lambda':'Lambda0',
         'eta*':"eta'", 'etaC':'eta_c',
         'D*+':'D*_0+', 'D*-':'D*_0-', 'D*0':'D*_00',
         'Ds+':'D_s+', 'Ds-':'D_s-', 'Ds*+':'D*_0s+', 'Ds*-':'D*_0s-',
         'SigmaC++':'Sigma_c++', 'SigmaC+':'Sigma_c+', 'SigmaC0':'Sigma_c0',
         'SigmaC--':'Sigma_cbar--', 'SigmaC-':'Sigma_cbar-',
         'SigmaC*++':'Sigma*_c++', 'SigmaC*+':'Sigma*_c+', 'SigmaC*0':'Sigma*_c0',
         'SigmaC--':'Sigma_c--', 'SigmaC-':'Sigma_c-'}
    
    def _load_xml(self, file_path, use_cache):
        """Reads the xml and pics out particle data only. If no decay length
        is given, it will calculated from the width."""
         
        import xml.etree.ElementTree as ET
        import os
        import numpy as np
        xmlname = None
        base = os.path.dirname(os.path.abspath(__file__))
        searchpaths = [base + '/ParticleData.xml',
                       'ParticleData.xml','../ParticleData.xml',
                       'ParticleDataTool/ParticleData.xml']
        for p in searchpaths:
            if os.path.isfile(p):
                xmlname = p
                break
        if xmlname == None:
            raise Exception('ParticleDataTool::_load_xml(): ' +
                'XML file not found.')
        root = ET.parse(xmlname).getroot()
        self.pytname2data = {}
        self.pdg_id2data = {}
        GeVfm = 0.19732696312541853
        for child in root:
            if child.tag == 'particle':
                m0 = float(child.attrib['m0'])
                charge = int(child.attrib['chargeType']) / 3
                ctau = 0.
                if 'tau0' in child.attrib:
                    ctau = 0.1 * float(child.attrib['tau0'])
                elif 'mWidth' in child.attrib:
                    mWidth = float(child.attrib['mWidth'])
                    ctau = GeVfm / (mWidth) * 1e-15 * 100.0  # in cm
                elif child.attrib['id'] in ['4314', '4324', '311', '433']:
                    ctau = 0.0
                elif child.attrib['id'] in ["2212", "22", "11", "12", "14", "16"]:
                    ctau = np.inf
                else:
                    continue
                pdgid = int(child.attrib['id'])
                self.pytname2data[child.attrib['name']] = (m0, ctau, pdgid, charge)
                self.pdg_id2data[pdgid] = (m0, ctau, child.attrib['name'], charge)
                try:
                    self.pytname2data[child.attrib['antiName']] = (m0, ctau, -pdgid, -charge)
                    self.pdg_id2data[-pdgid] = (m0, ctau, child.attrib['antiName'], -charge)
                except:
                    pass
                
        self.extend_tables()
        if not use_cache:
            return
        
        import cPickle as pickle
        pickle.dump((self.pytname2data, self.pdg_id2data),
                    open(file_path, 'wb'), protocol=-1)
                
    def extend_tables(self):
        """Inserts aliases for MCEq.
        """
        # 70XX prompt leptons
        self.pdg_id2data[7012] = self.pdg_id2data[12]
        self.pdg_id2data[7013] = self.pdg_id2data[13]
        self.pdg_id2data[7014] = self.pdg_id2data[14]
        self.pdg_id2data[7016] = self.pdg_id2data[16]
        self.pdg_id2data[-7012] = self.pdg_id2data[-12]
        self.pdg_id2data[-7013] = self.pdg_id2data[-13]
        self.pdg_id2data[-7014] = self.pdg_id2data[-14]
        self.pdg_id2data[-7016] = self.pdg_id2data[-16]
        
        # 71XX leptons from pion decay
        self.pdg_id2data[7112] = self.pdg_id2data[12]
        self.pdg_id2data[7113] = self.pdg_id2data[13]
        self.pdg_id2data[7114] = self.pdg_id2data[14]
        self.pdg_id2data[7116] = self.pdg_id2data[16]
        self.pdg_id2data[-7112] = self.pdg_id2data[-12]
        self.pdg_id2data[-7113] = self.pdg_id2data[-13]
        self.pdg_id2data[-7114] = self.pdg_id2data[-14]
        self.pdg_id2data[-7116] = self.pdg_id2data[-16]
        
        # 72XX leptons from kaon decay
        self.pdg_id2data[7212] = self.pdg_id2data[12]
        self.pdg_id2data[7213] = self.pdg_id2data[13]
        self.pdg_id2data[7214] = self.pdg_id2data[14]
        self.pdg_id2data[7216] = self.pdg_id2data[16]
        self.pdg_id2data[-7212] = self.pdg_id2data[-12]
        self.pdg_id2data[-7213] = self.pdg_id2data[-13]
        self.pdg_id2data[-7214] = self.pdg_id2data[-14]
        self.pdg_id2data[-7216] = self.pdg_id2data[-16]
        
        # 73XX multi-purpose category 
        self.pdg_id2data[7312] = self.pdg_id2data[12]
        self.pdg_id2data[7313] = self.pdg_id2data[13]
        self.pdg_id2data[7314] = self.pdg_id2data[14]
        self.pdg_id2data[7316] = self.pdg_id2data[16]
        self.pdg_id2data[-7312] = self.pdg_id2data[-12]
        self.pdg_id2data[-7313] = self.pdg_id2data[-13]
        self.pdg_id2data[-7314] = self.pdg_id2data[-14]
        self.pdg_id2data[-7316] = self.pdg_id2data[-16]
    
    def pdg_id(self, str_id):
        """Returns PDG particle ID.
        
        Args:
          str_id (str): PYTHIA style name of particle
        
        Returns:
          (int): PDG ID
        """
        if str_id in self.str_alias_table:
                str_id = self.str_alias_table[str_id]
         
        return int(self.pytname2data[str_id][2])
    
    def mass(self, pdg_id):
        """Returns particle mass in GeV. The mass is calculated from
        the width if not given in the XML table.
        
        Args:
          pdg_id (int): particle PDG ID
        
        Returns:
          (float): mass in GeV
        """
        
        try:
            return float(self.pdg_id2data[pdg_id][0])
        except:
            if pdg_id in self.str_alias_table:
                pdg_id = self.str_alias_table[pdg_id]
            return float(self.pytname2data[pdg_id][0])
    
    def ctau(self, pdg_id):
        """Returns decay length in cm.
        
        Args:
          pdg_id (int): particle PDG ID
        
        Returns:
          (float): decay length :math:`ctau` in cm
        """
        
        try:
            return float(self.pdg_id2data[pdg_id][1])
        except:
            if pdg_id in self.str_alias_table:
                pdg_id = self.str_alias_table[pdg_id]
            return float(self.pytname2data[pdg_id][1])
    
    def name(self, pdg_id):
        """Returns PYTHIA particle name.

        Args:
          pdg_id (int): particle PDG ID
        
        Returns:
          (str): particle name string
        """
        
        try:
            return self.pdg_id2data[pdg_id][2]
        except:
            if pdg_id in self.str_alias_table:
                pdg_id = self.str_alias_table[pdg_id]
            return self.pytname2data[pdg_id][2]
    
    def charge(self, pdg_id):
        """Returns charge.
        
        Args:
          pdg_id (int): particle PDG ID
        
        Returns:
          (float): charge
        """
        
        try:
            return float(self.pdg_id2data[pdg_id][3])
        except:
            print "Exception:", pdg_id
            if pdg_id in self.str_alias_table:
                pdg_id = self.str_alias_table[pdg_id]
            return float(self.pytname2data[pdg_id][3])
    
class InteractionModelParticleTable():
    """This abstract class provides conversions from interaction model 
    specific particle IDs/names to PDG IDs and vice versa.
    
    Interaction model specifics can be added by deriving from this class
    like it is done in :class:`SibyllParticleTable`, 
    :class:`QGSJetParticleTable` and :class:`DpmJetParticleTable`. 
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        # : converts model ID to PDG ID
        self.modid2pdg = {}
        # : converts PDG ID to model ID 
        self.pdg2modid = {}
        # : converts model specific name to model ID
        self.modname2modid = {}
        # : converts model specific name to PDG ID
        self.modname2pdg = {}
        # : converts PDG ID to model specific name
        self.pdg2modname = {}
        # : converts model ID to model specific name
        self.modid2modname = {}
        # : list of allowed model IDs
        self.mod_ids = []
        # : list of allowed PDG IDs
        self.pdg_ids = []
        # : stores the list of meson PDG IDs
        self.mesons = []
        # : stores the list of baryon PDG IDs
        self.baryons = []
        
        try:
            part_table = self.part_table
        except:
            raise Exception(self.__class__.__name__ + 
                            '(): Error particle table not defined.')

        # Fill mapping dictionaries
        for modname, pids in part_table.iteritems():
            mod_id, pdg_id = pids
            self.modid2pdg[mod_id] = pdg_id
            self.pdg2modid[pdg_id] = mod_id
            self.pdg2modname[pdg_id] = modname
            self.modid2modname[mod_id] = modname
            self.modname2modid[modname] = mod_id
            self.modname2pdg[modname] = pdg_id
            self.mod_ids.append(mod_id)
            self.pdg_ids.append(pdg_id)
        
        self.mod_ids.sort()
        self.pdg_ids.sort()
        
        # Check for consistency and duplicates 
        assert(len(self.mod_ids) == len(set(self.mod_ids))), \
            "InteractionModelParticleTable error 1."
        
        assert(len(self.pdg_ids) == len(set(self.pdg_ids)) 
               == len(self.mod_ids)), "InteractionModelParticleTable error 2."
        
        assert(len(self.modname2pdg.keys()) == len(set(self.modname2pdg.keys()))
               == len(self.mod_ids)), "InteractionModelParticleTable error 3."

        # Add index extensions/aliases for leptons
        self.extend_tables()
        
        self.mesons = [m for m in self.get_list_of_mesons(use_pdg=True) 
                       if not (10 < abs(m) < 15 or abs(m) == 16 or abs(m) == 22
                       or m in self.leptons)] 
        self.baryons = self.get_list_of_baryons(use_pdg=True)
        
        
    def extend_tables(self):
        """Extends the tables with additional aliases for
        the MCEq program.
        
        The method is called in :func:`__init__` nad theres no need
        for subsequent calls. Additional categories have to be 
        added here first, prior modifying MCEq.
        """
        leptons = [('nue', 12),
                   ('mu-', 13),
                   ('numu', 14),
                   ('nutau', 16)]
        
        antileptons = [('antinue', -12),
                       ('mu+', -13),
                       ('antinumu', -14),
                       ('antinutau', -16)]
        
        aliases = [('', 0),  # standard
                   ('pr_', 7000),  # prompt
                   ('pi_', 7100),  # from pion decay
                   ('k_', 7200),  # from kaon decay
                   ('obs_', 7300)]  # multi-purpose
        
        al_idxs, antial_idxs = [], []
        
        for prefix, idxadd in aliases:
            for lepname, lepidx in leptons:
                self.pdg2modname[idxadd + lepidx] = prefix + lepname
                self.modname2pdg[prefix + lepname] = idxadd + lepidx
                al_idxs.append(idxadd + lepidx)
            for lepname, lepidx in antileptons:
                self.pdg2modname[-idxadd + lepidx] = prefix + lepname
                self.modname2pdg[prefix + lepname] = -idxadd + lepidx
                antial_idxs.append(-idxadd + lepidx)
        
        self.leptons = al_idxs + antial_idxs
    
    def get_list_of_mesons(self, use_pdg=False):
        """Returns list of meson names or PDG IDs.
        
        Args:
          use_pdg (bool, optional): If True, PDG IDs are return
                                    otherwise particle names
        Returns:
          list: list of meson names or PDG IDs
        """
        if not use_pdg:
            return [self.modid2modname[pid] for pid in self.meson_range]
        else:
            return [self.modid2pdg[pid] for pid in self.meson_range]
    
    def get_list_of_baryons(self, use_pdg=False):
        """Returns list of baryon names or PDG IDs.
        
        Args:
          use_pdg (bool, optional): If True, PDG IDs are return
                                    otherwise particle names
        Returns:
          list: list of baryon names or PDG IDs
        """
        if not use_pdg:
            return [self.modid2modname[pid] for pid in self.baryon_range]
        else:
            return [self.modid2pdg[pid] for pid in self.baryon_range]
 
class SibyllParticleTable(InteractionModelParticleTable):
    """This derived class provides conversions from SIBYLL particle 
    IDs/names to PDG IDs and vice versa.
    
    The table part_table is written by hand from the manual of SIBYLL 2.3.
    """
    def __init__(self):
        self.part_table = {'gamma':(1, 22), 'e+':(2, -11), 'e-':(3, 11),
                    'mu+':(4, -13), 'mu-':(5, 13), 'pi0':(6, 111),
                    'pi+':(7, 211), 'pi-':(8, -211), 'K+':(9, 321), 'K-':(10, -321),
                    'K0L':(11, 130), 'K0S':(12, 310), 'p':(13, 2212), 'n':(14, 2112),
                    'nue':(15, 12), 'antinue':(16, -12), 'numu':(17, 14), 'antinumu':(18, -14),
                    'nutau':(19, 16), 'antinutau':(20, -16), 'tau+':(21, -15), 'tau-':(22, 15),
                    'eta':(23, 221), 'eta*':(24, 331),
                    'rho+':(25, 213), 'rho-':(26, -213), 'rho0':(27, 113),
                    'K*+':(28, 323), 'K*-':(29, -323), 'K*0':(30, 313), 'K*0-bar':(31, -313),
                    'omega':(32, 223), 'phi':(33, 333), 'Sigma+':(34, 3222), 'Sigma0':(35, 3212),
                    'Sigma-':(36, 3112), 'Xi0':(37, 3322), 'Xi-':(38, 3312), 'Lambda0':(39, 3122),
                    'Delta++':(40, 2224), 'Delta+':(41, 2214), 'Delta0':(42, 2114), 'Delta-':(43, 1114),
                    'Sigma*+':(44, 3224), 'Sigma*0':(45, 3214), 'Sigma*-':(46, 3114),
                    'Xi*0':(47, 3324), 'Xi*-':(48, 3314), 'Omega-':(49, 3334),
                    'D+':(59, 411), 'D-':(60, -411), 'D0':(71, 421), 'D0-bar':(72, -421), 'etaC':(73, 441),
                    'Ds+':(74, 431), 'Ds-':(75, -431), 'Ds*+':(76, 433), 'Ds*-':(77, -433),
                    'D*+':(78, 413), 'D*-':(79, -413), 'D*0':(80, 10421), 'D*0-bar':(81, -10421),
                     'jpsi':(83, 443), 'SigmaC++':(84, 4222), 'SigmaC+':(85, 4212), 'SigmaC0':(86, 4112),
                     'XiC+':(87, 4232), 'XiC0':(88, 4132), 'LambdaC+':(89, 4122),
                    'SigmaC*++':(94, 4224), 'SigmaC*+':(95, 4214), 'SigmaC*0':(96, 4114),
                    'XiC*+':(97, 4324), 'XiC*0':(98, 4314), 'OmegaC0':(99, 4332)}
        
        
        self.baryon_range = []
        temp_dict = {}
        for name, (modid, pdgid) in self.part_table.iteritems():
            if (abs(pdgid) > 1000) and (abs(pdgid) < 7000):
                temp_dict[name + '-bar'] = (-modid, -pdgid)
                self.baryon_range.append(modid)
                self.baryon_range.append(-modid)
        self.baryon_range.sort()
        self.part_table.update(temp_dict)
        
        self.meson_range = []
        # Force tau leptons into the meson group, since the tau lepton has
        # similar behavior to mesons in current applications of this module
         
        for name, (modid, pdgid) in self.part_table.iteritems():
            if (modid not in self.baryon_range and (abs(pdgid) > 100 
                                                    or abs(pdgid) == 15)):
                self.meson_range.append(modid)
        self.meson_range.sort()
        
        InteractionModelParticleTable.__init__(self)

class QGSJetParticleTable(InteractionModelParticleTable):
    """This derived class provides conversions from QGSJET particle 
    IDs/names to PDG IDs and vice versa.
    
    The table part_table is written by hand based on the source code 
    documentation of QGSJET-II-04. This class also converts indices of
    earlier versions down to QGSJET01c.
    
    For compatibility reasons the particle charge is stored in the
    :attr:`charge_tab` dictionary and can be accessed with model ids.
    """
    
    # : dictionary provides lookup of particle charge from model IDs
    charge_tab = {}
    
    def __init__(self):
        self.part_table = {
           'pi0':(0, 111), 'pi+':(1, 211), 'pi-':(-1, -211), 'p':(2, 2212),
           'p-bar':(-2, -2212), 'n':(3, 2112), 'n-bar':(-3, -2112),
           'K+':(4, 321), 'K-':(-4, -321), 'K0S':(5, 310), 'K0L':(-5, 130),
           'Lambda0':(6, 3122), 'Lambda0-bar':(-6, -3122), 'D+':(7, 411),
           'D-':(-7, -411), 'D0':(8, 421), 'D0-bar':(-8, -421),
           'LambdaC+':(9, 4122), 'LambdaC+-bar':(-9, -4122),
           'eta':(10, 221), 'rho0':(-10, 113)}

        pytab = PYTHIAParticleData()
        self.baryon_range = []
        temp_dict = {}
        for (modid, pdgid) in self.part_table.itervalues():
            self.charge_tab[modid] = pytab.charge(pdgid)
            if (abs(pdgid) > 1000) and (abs(pdgid) < 7000):
                self.baryon_range.append(modid)
        self.baryon_range.sort()
        self.part_table.update(temp_dict)
        
        self.meson_range = []
        for (modid, pdgid) in self.part_table.itervalues():
            if (modid not in self.baryon_range and abs(pdgid) > 100):
                self.meson_range.append(modid)
        self.meson_range.sort()
        
        InteractionModelParticleTable.__init__(self)

#===============================================================================
# QGSJetIIParticleTable
#===============================================================================
class DpmJetParticleTable(SibyllParticleTable):
    """This derived class provides conversions from DPMJET-III particle 
    IDs/names to PDG IDs and vice versa and derives from 
    :class:`SibyllParticleTable`.
    
    In principle DPMJET uses the PDG indices. However, the PDG table
    provides information about special or hypothtical particles which are not
    important for this research. Therefore, the DPMJET tables are contain the
    same particles as SIBYLL which can be clearly interpreted as meson, baryon
    or lepton.
    """
    def __init__(self):
        SibyllParticleTable.__init__(self)
        self.modid2modname = self.pdg2modname
        self.mod_ids = [self.modid2pdg[sid] for sid in self.mod_ids]
        self.modid2pdg = {}
        for mod_id in self.mod_ids:
            self.modid2pdg[mod_id] = mod_id


def print_stable(life_time_greater_then=1e-10):
    """Prints a list of particles with a lifetime longer than
    specified argument value in s."""
    pyth_data = PYTHIAParticleData()
    
    print '\nKnown particles which lifetimes longer than {0:1.0e} s:\n'.format(
                                                        life_time_greater_then)
    print '{0:20s}  {1:10s}  {2:8s}'.format('Name', 'ctau [cm]', 'PDG ID')
    templ = '{0:20s} {1:10.3g} {2:8}'
    for pname in pyth_data.pytname2data.iterkeys():
        if pyth_data.ctau(pname) >= life_time_greater_then * 2.99e10 and \
            pyth_data.pdg_id(pname) > 0:  # and pyth_data.ctau(pname) < 1e10:
            print templ.format(pname, pyth_data.ctau(pname),
                               pyth_data.pdg_id(pname))    

def make_stable_list(life_time_greater_then):
    """Returns a list of particles PDG IDs with a lifetime longer than
    specified argument value in s. Stable particles, such as photons,
    neutrinos, nucleons and electrons are not included."""
    
    pyth_data = PYTHIAParticleData()
    particle_list = []
    
    for pname in pyth_data.pytname2data.iterkeys():
        if pyth_data.ctau(pname) >= life_time_greater_then * 2.99e10 and \
          pyth_data.ctau(pname) < 1e30:
            particle_list.append(pyth_data.pdg_id(pname))
    
    return particle_list
            
    
    
def test(): 
    """Test driver to show how to use the classes of this module."""
    pyth_data = PYTHIAParticleData()
    
    # List all available particles except intermediate types using the dictionary
    for pname, pvalues in pyth_data.pytname2data.iteritems():
        if pname.find('~') == -1:
            print ('{name:18s}: m0[GeV] = {m0:10.3e}, ctau[cm] = {ctau:10.3e},' + 
                   ' PDG_ID = {pdgid:10}, charge = {charge}').format(
                    name=pname, m0=pvalues[0],
                    ctau=pvalues[1], pdgid=pvalues[2], charge=pvalues[3])
    
    # Or access data using the functions (e.g. list particles (without anti-particles 
    # with lifetimes longer than D0)
    print ('\nKnown particles which lifetimes ' + 
           'longer than that of D0 ({0}cm).').format(pyth_data.ctau('D0'))
    print_stable(pyth_data.ctau('D0') / 2.99e10)
    
    print make_stable_list(1e-8)
    
    print "Example of index translation between model indices."
    # Translate SIBYLL particle codes to PYTHIA/PDG conventions
    sibtab = SibyllParticleTable()    
    for sib_id in sibtab.mod_ids:
        line = "SIBYLL ID: {0}\t SIBYLL name: {1:12s}\tPDG ID: {2}\t PYTHIA name {3}"
        pdg_id = sibtab.modid2pdg[sib_id]
        print line.format(sib_id, sibtab.modid2modname[sib_id], pdg_id, pyth_data.pdg_id2data[pdg_id][2])

            
if __name__ == '__main__':
    test()            
            
