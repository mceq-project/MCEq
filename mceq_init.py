import os
import matplotlib.pyplot as plt
import numpy as np


#import solver related modules
from MCEq.core import MCEqRun
from mceq_config import config, mceq_config_without
#import primary model choices
import CRFluxModels as pm

mceq_run = MCEqRun(
#provide the string of the interaction model
interaction_model='SIBYLL2.3',
#primary cosmic ray flux model
#support a tuple (primary model class (not instance!), arguments)
primary_model=(pm.HillasGaisser2012, "H3a"),
# Zenith angle in degrees. 0=vertical, 90=horizontal
theta_deg=None,
#expand the rest of the options from mceq_config.py
density_model= ('CORSIKA', ('BK_USStd', None)),
**mceq_config_without(["density_model"]))