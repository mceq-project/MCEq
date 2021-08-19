#import matplotlib.pyplot as plt
#import numpy as np

#import solver related modules
from MCEq.core import MCEqRun
import mceq_config as config
#import primary model choices
import crflux.models as pm

mceq_run = MCEqRun(
    #provide the string of the interaction model
    interaction_model='SIBYLL2.3d',
    #primary cosmic ray flux model
    primary_model = (pm.HillasGaisser2012, "H3a"),
    # Zenith angle in degrees. 0=vertical, 90=horizontal
    theta_deg=85.0
)

for i in range(2):
    mceq_run.solve()




