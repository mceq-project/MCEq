from __future__ import print_function
import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm

config.debug_level = 5
config.mceq_db_fname = 'mceq_db_lext_dpm191.h5'
config.kernel_config = 'CUDA'
config.CUDA_GPU_ID = 2
config.mkl_threads = 8

mceq = MCEqRun(
    interaction_model='SIBYLL23C',
    theta_deg=0.,
    primary_model=(pm.HillasGaisser2012, 'H3a')
)

mceq.set_theta_deg(60.)
mceq.solve()

print(mceq.get_solution('mu+', 3) + mceq.get_solution('mu-', 3))
