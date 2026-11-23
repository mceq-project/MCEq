With ``enable_cont_rad_loss = False`` the matrix builder dropped the
continuous-loss band entirely, although the database backend loads the
ionization dE/dX for that case; muons and e± lost no energy. The band is
built again from the ionization set, as in v1.4.
