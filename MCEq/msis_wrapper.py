#! /usr/bin/env python
from mceq_config import config

if config['msis_python'] == 'ctypes':
    from c_msis_interface import *
else:
    from nrlmsise_00_header import *
    from nrlmsise_00 import gtd7

#===============================================================================
# NRLMSISE00
#===============================================================================
class NRLMSISE00Base():
    def __init__(self):
        self.input = nrlmsise_input()
        self.output = nrlmsise_output()
        self.flags = nrlmsise_flags()
        self.month2doy = {'January':1,
                          'February':32,
                          'March':60,
                          'April':91,
                          'May':121,
                          'June':152,
                          'July':182,
                          'August':213,
                          'September':244,
                          'October':274,
                          'November':305,
                          'December':335}
        
        self.locations = {'SouthPole':(-90., 0., 2834.*100.),
                          'Karlsruhe':(49., 8.4, 110. *100.)}
        
        self.daytimes = {'day':43200.,
                         'night':0.}
        self.current_location = 'SouthPole'
        self.init_default_values()
    
    def surface_vert_depth(self, loc='SouthPole', month='June'):
        self.set_location('SouthPole')
        self.set_season('June')
        print loc, month, self.height2depth(self.alt_surface)
        
    def height2depth(self, altitude_cm):
        from scipy.integrate import quad
        return quad(self.get_density, altitude_cm, 112.8 * 1e5,
                    epsrel=0.001)[0]

class pyNRLMSISE00(NRLMSISE00Base):                      
    def init_default_values(self):
        """Sets default to June at South Pole"""
        self.input.doy = self.month2doy['June']  # Day of year
        self.input.year = 0  # No effect
        self.input.sec = self.daytimes['day']  # 12:00
        self.input.alt = self.locations[self.current_location][2]
        self.input.g_lat = self.locations[self.current_location][1]
        self.input.g_long = self.locations[self.current_location][0]
        self.input.lst = self.input.sec / 3600. + \
                                  self.input.g_long / 15.
        # Do not touch this except you know what you are doing
        self.input.f107A = 150.
        self.input.f107 = 150.
        self.input.ap = 4.
        self.input.ap_a = ap_array()
        self.alt_surface = self.locations[self.current_location][2]
        
        self.flags.switches[0] = 0
        for i in range(1, 24):
            self.flags.switches[i] = 1
            
    def set_location(self, tag):
        if tag not in self.locations.keys():
            raise Exception("NRLMSISE00::set_location(): Unknown location tag '{0}'.".format(tag))
        self.input.alt = self.locations[tag][2]
        self.set_location_coord(self.locations[tag][1], self.locations[tag][0])
        self.current_location = tag
        self.alt_surface = self.locations[self.current_location][2]

    def set_location_coord(self, latitude, longitude):
        if abs(latitude) > 180 or abs(longitude) > 90:
            raise Exception("NRLMSISE00::set_location_coord(): Invalid input.")
        self.input.g_lat = latitude
        self.input.g_long = longitude
    
    def set_season(self, tag):
        if tag not in self.month2doy.keys():
            raise Exception("NRLMSISE00::set_location(): Unknown season tag.")
        self.input.doy = self.month2doy[tag]
        
    def get_density(self, altitude_cm):
        self.input.alt = altitude_cm / 1e5
        gtd7(self.input, self.flags, self.output)
        return self.output.d[5]


class cNRLMSISE00(NRLMSISE00Base):                      
    def init_default_values(self):
        """Sets default to June at South Pole"""
        self.input.doy = c_int(self.month2doy['June'])  # Day of year
        self.input.year = c_int(0)  # No effect
        self.input.sec = c_double(self.daytimes['day'])  # 12:00
        self.input.alt = c_double(self.locations[self.current_location][2])
        self.input.g_lat = c_double(self.locations[self.current_location][1])
        self.input.g_long = c_double(self.locations[self.current_location][0])
        self.input.lst = c_double(self.input.sec.value / 3600. + \
                                  self.input.g_long.value / 15.)
        # Do not touch this except you know what you are doing
        self.input.f107A = c_double(150.)
        self.input.f107 = c_double(150.)
        self.input.ap = c_double(4.)
        self.input.ap_a = pointer(ap_array())
        self.alt_surface = self.locations[self.current_location][2]
        
        self.flags.switches[0] =c_int(0)
        for i in range(1, 24):
            self.flags.switches[i] = c_int(1)
            
    def set_location(self, tag):
        if tag not in self.locations.keys():
            raise Exception("NRLMSISE00::set_location(): Unknown location tag '{0}'.".format(tag))
        self.input.alt = c_double(self.locations[tag][2])
        self.set_location_coord(self.locations[tag][1], self.locations[tag][0])
        self.current_location = tag
        self.alt_surface = self.locations[self.current_location][2]

    def set_location_coord(self, latitude, longitude):
        if abs(latitude) > 180 or abs(longitude) > 90:
            raise Exception("NRLMSISE00::set_location_coord(): Invalid input.")
        self.input.g_lat = c_double(latitude)
        self.input.g_long = c_double(longitude)
    
    def set_season(self, tag):
        if tag not in self.month2doy.keys():
            raise Exception("NRLMSISE00::set_location(): Unknown season tag.")
        self.input.doy = self.month2doy[tag]
        
    def get_density(self, altitude_cm):
        input = self.input
        input.alt = c_double(altitude_cm / 1e5)
        msis.gtd7_py(input.year, input.doy, input.sec,
                     input.alt, input.g_lat,
                     input.g_long, input.lst,
                     input.f107A, input.f107,
                     input.ap, input.ap_a,
                     byref(self.flags), byref(self.output))
        return self.output.d[5]


def test():
    import numpy as np
    import matplotlib.pyplot as plt
    
    msis = NRLMSISE00()
    den = np.vectorize(msis.get_density)
    
    plt.figure(figsize=(16, 5))
    plt.suptitle('NRLMSISE-00')
    
    h_vec = np.linspace(0, 112.8 * 1e5, 500)
    msis.set_season('January')
    msis.set_location('SouthPole')
    den_sp_jan = den(h_vec)
    
    msis.set_season('January')
    msis.set_location('Karlsruhe')
    den_ka_jan = den(h_vec)
    
    plt.subplot(131)
    plt.semilogy(h_vec / 1e5, den_sp_jan, label='MSIS00: SP Jan.')
    plt.semilogy(h_vec / 1e5, den_ka_jan, label='MSIS00: KA Jan.')
    plt.legend()
    plt.xlabel('vertical height in km')
    plt.ylabel(r'density $\rho(h)$ in g/cm$^3$')

    plt.subplot(132)
    plt.plot(h_vec / 1e5, den_ka_jan / den_sp_jan, label='MSIS00: KA/SP')
    plt.xlabel('vertical height in km')
    plt.ylabel(r'density ratio')
    plt.legend(loc='upper left')
    
    plt.subplot(133)
    msis.set_location('SouthPole')
    for i in range(360 / 30):
        msis.input.doy = i * 30
        plt.plot(h_vec / 1e5, den(h_vec) / den_sp_jan, label=str(i + 1))
    plt.legend(ncol=2, loc=3)
    plt.title('MSIS00: SouthPole')
    plt.xlabel('vertical height in km')
    plt.ylabel(r'$\rho$(Month) / $\rho$(January)')
    plt.ylim(ymin=0.6)
    plt.tight_layout()
    
    plt.figure(figsize=(6, 5))
    h2d = np.vectorize(msis.height2depth)
    plt.semilogy(h_vec / 1e5, h2d(h_vec))
    plt.ylabel(r'Slant depth X [g/cm$^2$]')
    plt.xlabel(r'Atmospheric height $h$ [km]')
    plt.subplots_adjust(left=0.15, bottom=0.11)
    plt.show()

if __name__ == '__main__':
    test()
    
