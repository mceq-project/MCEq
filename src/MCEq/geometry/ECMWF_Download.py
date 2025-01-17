

from datetime import datetime
import pandas as pd
import os
import glob
import numpy as np
import mceq_config as config

pressure_lvls = ['1', '2', '3',
                 '5', '7', '10',
                 '20', '30', '50',
                 '70', '100', '125',
                 '150', '175', '200',
                 '225', '250', '300',
                 '350', '400', '450',
                 '500', '550', '600',
                 '650', '700', '750',
                 '775', '800', '825',
                 '850', '875', '900',
                 '925', '950', '975',
                 '1000',]


def get_api_settings(date_string, time):
    '''
    Get the settings for the ERA5 API
    Args:
        date_string (str): date in the format YYYYMMDD
        time (str): time in the format HH:MM
    Returns:
        ERA5_settings (dict): dictionary with the settings for the API
    '''
    Year = date_string[0:4]
    Month = date_string[4:6]
    Day = date_string[6:8]
    ERA5_settings = {
        'product_type': ['reanalysis'],
        'data_format': 'grib',
        'variable': ['geopotential',
                     'temperature',],
        'pressure_level': pressure_lvls,
        'year': [Year],
        'month': [Month],
        'day': [Day],
        'time': [time]}
    return ERA5_settings


def Download_ERA5(Outdir, date, time):
    '''
    Download the ERA5 data for a given date and time
    uses cdsapi to download the data.
    Needs an account at https://cds.climate.copernicus.eu/
    and the cdsapi package installed and a .cdsapirc file:
    documentation: https://cds.climate.copernicus.eu/how-to-api
    Args:
        Outdir (str): path to the output directory
        date (str): date in the format YYYYMMDD
        time (str): time in the format HH:MM
    Returns:
        Filename (str): path to the downloaded grib file
    '''
    import cdsapi
    if not os.path.exists(Outdir+date[0:4]):
        os.makedirs(Outdir+date[0:4])
    Filename = Outdir+date[0:4]+'/ERA5_'+date+'_'+time+'.grib'
    if glob.glob(Filename) == []:
        cds = cdsapi.Client()
        print("Downloading data to file: ", Filename)
        cds.retrieve(
            'reanalysis-era5-pressure-levels',
            get_api_settings(date, time),
            Filename
            )
    else:
        print(f"Data already downloaded to {Filename}")
    return Filename


def read_grib_data(Filename):
    '''
    Read the grib file and extract the temperature,
    geopotential height, latitude, longitude and pressure levels
    Args:
        Filename (str): path to the grib file
    Returns:
        temps (np.array): temperature in K
        height (np.array): geopotential height in m
        lat (np.array): latitude in degrees
        long (np.array): longitude in degrees
        pressure_lvls (np.array): pressure levels in
    '''
    import pygrib
    from collections import defaultdict
    grbs = pygrib.open(Filename)
    grbs.seek(0)
    Data = defaultdict(list)

    for i, grb in enumerate(grbs):
        if 'Temperature' in str(grbs.message(i+1)):
            Data["Temperature"].append(grb.values)  # K
        elif 'Geopotential' in str(grbs.message(i+1)):
            Data["Height"].append(grb.values)  # m^2/s^2
        angular_grid = grb.latlons()
        lat = angular_grid[0]
        long = angular_grid[1]
    Output = {}
    Pressure, _, _ = np.meshgrid(np.array(pressure_lvls).astype(float),
                                 lat[:, 0],
                                 long[0, :],
                                 indexing='ij')
    Output["Temperature"] = np.array(Data["Temperature"])
    Output["Height"] = np.array(Data["Height"])/9.80665*1e2  # cm
    Output["Latitude"] = lat[:, 0]
    Output["Longitude"] = long[0, :]
    Output["Pressure"] = Pressure
    return Output


def Download_ERA5_model_lvl(Outdir, date, time):
    import cdsapi
    if not os.path.exists(Outdir+date[0:4]):
        os.makedirs(Outdir+date[0:4])
    cds = cdsapi.Client()
    Filename = Outdir+date[0:4] + \
        '/ERA5_Model_Lvl_single_day_'+date+'_' + time + '.grib2'
    if glob.glob(Filename) == []:
        Full_month_file = Outdir+date[0:4]+'/ERA5_Model_Lvl_' + \
                            date[0:6]+'01'+'.grib2'
        print(Full_month_file)
        if glob.glob(Full_month_file) == []:
            print("Downloading data to file: ", Filename)
            date = datetime.strptime(date, "%Y%m%d")
            cds.retrieve('reanalysis-era5-complete',
                         {'class': 'ea',
                          'date': datetime.strftime(date, "%Y-%m-%d"),
                          'expver': '1',
                          'levelist': '1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/'
                                      '17/18/19/20/21/22/23/24/25/26/27/28/29/'
                                      '30/31/32/33/34/35/36/37/38/39/40/41/42/'
                                      '43/44/45/46/47/48/49/50/51/52/53/54/55/'
                                      '56/57/58/59/60/61/62/63/64/65/66/67/68/'
                                      '69/70/71/72/73/74/75/76/77/78/79/80/81/'
                                      '82/83/84/85/86/87/88/89/90/91/92/93/94/'
                                      '95/96/97/98/99/100/101/102/103/104/105/'
                                      '106/107/108/109/110/111/112/113/114/'
                                      '115/116/117/118/119/120/121/122/123/'
                                      '124/125/126/127/128/129/130/131/132/'
                                      '133/134/135/136/137',
                          'levtype': 'ml',
                          'param': '129/130/152/133',
                          'stream': 'oper',
                          'grid': '1.0/1.0',
                          'time': time,
                          'type': 'an'},
                         Filename)
        else:
            Filename = Full_month_file
    return Filename


def read_grib2_Data(Filename, date=None, eccodes_dir=""):
    '''
    read tempature, geopotential height, latitude, longitude and pressure
    from a grib2 file. This requires the cfgrib package to be installed
    and the eccodes C-library to be installed. To test if the eccodes library
    was installed correctly, run the following command in the terminal:
    python -m cfgrib selfcheck
    cfgrib: https://github.com/ecmwf/cfgrib
    eccodes: https://confluence.ecmwf.int/display/ECC/
    Args:
        Filename (str): path to the grib2 file
        date (str): date in the format YYYYMMDD
    Returns:
        Output (dict): dictionary with the temperature, geopotential height,
                       latitude, longitude and pressure levels
    '''
    os.environ["ECCODES_DIR"] = eccodes_dir
    import cfgrib
    print(Filename)
    data = cfgrib.open_datasets(Filename, engine='cfgrib')
    try:
        base_data = data[0].sel(time=date)
        grid_data = data[1].sel(time=date)
        t = grid_data['t'].values[0]  # temperature at full level j
        q = grid_data['q'].values[0]  # specific humidity at full level j
        lnsp = base_data["lnsp"].values[0]
        z_ground = base_data.z.values[0]
        lat = base_data['latitude'].values[0]
        long = base_data['longitude'].values[0]
    except KeyError:
        base_data = data[0]
        grid_data = data[1]
        t = grid_data['t'].values  # temperature at full level j
        q = grid_data['q'].values  # specific humidity at full level j
        lnsp = base_data["lnsp"].values
        z_ground = base_data.z.values
        lat = base_data['latitude'].values
        long = base_data['longitude'].values
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mldf = pd.read_csv(dir_path + '/akbk.csv', index_col='n')
    Height, Temp, Press = geopot(lnsp, t, q, z_ground, mldf)
    Output = {}
    Output["Temperature"] = Temp
    Output["Height"] = Height
    Output["Latitude"] = lat
    Output["Longitude"] = long
    Output["Pressure"] = Press
    return Output


def geopot(lnsp, t, q, z_ground, ml_df):
    '''
    source:  IFS Documentation - Cy41r1, Part III:
             Dynamics and Numerical Procedures
    https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
    https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height
    all equations referenced are found in this document
    '''
    # gas constants for dry air and water vapor
    R_D = 287.06    # dry air,     J K**-1 kg**-1
    R_V = 461       # water vapor, J K**-1 kg**-1
    g = 9.80665     # m/s**2
    # get pressure at levels and half levels
    a = ml_df.loc[:, 'a [Pa]']
    a = a.to_numpy()[:, np.newaxis, np.newaxis]
    b = ml_df.loc[:, 'b']
    b = b.to_numpy()[:, np.newaxis, np.newaxis]
    # surface pressure is given in log-values
    p_k = a+b*np.exp(lnsp)[np.newaxis]
    p_up = p_k[1:, :, :]  # pressure at half layer above (lower than pl)
    p_low = p_k[:-1, :, :]  # pressure at half layer below (higher than pu)
    delta_p_k = p_up-p_low

    # calculate coefficient alpha_k
    # as given by equation 2.23
    alpha_k = 1-(p_low/delta_p_k)*np.log(p_up/p_low)
    alpha_k[0] = np.log(2)
    # get temperature and specific humidity at full levels
    tv = t*(1+(R_V/R_D-1.0)*q)  # virtual temperature at full level j

    # need to add geopotential at surface
    # implementation of equation 2.21
    #   level 0.5 corresponds to the top of the atmosphere
    #   level 137.5 to the surface
    #   full levels range from 1 to 137
    phi_hl = []
    for k in range(0, 137):
        s = z_ground +\
            np.sum(R_D*tv[k+1:]*np.log(p_up[k+1:]/p_low[k+1:]), axis=0)
        phi_hl.append(s)
    phi_hl = np.array(phi_hl)
    phi_k = phi_hl + alpha_k*R_D*tv
    # get longitude and latitude
    p = 1/2.*(p_low+p_up)/1e2  # hPa
    z = phi_k/g*1e2  # cm
    # geopotential height to altitude
    height = config.r_E * z/(config.r_E - z)  # cm
    return height, t, p
