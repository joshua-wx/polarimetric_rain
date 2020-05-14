# Python Standard Library
import os
import re
import glob
import time
import fnmatch
import datetime

# Other Libraries
import pyart
import scipy
import cftime
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

def _rolling_window(a, window):
    """ Create a rolling window object for application of functions
    eg: result=np.ma.std(array, 11), 1). """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def smooth_masked(raw_data, wind_len=11, min_valid=6, wind_type='median'):
    """
    Smoothes the data using a rolling window.
    data with less than n valid points is masked.
    Parameters
    ----------
    raw_data : float masked array
        The data to smooth.
    win_len : float
        Length of the moving window.
    min_valid : float
        Minimum number of valid points for the smoothing to be valid.
    wind_type : str
        Type of window. Can be median or mean.
    Returns
    -------
    data_smooth : float masked array
        Smoothed data.
    """
    valid_wind = ['median', 'mean']
    if wind_type not in valid_wind:
        raise ValueError(
            "Window " + win_type + " is none of " + ' '.join(valid_wind))

    # we want an odd window
    if wind_len % 2 == 0:
        wind_len += 1
    half_wind = int((wind_len-1)/2)

    # initialize smoothed data
    nrays, nbins = np.shape(raw_data)
    data_smooth = np.ma.zeros((nrays, nbins))
    data_smooth[:] = np.ma.masked
    data_smooth.set_fill_value(np.nan)

    mask = np.ma.getmaskarray(raw_data)
    valid = np.logical_not(mask)

    mask_wind = _rolling_window(mask, wind_len)
    valid_wind = np.logical_not(mask_wind).astype(int)
    nvalid = np.sum(valid_wind, -1)

    data_wind = _rolling_window(raw_data, wind_len)

    # check which gates are valid
    ind_valid = np.logical_and(
        nvalid >= min_valid, valid[:, half_wind:-half_wind]).nonzero()

    data_smooth[ind_valid[0], ind_valid[1]+half_wind] = (
        eval('np.ma.' + wind_type + '(data_wind, axis=-1)')[ind_valid])

    return data_smooth



def generate_isom(radar):
    """
    Generate fields for radar object that is height relative to the
    melting level (at the radar site using era5 data)
    Parameters:
    ===========
    radar:
        Py-ART radar object.
    Returns:
    ========
    iso0_info_dict: dict
        Height field relative to melting level
    """
    grlat = radar.latitude['data'][0]
    grlon = radar.longitude['data'][0]
    dtime = pd.Timestamp(cftime.num2pydate(radar.time['data'][0], radar.time['units']))

    year = dtime.year
    era5 = f'/g/data/rq0/admin/temperature_profiles/era5_data/{year}_openradar_temp_geopot.nc'
    if not os.path.isfile(era5):
        raise FileNotFoundError(f'{era5}: no such file for temperature.')

    # Getting the temperature
    dset = xr.open_dataset(era5)
    temp = dset.sel(longitude=grlon, latitude=grlat, time=dtime, method='nearest')
    
    #extract data
    geopot_profile = np.array(temp.z.values/9.80665) #geopot -> geopotH
    temp_profile = np.array(temp.t.values - 273.15)
    
    #append surface data using lowest level
    geopot_profile = np.append(geopot_profile,[0])
    temp_profile = np.append(temp_profile, temp_profile[-1])
        
    #find melting level
    melting_level = find_melting_level(temp_profile, geopot_profile)
    
    # retrieve the Z coordinates of the radar gates
    rg, azg = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    rg, eleg = np.meshgrid(radar.range['data'], radar.elevation['data'])
    _, _, z = pyart.core.antenna_to_cartesian(rg / 1000.0, azg, eleg)
    
    #calculate height above melting level
    isom_data = (radar.altitude['data'] + z) - melting_level
    isom_data[isom_data<0] = 0
    
    isom_info_dict = {'data': isom_data, # relative to melting level
                      'long_name': 'Height relative to (H0+H10)/2 level',
                      'standard_name': 'relative_melting_level_height',
                      'units': 'm'}
    
    radar.add_field('height_over_isom', isom_info_dict)

    return radar

def _sounding_interp(snd_temp,snd_height,target_temp):
    """
    Provides an linear interpolated height for a target temperature using a sounding vertical profile. 
    Looks for first instance of temperature below target_temp from surface upward.

    Parameters:
    ===========
    snd_temp: ndarray
        temperature data (degrees C)
    snd_height: ndarray
        relative height data (m)
    target_temp: float
        target temperature to find height at (m)

    Returns:
    ========
    intp_h: float
        interpolated height of target_temp (m)
    """

    snd_temp = np.flip(snd_temp)
    snd_height = np.flip(snd_height)
    
    intp_h = np.nan

    #find index above and below freezing level
    mask      = np.where(snd_temp<target_temp)
    above_ind = mask[0][0]
    #check to ensure operation is possible
    if above_ind > 0:
        #index below 
        below_ind  = above_ind-1
        #apply linear interplation to points above and below target_temp
        set_interp = interp1d(snd_temp[below_ind:above_ind+1], snd_height[below_ind:above_ind+1], kind='linear')
        #apply interpolant
        intp_h     = set_interp(target_temp)   
        return intp_h
    else:
        return target_temp[0]
    
def find_melting_level(temp_profile, geop_profile):
    #interpolate to required levels
    minus10_h = _sounding_interp(temp_profile, geop_profile, -10.)
    fz_h = _sounding_interp(temp_profile, geop_profile, 0.)
    #calculate base of melting level
    return (minus10_h+fz_h)/2

def add_ncar_pid(radar, derived_path, vol_ffn):
    
    """
    pid.cl	    (1)	 # Cloud                        
    pid.drz     (2)  # Drizzle                      
    pid.lr	    (3)  # Light_Rain                   
    pid.mr	    (4)  # Moderate_Rain                
    pid.hr	    (5)  # Heavy_Rain                   
    pid.ha	    (6)  # Hail                         
    pid.rh	    (7)  # Rain_Hail_Mixture            
    pid.gsh     (8)  # Graupel_Small_Hail           
    pid.grr     (9)  # Graupel_Rain                 
    pid.ds	    (10) # Dry_Snow                     
    pid.ws	    (11) # Wet_Snow                     
    pid.ic	    (12) # Ice_Crystals                 
    pid.iic     (13) # Irreg_Ice_Crystals           
    pid.sld     (14) # Supercooled_Liquid_Droplets  
    pid.bgs     (15) # Flying_Insects               
    pid.trip2   (16) # Second trip                  
    pid.gcl     (17) # Ground_Clutter          
    pid.sat     (18) # Receiver saturation
    """
    
    #build derived filename
    vol_fn = os.path.basename(vol_ffn)[6:-12]
    derived_fn = 'cp2-derived_' + vol_fn + '.mdv'
    derived_ffn = derived_path + '/' + derived_fn
    #load
    derived = pyart.io.read(derived_ffn, file_field_names=True)
    #extract PID
    pid = derived.fields['PID']
    #add to radar
    radar.add_field('PID', pid, replace_existing=True)
    return radar
    