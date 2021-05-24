"""
Codes for correcting and estimating various radar and meteorological parameters.

@title: radar_codes
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 04/04/2017
@date: 14/04/2020

.. autosummary::
    :toctree: generated/

    _my_snr_from_reflectivity
    _nearest
    check_azimuth
    check_reflectivity
    check_year
    correct_rhohv
    correct_zdr
    get_radiosoundings
    read_radar
    snr_and_sounding
"""
# Python Standard Library
import os
import re
from glob import glob
import time
import fnmatch
from datetime import datetime

# Other Libraries
import pyart
import scipy
import cftime
import netCDF4
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

def get_wavelength(h5_ffn):
    hfile = h5py.File(h5_ffn, 'r')
    global_how = hfile['how'].attrs
    return global_how['wavelength']

def snr_from_reflectivity(radar, refl_field='DBZ'):
    """
    Just in case pyart.retrieve.calculate_snr_from_reflectivity, I can calculate
    it 'by hand'.
    Parameter:
    ===========
    radar:
        Py-ART radar structure.
    refl_field_name: str
        Name of the reflectivity field.
    Return:
    =======
    snr: dict
        Signal to noise ratio.
    """
    range_grid, _ = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    range_grid += 1  # Cause of 0

    # remove range scale.. This is basically the radar constant scaled dBm
    pseudo_power = (radar.fields[refl_field]['data'] - 20.0 * np.log10(range_grid / 1000.0))
    # The noise_floor_estimate can fail sometimes in pyart, that's the reason
    # why this whole function exists.
    noise_floor_estimate = -40
    snr_data = pseudo_power - noise_floor_estimate

    return snr_data

def _nearest(items, pivot):
    """
    Find the nearest item.

    Parameters:
    ===========
        items:
            List of item.
        pivot:
            Item we're looking for.

    Returns:
    ========
        item:
            Value of the nearest item found.
    """
    return min(items, key=lambda x: abs(x - pivot))


def check_reflectivity(radar, refl_field_name='DBZ'):
    """
    Checking if radar has a proper reflectivity field.  It's a minor problem
    concerning a few days in 2011 for CPOL.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_field_name: str
            Name of the reflectivity field.

    Return:
    =======
    True if radar has a non-empty reflectivity field.
    """
    dbz = radar.fields[refl_field_name]['data']

    if np.ma.isMaskedArray(dbz):
        if dbz.count() == 0:
            # Reflectivity field is empty.
            return False

    return True

def correct_rhohv(radar, rhohv_name='RHOHV', snr_name='SNR'):
    """
    Correct cross correlation ratio (RHOHV) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 5)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        rhohv_name: str
            Cross correlation field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        rho_corr: array
            Corrected cross correlation ratio.
    """
    rhohv = radar.fields[rhohv_name]['data'].copy()
    snr = radar.fields[snr_name]['data'].copy()

    #appears to be done onsite at CP2
#     natural_snr = 10**(0.1 * snr)
#     natural_snr = natural_snr.filled(-9999)
#     rho_corr = rhohv * (1 + 1 / natural_snr)
    rho_corr = rhohv
    
    # Not allowing the corrected RHOHV to be lower than the raw rhohv
    rho_corr[np.isnan(rho_corr) | (rho_corr < 0) | (rho_corr > 1)] = 1
    try:
        rho_corr = rho_corr.filled(1)
    except Exception:
        pass

    return np.ma.masked_array(rho_corr, rhohv.mask)


def correct_standard_name(radar):
    """
    'standard_name' is a protected keyword for metadata in the CF conventions.
    To respect the CF conventions we can only use the standard_name field that
    exists in the CF table.

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    try:
        radar.range.pop('standard_name')
        radar.azimuth.pop('standard_name')
        radar.elevation.pop('standard_name')
    except Exception:
        pass

    try:
        radar.sweep_number.pop('standard_name')
        radar.fixed_angle.pop('standard_name')
        radar.sweep_mode.pop('standard_name')
    except Exception:
        pass

    good_keys = ['corrected_reflectivity', 'total_power', 'radar_estimated_rain_rate', 'corrected_velocity']
    for k in radar.fields.keys():
        if k not in good_keys:
            try:
                radar.fields[k].pop('standard_name')
            except Exception:
                continue

    try:
        radar.fields['velocity']['standard_name'] = 'radial_velocity_of_scatterers_away_from_instrument'
        radar.fields['velocity']['long_name'] = 'Doppler radial velocity of scatterers away from instrument'
    except KeyError:
        pass

    radar.latitude['standard_name'] = 'latitude'
    radar.longitude['standard_name'] = 'longitude'
    radar.altitude['standard_name'] = 'altitude'

    return None


def correct_zdr(radar, zdr_name='ZDR', snr_name='SNR'):
    """
    Correct differential reflectivity (ZDR) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 6)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        corr_zdr: array
            Corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]['data'].copy()
    snr = radar.fields[snr_name]['data'].copy()
    alpha = 1.48
    natural_zdr = 10 ** (0.1 * zdr)
    natural_snr = 10 ** (0.1 * snr)
    corr_zdr = 10 * np.log10((alpha * natural_snr * natural_zdr) / (alpha * natural_snr + alpha - natural_zdr))

    return corr_zdr


def coverage_content_type(radar):
    """
    Adding metadata for compatibility with ACDD-1.3

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    radar.range['coverage_content_type'] = 'coordinate'
    radar.azimuth['coverage_content_type'] = 'coordinate'
    radar.elevation['coverage_content_type'] = 'coordinate'
    radar.latitude['coverage_content_type'] = 'coordinate'
    radar.longitude['coverage_content_type'] = 'coordinate'
    radar.altitude['coverage_content_type'] = 'coordinate'

    radar.sweep_number['coverage_content_type'] = 'auxiliaryInformation'
    radar.fixed_angle['coverage_content_type'] = 'auxiliaryInformation'
    radar.sweep_mode['coverage_content_type'] = 'auxiliaryInformation'

    for k in radar.fields.keys():
        if k == 'radar_echo_classification':
            radar.fields[k]['coverage_content_type'] = 'thematicClassification'
        elif k in ['normalized_coherent_power', 'normalized_coherent_power_v']:
            radar.fields[k]['coverage_content_type'] = 'qualityInformation'
        else:
            radar.fields[k]['coverage_content_type'] = 'physicalMeasurement'

    return None


def read_radar(radar_file_name):
    """
    Read the input radar file.

    Parameter:
    ==========
        radar_file_name: str
            Radar file name.

    Return:
    =======
        radar: struct
            Py-ART radar structure.
    """
    # Read the input radar file.
    try:
        #load radar
        radar = pyart.io.read_mdv(radar_file_name, file_field_names=True)
        
    except Exception:
        raise
        
    radar.fields['VEL']['units'] = "m s-1"
    return radar


def temperature_profile_access(radar, source='access'):
    """
    Compute the signal-to-noise ratio as well as interpolating the radiosounding
    temperature on to the radar grid. The function looks for the radiosoundings
    that happened at the closest time from the radar. There is no time
    difference limit.
    Parameters:
    ===========
    radar:
        Py-ART radar object.
    source:
        string of either access (access g) or era5
    Returns:
    ========
    z_dict: dict
        Altitude in m, interpolated at each radar gates.
    temp_info_dict: dict
        Temperature in Celsius, interpolated at each radar gates.
    """
    grlat = radar.latitude['data'][0]
    grlon = radar.longitude['data'][0]
    dtime = pd.Timestamp(cftime.num2pydate(radar.time['data'][0], radar.time['units']))

    #build paths
    request_date = datetime.strftime(dtime, '%Y%m%d')
    request_time = str(round(dtime.hour/6)*6).zfill(2) + '00'
    if request_time == '2400':
        request_time = '0000'
    
    if source == 'access':
        if dtime < datetime.strptime('20200924', '%Y%m%d'):
            #APS2
            access_root = '/g/data/lb4/ops_aps2/access-g/1' #access g
        else:
            #APS3
            access_root = '/g/data/wr45/ops_aps3/access-g/1' #access g
        access_folder = '/'.join([access_root, request_date, request_time, 'an', 'pl'])
        #build filenames
        temp_ffn = access_folder + '/air_temp.nc'
        geop_ffn = access_folder + '/geop_ht.nc'
        if not os.path.isfile(temp_ffn):
            raise FileNotFoundError(f'{temp_ffn}: no such file for temperature.')
        if not os.path.isfile(geop_ffn):
            raise FileNotFoundError(f'{geop_ffn}: no such file for geopotential.')
        #extract data
        with xr.open_dataset(temp_ffn) as temp_ds:
            temp_profile = temp_ds.air_temp.sel(lon=grlon, method='nearest').sel(lat=grlat, method='nearest').data[0] - 273.15 #units: deg C
        with xr.open_dataset(geop_ffn) as geop_ds:
            geopot_profile = geop_ds.geop_ht.sel(lon=grlon, method='nearest').sel(lat=grlat, method='nearest').data[0] #units: m
    elif source == "era5":
        #set era path
        era5_root = '/g/data/rt52/era5/pressure-levels/reanalysis'
        #build file paths
        month_str = dtime.month
        year_str = dtime.year
        temp_ffn = glob(f'{era5_root}/t/{year_str}/t_era5_oper_pl_{year_str}{month_str:02}*.nc')[0]
        geop_ffn = glob(f'{era5_root}/z/{year_str}/z_era5_oper_pl_{year_str}{month_str:02}*.nc')[0]
        #extract data
        with xr.open_dataset(temp_ffn) as temp_ds:
            temp_data = temp_ds.t.sel(longitude=grlon, method='nearest').sel(latitude=grlat, method='nearest').sel(time=dtime, method='nearest').data[:] - 273.15 #units: deg K -> C
        with xr.open_dataset(geop_ffn) as geop_ds:
            geop_data = geop_ds.z.sel(longitude=grlon, method='nearest').sel(latitude=grlat, method='nearest').sel(time=dtime, method='nearest').data[:]/9.80665 #units: m**2 s**-2 -> m
        #flipdata (ground is first row)
        temp_profile = np.flipud(temp_data)
        geopot_profile = np.flipud(geop_data)
        

    
    #append surface data using lowest level
    geopot_profile = np.append([0], geopot_profile)
    temp_profile = np.append(temp_profile[0], temp_profile)
    
    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(temp_profile, geopot_profile, radar)
    
    temp_info_dict = {'data': temp_dict['data'],  # Switch to celsius.
                      'long_name': 'Sounding temperature at gate',
                      'standard_name': 'temperature',
                      'valid_min': -100, 'valid_max': 100,
                      'units': 'degrees Celsius',
                      'comment': 'Radiosounding date: %s' % (dtime.strftime("%Y/%m/%d"))}
    
    #generate isom dataset
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
    

    return z_dict, temp_info_dict, isom_info_dict


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

    intp_h = np.nan

    #check if target_temp is warmer than lowest level in sounding
    if target_temp>snd_temp[0]:
        print('warning, target temp level below sounding, returning ground level (0m)')
        return 0.
    
    # find index above and below freezing level
    mask = np.where(snd_temp < target_temp)
    above_ind = mask[0][0]

    # index below
    below_ind = above_ind - 1
    # apply linear interplation to points above and below target_temp
    set_interp = interp1d(
        snd_temp[below_ind:above_ind+1],
        snd_height[below_ind:above_ind+1], kind='linear')
    # apply interpolant
    intp_h = set_interp(target_temp)
    
    return intp_h

    
def find_melting_level(temp_profile, geop_profile):
    #interpolate to required levels
    plus10_h = _sounding_interp(temp_profile, geop_profile, 10.)
    fz_h = _sounding_interp(temp_profile, geop_profile, 0.)
    #calculate base of melting level
    return (plus10_h+fz_h)/2