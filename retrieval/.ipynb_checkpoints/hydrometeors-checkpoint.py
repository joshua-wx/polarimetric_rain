"""
Codes for estimating various parameters related to Hydrometeors.

@title: hydrometeors
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 04/04/2017
@date: 20/11/2017

.. autosummary::
    :toctree: generated/

    dsd_retrieval
    hydrometeor_classification
    liquid_ice_mass
    merhala_class_convstrat
    rainfall_rate
"""
# Other Libraries
import pyart
import numpy as np
import h5py

from csu_radartools import csu_blended_rain, csu_fhc

def csu_hca(radar, gatefilter, kdp_name, zdr_name, band, refl_name='DBZ_CORR',
                   rhohv_name='RHOHV_CORR',
                   temperature_name='temperature',
                   height_name='height'):
    """
    Compute CSU hydrometeo classification.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    refl_name: str
        Reflectivity field name.
    zdr_name: str
        ZDR field name.
    kdp_name: str
        KDP field name.
    rhohv_name: str
        RHOHV field name.
    temperature_name: str
        Sounding temperature field name.
    height: str
        Gate height field name.

    Returns:
    ========
    hydro_meta: dict
        Hydrometeor classification.
    """
    refl = radar.fields[refl_name]['data'].copy().filled(np.NaN)
    zdr = radar.fields[zdr_name]['data'].copy().filled(np.NaN)
    try:
        kdp = radar.fields[kdp_name]['data'].copy().filled(np.NaN)
    except AttributeError:
        kdp = radar.fields[kdp_name]['data'].copy()
    rhohv = radar.fields[rhohv_name]['data']
    try:
        radar_T = radar.fields[temperature_name]['data']
        use_temperature = True
    except Exception:
        use_temperature = False

    if use_temperature:
        scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp, use_temp=True, band=band, T=radar_T)
    else:
        scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp, use_temp=False, band=band)

    hydro = np.argmax(scores, axis=0) + 1
    hydro[gatefilter.gate_excluded] = 0
    hydro_data = np.ma.masked_equal(hydro.astype(np.int16), 0)

    the_comments = "1: Drizzle; 2: Rain; 3: Ice Crystals; 4: Aggregates; " +\
                   "5: Wet Snow; 6: Vertical Ice; 7: LD Graupel; 8: HD Graupel; 9: Hail; 10: Big Drops"

    hydro_meta = {'data': hydro_data, 'units': ' ', 'long_name': 'CSU Hydrometeor classification', '_FillValue': np.int16(0),
                  'standard_name': 'Hydrometeor_ID', 'comments': the_comments}

    return hydro_meta

def insert_ncar_pid(radar, odim_ffn):

    """
    extracts the NCAR PID from BOM ODIMH5 files into a CFRADIAL-type format and returns
    the radar object containing this new field with the required metadata
    """
    sweep_shape = np.shape(radar.get_field(0,'reflectivity'))
    pid_volume = None
    with h5py.File(odim_ffn, 'r') as f:
        h5keys = list(f.keys())
        #init 
        if 'how' in h5keys:
            h5keys.remove('how')
        if 'what' in h5keys:
            h5keys.remove('what')     
        if 'where' in h5keys:
            h5keys.remove('where')
        n_keys = len(h5keys)

        #collate padded sweeps into a volume
        for i in range(n_keys):
            ds_name = 'dataset' + str(i+1)
            pid_sweep = np.array(f[ds_name]['quality1']['data'])
            shape = pid_sweep.shape
            padded_pid_sweep = np.zeros(sweep_shape)
            padded_pid_sweep[:shape[0],:shape[1]] = pid_sweep
            if pid_volume is None:
                pid_volume = padded_pid_sweep
            else:
                pid_volume = np.vstack((pid_volume, padded_pid_sweep))

    #add to radar object
    the_comments = "0: nodata; 1: Cloud; 2: Drizzle; 3: Light_Rain; 4: Moderate_Rain; 5: Heavy_Rain; " +\
                   "6: Hail; 7: Rain_Hail_Mixture; 8: Graupel_Small_Hail; 9: Graupel_Rain; " +\
                   "10: Dry_Snow; 11: Wet_Snow; 12: Ice_Crystals; 13: Irreg_Ice_Crystals; " +\
                   "14: Supercooled_Liquid_Droplets; 15: Flying_Insects; 16: Second_Trip; 17: Ground_Clutter; " +\
                   "18: misc1; 19: misc2"
    pid_meta = {'data': pid_volume, 'units': ' ', 'long_name': 'NCAR Hydrometeor classification', '_FillValue': np.int16(0),
                  'standard_name': 'Hydrometeor_ID', 'comments': the_comments}

    radar.add_field('radar_echo_classification', pid_meta, replace_existing=True)

    return radar

def merhala_class_convstrat(radar, dbz_name="DBZ_CORR", rain_name="radar_estimated_rain_rate",
                            d0_name="D0", nw_name="NW"):
    """
    Merhala Thurai's has a criteria for classifying rain either Stratiform
    Convective or Mixed, based on the D-Zero value and the log10(Nw) value.
    Merhala's rain classification is 1 for Stratiform, 2 for Convective and 3
    for Mixed, 0 if no rain.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    dbz_name: str
        Reflectivity field name.

    Returns:
    ========
    class_meta: dict
        Merhala Thurai classification.
    """
    # Extracting data.
    d0 = radar.fields[d0_name]['data']
    nw = radar.fields[nw_name]['data']
    dbz = radar.fields[dbz_name]['data']

    classification = np.zeros(dbz.shape, dtype=np.int16)

    # Invalid data
    pos0 = (d0 >= -5) & (d0 <= 100)
    pos1 = (nw >= -10) & (nw <= 100)

    # Classification index.
    indexa = nw - 6.4 + 1.7 * d0

    # Classifying
    classification[(indexa > 0.1) & (dbz > 20)] = 2
    classification[(indexa > 0.1) & (dbz <= 20)] = 1
    classification[indexa < -0.1] = 1
    classification[(indexa >= -0.1) & (indexa <= 0.1)] = 3

    # Masking invalid data.
    classification = np.ma.masked_where(~pos0 | ~pos1 | dbz.mask, classification)

    # Generate metada.
    class_meta = {'data': classification,
                  'long_name': 'thurai_echo_classification',
                  'valid_min': 0,
                  'valid_max': 3,
                  'comment_1': 'Convective-stratiform echo classification based on Merhala Thurai',
                  'comment_2': '0 = Undefined, 1 = Stratiform, 2 = Convective, 3 = Mixed'}

    return class_meta


def rainfall_rate(radar, gatefilter, kdp_name, zdr_name, refl_name='DBZ_CORR',
                  hydro_name='radar_echo_classification', temperature_name='temperature'):
    """
    Rainfall rate algorithm from csu_radartools.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    refl_name: str
        Reflectivity field name.
    zdr_name: str
        ZDR field name.
    kdp_name: str
        KDP field name.
    hydro_name: str
        Hydrometeor classification field name.

    Returns:
    ========
    rainrate: dict
        Rainfall rate.
    """
    dbz = radar.fields[refl_name]['data'].filled(np.NaN)
    zdr = radar.fields[zdr_name]['data'].filled(np.NaN)
    fhc = radar.fields[hydro_name]['data']
    try:
        kdp = radar.fields[kdp_name]['data'].filled(np.NaN)
    except AttributeError:
        kdp = radar.fields[kdp_name]['data']

    rain, _ = csu_blended_rain.calc_blended_rain_tropical(dz=dbz, zdr=zdr, kdp=kdp, fhc=fhc, band='C')

    rain[(gatefilter.gate_excluded) | np.isnan(rain) | (rain < 0)] = 0

    try:
        temp = radar.fields[temperature_name]['data']
        rain[temp < 0] = 0
    except Exception:
        pass

    rainrate = {"long_name": 'Blended Rainfall Rate',
                "units": "mm h-1",
                "standard_name": "rainfall_rate",
                '_Least_significant_digit': 2,
                '_FillValue': np.NaN,
                "description": "Rainfall rate algorithm based on Thompson et al. 2016.",
                "data": rain.astype(np.float32)}

    return rainrate
