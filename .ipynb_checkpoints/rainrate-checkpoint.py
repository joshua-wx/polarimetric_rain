import numpy as np

import pyart

def _invpower_func(z, a, b):
    return (z / a) ** (1. / b)

def _power_func(data, a, b):
    return a*(data**b)


def conventional(radar, alpha=60, beta=1.7, refl_field='reflectivity', zr_field='zr_rainrate'):
    """
    WHAT: retrieve conventional rain rates using ZR technique
    INPUTS:
        radar: pyart radar object
        alpha/beta: coefficents used in inverse powerlaw function to derive rainrate from Z (float)
        various field names for input and output
    OUTPUTS:
        radar: pyart radar object
    """
    
    
    #for IDR66, RF: a=92, b=1.7, newZR: a=283.4, b=1.32
    
    #get reflectivity field
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data'].copy()
    refl_z = 10.**(np.asarray(refl)/10.)
    
    #calculate rain
    rr_data = _invpower_func(refl_z, alpha, beta)
    
    #empty rain fields
    rain_field = pyart.config.get_metadata('radar_estimated_rain_rate')
    rain_field['data'] = rr_data
    
    #add to radar
    radar.add_field(zr_field, rain_field, replace_existing=True)
    radar.fields[zr_field]['standard_name'] = 'zr_rainrate'
    radar.fields[zr_field]['long_name'] = 'Rainrate from R(Z)'
    
    return radar
    
def polarimetric(radar, refl_field='sm_reflectivity', ah_field='specific_attenuation', kdp_field='corrected_specific_differential_phase',
                 zr_field='zr_rainrate', ahr_field='ah_rainrate', kdpr_field='kdp_rainrate', hybridr_field='hybrid_rainrate',
                 refl_threshold=50.,
                 ah_a=4120., ah_b=1.03,
                 kdp_a=27., kdp_b=0.77):
    
    """
    WHAT: retrieve polarimetric rain rates for ah, kdp and hybrid kdp/ah technique
    INPUTS:
        radar: pyart radar object
        refl_threshold: threshold to define transition from ah to kdp rainrate retrieval (dB, float)
        ah_a/ah_b: coefficents used in powerlaw function to derive rainrate from ah (float)
        kdp_a/kdp_b: coefficents used in powerlaw function to derive rainrate from kdp (float)
        various field names used for input and output
    OUTPUTS:
        radar: pyart radar object
    """
    
    #get fields
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data'].copy()
    radar.check_field_exists(ah_field)
    ah = radar.fields[ah_field]['data'].copy()
    radar.check_field_exists(kdp_field)
    kdp = radar.fields[kdp_field]['data'].copy()
    
    #retrieve rainrates
    ah_rain = _power_func(ah, ah_a, ah_b)
    kdp_rain = _power_func(kdp, kdp_a, kdp_b)

    #create rain and hail masks
    hail_mask = refl>refl_threshold
    
    #crate hybrid kdp/ah rainrate
    hybrid_rain = ah_rain.copy()
    hybrid_rain[hail_mask] = kdp_rain[hail_mask]
    
    #add fields to radar object
    radar.add_field_like(zr_field, hybridr_field, hybrid_rain, replace_existing=True)
    radar.add_field_like(zr_field, ahr_field, ah_rain, replace_existing=True)
    radar.add_field_like(zr_field, kdpr_field, kdp_rain, replace_existing=True)
    
    #update names
    radar.fields[hybridr_field]['standard_name'] = 'hydrid_a_and_kdp_rainrate'
    radar.fields[hybridr_field]['long_name'] = 'Rainrate from R(A) and R(kdp) using reflectivity threshold of ' + str(refl_threshold)
    radar.fields[ahr_field]['standard_name'] = 'a_rainrate'
    radar.fields[ahr_field]['long_name'] = 'Rainrate from R(A)'
    radar.fields[kdpr_field]['standard_name'] = 'kdp_rainrate'
    radar.fields[kdpr_field]['long_name'] = 'Rainrate from R(kdp)'
    
    return radar