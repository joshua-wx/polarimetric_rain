import cftime
import numpy as np
from matplotlib import pyplot as plt

from skimage import morphology
from scipy.integrate import cumtrapz
from sklearn.linear_model import LinearRegression

import pyart

import common

def _find_z_zdr_slope(alpha_dict):
    
    """
    WHAT: fits slope to Z-ZDR pairs stored in alpha direct object. Used for estimating alpha
    INPUT:
        alpha_dict: dictionary containing Z and ZDR pairs and alpha ts
    OUTPUT:
        slope coefficent (float)
    """
    
    plot_fits=False
    
    #setup bins
    z_bin_width = 2
    z_bin_centres = np.arange(20, 50+z_bin_width, z_bin_width)
    median_zdr = np.zeros_like(z_bin_centres, dtype=float)
    #run binning
    for i, centre in enumerate(z_bin_centres):
        bin_lower = centre - z_bin_width/2
        bin_upper = centre + z_bin_width/2
        bin_mask = np.logical_and(alpha_dict['z_pairs']>=bin_lower, alpha_dict['z_pairs']<bin_upper)
        median_zdr[i] = np.median(alpha_dict['zdr_pairs'][bin_mask])
    #linear regression
    LS = LinearRegression()
    LS.fit(z_bin_centres.reshape(-1, 1), median_zdr) #, sample_weight=sample_weight)
    
    #calculate slope
    if plot_fits:
        plt.plot(alpha_dict['z_pairs'], alpha_dict['zdr_pairs'], 'k.', label='Pairs')
        plt.plot(z_bin_centres, median_zdr, 'r.', label='Median ZDR')
        plt.plot(z_bin_centres, LS.predict(z_bin_centres.reshape(-1, 1)), 'b-', label='LS fit')
        plt.legend()
        plt.xlabel('Z')
        plt.ylabel('ZDR')
    
    return LS.coef_[0]
    
def estimate_alpha(radar, alpha_dict, default_alpha=0.015, pair_threshold=30000,
                   min_z=20, max_z=50, min_zdr=-4, max_zdr=4, min_rhohv=0.98, z_zdr_slope_threshold=0.045,
                  refl_field='reflectivity', zdr_field='corrected_differential_reflectivity', rhohv_field='corrected_cross_correlation_ratio',
                  isom_field='height_over_isom'):
    
    """
    WHAT: Estimate alpha by accumulating Z - ZDR pairs across scans until the pair threshold has been reaches,
            and then fitting a slope to these pairs using _find_z_zdr_slope.
    INPUT:
        radar: pyart radar object
        alpha_dict: dictionary containing Z and ZDR pairs and alpha ts
        default_alpha: default alpha to use (dependent on wavelength, float)
        pair_threshold: number of pairs required for Z-ZDR slope calculation (int)
        min_z: minimum reflectivity for pairs (float, dB)
        max_z: maximum reflectivity for pairs (float, dB)
        max_zdr: minimum differential reflectivity for pairs (float, dB)
        min_zdr: maximum differential reflectivity for pairs (float, dB)
        min_rhohv: minimum cross correlation for pairs (float)
        z_zdr_slope_threshold: slope values below this threshold use the default alpha (float)
        
    OUTPUT:
        alpha_dict: dictionary containing Z and ZDR pairs and alpha ts
    """
        
    verbose = False
    
    #get radar time
    radar_starttime = cftime.num2pydate(radar.time['data'][0], radar.time['units'])
    #extract data
    z_data = radar.get_field(0, refl_field).filled()
    zdr_data = radar.get_field(0, zdr_field).filled()
    rhohv_data = radar.get_field(0, rhohv_field).filled()
    isom_data = radar.get_field(0, isom_field)
    
    #build masks
    z_mask = np.logical_and(z_data>=min_z, z_data<=max_z)
    zdr_mask = np.logical_and(zdr_data>=min_zdr, zdr_data<=max_zdr)
    rhv_mask = rhohv_data>min_rhohv
    h_mask = isom_data==0 #below melting level
    final_mask = z_mask & zdr_mask & rhv_mask & h_mask
    
    #collate z and zdr pairs
    alpha_dict['z_pairs'] = np.append(alpha_dict['z_pairs'] , z_data[final_mask])
    alpha_dict['zdr_pairs']  = np.append(alpha_dict['zdr_pairs'] , zdr_data[final_mask])
    
    #halt if insufficent number of pairs
    n_pairs = len(alpha_dict['z_pairs'])
    if n_pairs < pair_threshold:
        #update alpha timeseries
        if len(alpha_dict['alpha_ts'])>0:
            if verbose:
                print('insufficent pairs', n_pairs, '- Using previous alpha of', alpha_dict['alpha_ts'][-1])
            alpha_dict['alpha_ts'].append(alpha_dict['alpha_ts'][-1]) #update using last alpha
        else:
            if verbose:
                print('insufficent pairs', n_pairs, '- Using default alpha of', default_alpha)
            alpha_dict['alpha_ts'].append(default_alpha)#update to default alpha
        alpha_dict['dt_ts'].append(radar_starttime)
        return alpha_dict

    #find z-zdr slope
    if verbose:
        print(n_pairs, 'pairs found, finding Z-ZDR slope')
    K = _find_z_zdr_slope(alpha_dict)
    if verbose:
        print('slope value', K)

    #update alpha
    if K < z_zdr_slope_threshold:
        alpha = 0.049 - 0.75*K
    else:
        alpha = default_alpha
    if verbose:
        print('alpha value', alpha)

    #update timeseries
    alpha_dict['alpha_ts'].append(alpha)
    alpha_dict['dt_ts'].append(radar_starttime)
    #reset pairs
    alpha_dict['z_pairs'] = []
    alpha_dict['zdr_pairs'] = []
                           
    return alpha_dict


def retrieve_zphi(radar, alpha=0.015, beta=0.62, smooth_window_len=5, rhohv_edge_threshold=0.98, refl_edge_threshold=5,
         refl_field='reflectivity', phidp_field='corrected_differential_phase', rhohv_field='corrected_cross_correlation_ratio',
         hca_field='radar_echo_classification', isom_field='height_over_isom', ah_field='specific_attenuation'):
    
    """
    WHAT: Implementation of zphi technique for estimating specific attenuation from Ryzhkov et al.
    Adpated from pyart.
    
    INPUTS:
        radar: pyart radar object
        alpha: coefficent that is dependent on wavelength and DSD (float)
        beta: coefficent that's dependent on wavelength (float)
        smooth_window_len: used for calculating a moving average in the radial direction for the reflectivity field (int)
        rhohv_edge_threshold: threshold for detecting first and last gates used for total PIA calculation (float)
        refl_edge_threshold: threshold for detecting first and last gates used for total PIA calculation (float, dBZ)
        various field names
    
    OUTPUTS:
        radar: pyart radar object with specific attenuation field
    
    https://arm-doe.github.io/pyart/_modules/pyart/correct/attenuation.html#calculate_attenuation
    
    Ryzhkov et al. Potential Utilization of Specific Attenuation for Rainfall
    Estimation, Mitigation of Partial Beam Blockage, and Radar Networking,
    JAOT, 2014, 31, 599-619.
    """
    
    # extract fields and parameters from radar if they exist
    # reflectivity and differential phase must exist
    # create array to hold the output data
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data'].copy()
    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data'].copy()
    radar.check_field_exists(rhohv_field)
    rhohv = radar.fields[rhohv_field]['data'].copy()
    
    ah = np.ma.zeros(refl.shape, dtype='float64')
    
    #smooth reflectivity
    sm_refl = common.smooth_masked(refl, wind_len=smooth_window_len,
                            min_valid=1, wind_type='mean')
    radar.add_field_like(refl_field, 'sm_reflectivity', sm_refl, replace_existing=True)

    #load gatefilter
    gatefilter = pyart.correct.GateFilter(radar)
    
    #mask clutter
    pid = radar.fields[hca_field]['data']
    gatefilter.exclude_gates(np.ma.getmask(pid))
    #mask hail
    gatefilter.exclude_gates(pid==9)
    #mask data above melting level
    isom = radar.fields[isom_field]['data']
    gatefilter.exclude_gates(isom > 0)

    #create rhohv and z mask for determining r1 and r2
    edge_mask = np.logical_or(rhohv.filled(fill_value=0) < rhohv_edge_threshold, sm_refl.filled(fill_value=0) < refl_edge_threshold)

    #despeckle gatefilter (both foles and valid regions)
    valid_mask = gatefilter.gate_included
    valid_mask_filt = morphology.remove_small_holes(valid_mask, area_threshold=10)
    valid_mask_filt = morphology.remove_small_objects(valid_mask_filt, min_size=10)
    gatefilter.include_gates(valid_mask_filt)
                                                            
    #prepare phidp

    mask_phidp = np.ma.getmaskarray(phidp)
    mask_phidp = np.logical_and(mask_phidp, ~valid_mask_filt)
    corr_phidp = np.ma.masked_where(mask_phidp, phidp).filled(fill_value=0)
    
    #convert refl to z and gate spacing (in km)
    refl_linear = np.ma.power(10.0, 0.1 * beta * sm_refl).filled(fill_value=0)
    dr = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0

    #find end indicies in reject_mask
    end_gate_arr = np.zeros(radar.nrays, dtype='int32')
    start_gate_arr = np.zeros(radar.nrays, dtype='int32')
    #combine edge + gatefilter
    gate_mask = np.logical_and(gatefilter.gate_included, ~edge_mask)
    
    for ray in range(radar.nrays):
        ind_rng = np.where(gate_mask[ray, :] == 1)[0]
        if len(ind_rng) > 1:
            #CP2 experences invalid data in the first 5 gates. ignore these gates
            ind_rng = ind_rng[ind_rng>6]
        if len(ind_rng) > 1:
            # there are filtered gates: The last valid gate is one
            # before the first filter gate
            end_gate_arr[ray] = ind_rng[-1]-1 #ensures that index is -1 if all rays are masked
            start_gate_arr[ray] = ind_rng[0]
            
    for ray in range(radar.nrays):
        # perform attenuation calculation on a single ray

        # if number of valid range bins larger than smoothing window
        if end_gate_arr[ray]-start_gate_arr[ray] > smooth_window_len:
            # extract the ray's phase shift,
            # init. refl. correction and mask
            ray_phase_shift = corr_phidp[ray, start_gate_arr[ray]:end_gate_arr[ray]]
            ray_mask = valid_mask_filt[ray, start_gate_arr[ray]:end_gate_arr[ray]]
            ray_refl_linear = refl_linear[ray, start_gate_arr[ray]:end_gate_arr[ray]]

            # perform calculation if there is valid data
            last_six_good = np.where(np.ndarray.flatten(ray_mask) == 1)[0][-6:]
            if(len(last_six_good)) == 6:
                phidp_max = np.median(ray_phase_shift[last_six_good])
                self_cons_number = (
                    np.exp(0.23 * beta * alpha * phidp_max) - 1.0)
                I_indef = cumtrapz(0.46 * beta * dr * ray_refl_linear[::-1])
                I_indef = np.append(I_indef, I_indef[-1])[::-1]

                # set the specific attenutation and attenuation
                ah[ray, start_gate_arr[ray]:end_gate_arr[ray]] = (
                    ray_refl_linear * self_cons_number /
                    (I_indef[0] + self_cons_number * I_indef))
    #add ah into radar
    spec_at = pyart.config.get_metadata('specific_attenuation')
    ah_masked = np.ma.masked_where(gatefilter.gate_excluded, ah)
    spec_at['data'] = ah_masked
    spec_at['_FillValue'] = ah_masked.fill_value
    radar.add_field(ah_field, spec_at, replace_existing=True)
    
    return radar