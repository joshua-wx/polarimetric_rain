from sklearn.linear_model import LinearRegression
import cftime
import numpy as np
from matplotlib import pyplot as plt

import pyart



def _find_z_zdr_slope(alpha_dict):
    
    plot_fits=True
    
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
    
def main(radar, alpha_dict):
        
    verbose = True
        
    #limits
    min_z = 20.
    max_z = 50.
    min_zdr = -4.
    max_zdr = 4.
    min_rhv = 0.98
    pair_limit = 10000
    
    #get radar time
    radar_starttime = cftime.num2pydate(radar.time['data'][0], radar.time['units'])
    
    #extract data
    z_data = radar.get_field(0, 'reflectivity')
    zdr_data = radar.get_field(0, 'corrected_differential_reflectivity')
    rhv_data = radar.get_field(0, 'corrected_cross_correlation_ratio')
    isom_data = radar.get_field(0, 'height_over_isom')
    
    
    #build masks
    z_mask = np.logical_and(z_data>=min_z, z_data<=max_z)
    zdr_mask = np.logical_and(zdr_data>=min_zdr, zdr_data<=max_zdr)
    rhv_mask = rhv_data>min_rhv
    h_mask = isom_data<0
    final_mask = z_mask & zdr_mask & rhv_mask & h_mask
    
    #collate z and zdr pairs
    alpha_dict['z_pairs'] = np.append(alpha_dict['z_pairs'] , z_data[final_mask])
    alpha_dict['zdr_pairs']  = np.append(alpha_dict['zdr_pairs'] , zdr_data[final_mask])
    
    #halt if insufficent number of pairs
    n_pairs = len(alpha_dict['z_pairs'])
    if n_pairs < pair_limit:
        if verbose:
            print('insufficent pairs', n_pairs, 'using previous alpha of', alpha_dict['alpha_ts'][-1])
        #update alpha timeseries
        if len(alpha_dict['alpha_ts'])>0:
            alpha_dict['alpha_ts'].append(alpha_dict['alpha_ts'][-1]) #update using last alpha
        else:
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
    if K < 0.045:
        alpha = 0.049 - 0.75*K
    else:
        alpha = 0.015
    if verbose:
        print('alpha value', alpha)
    
    #update timeseries
    alpha_dict['alpha_ts'].append(alpha)
    alpha_dict['dt_ts'].append(radar_starttime)
    #reset pairs
    alpha_dict['z_pairs'] = []
    alpha_dict['zdr_pairs'] = []
                           
    return alpha_dict