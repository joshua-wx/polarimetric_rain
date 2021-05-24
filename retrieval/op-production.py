import os
import warnings
import argparse
import traceback
from datetime import datetime, timedelta

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

import matplotlib
matplotlib.use('agg')

import numpy as np
import pyart
from matplotlib import pyplot as plt
import cftime

import radar_codes
import filtering
import phase
import hydrometeors
import attenuation
import rainrate
import file_util
import gridding

import dask
import dask.bag as db
import gc

def daterange(date1, date2):
    """
    Generate date list between dates
    """
    date_list = []
    for n in range(int ((date2 - date1).days)+1):
        date_list.append(date1 + timedelta(n))
    return date_list

def buffer(vol_ffn):
    try:
        torrentfields(vol_ffn)
    except Exception as e:
        print('failed on', vol_ffn,'with',e)
        
        
def torrentfields(vol_ffn):

    print('processing', vol_ffn)

    #read radar volume
    radar = pyart.aux_io.read_odim_h5(vol_ffn, file_field_names=True)
    #get time from middle of first sweep
    start_ray_idx = radar.get_start(0)
    end_ray_idx   = radar.get_start(1)
    start_time = cftime.num2pydate(radar.time['data'][start_ray_idx], radar.time['units'])
    end_time = cftime.num2pydate(radar.time['data'][end_ray_idx], radar.time['units'])
    valid_time = start_time + (end_time-start_time)/2
    date_str = valid_time.strftime('%Y%m%d')
    
    #get radar band
    wavelength = radar_codes.get_wavelength(vol_ffn)
    if wavelength<8:
        band = 'C'
    else:
        band = 'S'
    if VERBOSE:
        print('band', band)
        
        
    #build output filenames
    cf_path = f'{cf_root}/{RADAR_ID:02}/{date_str}'
    cf_fn = f'{RADAR_ID:02}_{valid_time.strftime("%Y%m%d_%H%M%S")}.vol.nc' #this filename should match
    cf_ffn = f'{cf_path}/{cf_fn}'
    rf_path = f'{rf_root}/{RADAR_ID:02}/{date_str}'
    rf_fn = f'{RADAR_ID}_{valid_time.strftime("%Y%m%d_%H%M%S")}.prcp-rrate.nc' #this filename should match
    rf_ffn = f'{rf_path}/{rf_fn}'
    img_path = f'{img_root}/{RADAR_ID:02}/{date_str}'
    img_fn = f'{RADAR_ID:02}_{valid_time.strftime("%Y%m%d_%H%M%S")}.jpg' #this filename should match
    img_ffn = f'{img_path}/{img_fn}'
    #check if last file to be created (img_ffn) and SKIP_EXISTING are true
    if os.path.isfile(img_ffn) and SKIP_EXISTING:
        print('skipping:', vol_ffn, 'already processed and skipping enabled')
        return None
    ##################################################################################################
    #
    # Preprocessing
    #
    ##################################################################################################
    if VERBOSE2:
        print('Corrections')
    # Correct RHOHV
    rho_corr = radar_codes.correct_rhohv(radar, snr_name='SNRH')
    radar.add_field_like('RHOHV', 'RHOHV_CORR', rho_corr, replace_existing=True)

    # Correct ZDR
    corr_zdr = radar_codes.correct_zdr(radar, snr_name='SNRH')
    radar.add_field_like('ZDR', 'ZDR_CORR', corr_zdr, replace_existing=True)

#     from importlib import reload
#     reload(radar_codes)
    
    if VERBOSE2:
        print('Temperature Profile')
    # Temperature    
    height, temperature, isom = radar_codes.temperature_profile_access(radar)
    radar.add_field('temperature', temperature, replace_existing=True)
    radar.add_field('height', height, replace_existing=True)
    radar.add_field('height_over_isom', isom, replace_existing=True)

    if VERBOSE2:
        print('Gatefilter')
    # GateFilter
    gatefilter = filtering.do_gatefilter(radar,
                                         refl_name='DBZH',
                                         phidp_name="PHIDP",
                                         rhohv_name='RHOHV_CORR',
                                         zdr_name="ZDR_CORR",
                                         snr_name='SNRH')
    
    if VERBOSE2:
        print('PHIDP processing')
    # phidp filtering
    rawphi = radar.fields["PHIDP"]['data']
    sysphase = pyart.correct.phase_proc.det_sys_phase(radar, ncp_field="RHOHV", rhv_field="RHOHV", phidp_field="PHIDP", rhohv_lev=0.95)
    radar.add_field_like("PHIDP", "PHIDP_offset", rawphi - sysphase, replace_existing=True)
    print('system phase:', round(sysphase))
    #unfold phase
    unfphidic = pyart.correct.dealias_unwrap_phase(radar,
                                               gatefilter=gatefilter,
                                               skip_checks=True,
                                               vel_field='PHIDP_offset',
                                               nyquist_vel=180)
    radar.add_field_like("PHIDP", "PHIDP_unwraped", unfphidic['data'], replace_existing=True)

    #correct phase
    #calculate phidp from bringi technique
    phidp_b, kdp_b = phase.phidp_bringi(radar, gatefilter, phidp_field="PHIDP_unwraped", refl_field='DBZH')
    radar.add_field("PHIDP_B", phidp_b, replace_existing=True)
    radar.add_field('KDP_B', kdp_b, replace_existing=True)
    #add metadata on system phase
    radar.fields['PHIDP_B']['system_phase'] = round(sysphase)
    kdp_field_name = 'KDP_B'
    phidp_field_name = 'PHIDP_B'

    if VERBOSE2:
        print('HCA processing')
    
    #first try to use exisiting NCAR HCA into a field. Sometimes it is missing due a missing DP fields.
    try:
        hca_field = hydrometeors.extract_ncar_pid(radar, vol_ffn)
    except:
        #insert CSU HCA if NCAR PID is not in the file
        hca_field = hydrometeors.csu_hca(radar,
                                          gatefilter,
                                          kdp_name=kdp_field_name,
                                          zdr_name='ZDR_CORR',
                                          rhohv_name='RHOHV_CORR',
                                          refl_name='DBZH',
                                          band=band)
    radar.add_field('radar_echo_classification', hca_field, replace_existing=True)
    
    ##################################################################################################
    #
    # Retrievals
    #
    ##################################################################################################
#     from importlib import reload
#     reload(rainrate)
#     reload(attenuation)
    if VERBOSE2:
        print('QPE Estimate')
    #index index of lowest sweep
    first_idx = np.argmin(radar.fixed_angle['data'])
    #estimate alpha
    alpha, alpha_method = attenuation.estimate_alpha_zhang2020(radar, band, first_idx,
                                           refl_field='DBZH', zdr_field='ZDR_CORR', rhohv_field='RHOHV_CORR',
                                           verbose=VERBOSE)
    if VERBOSE2:
        print('QPE ZPHI')
    #estimate specific attenuation
    radar = attenuation.retrieve_zphi(radar, band, alpha=alpha, alpha_method=alpha_method,
                                     refl_field='DBZH', phidp_field=phidp_field_name, rhohv_field='RHOHV_CORR')
    if VERBOSE2:
        print('QPE Retrieve')
    if RADAR_ID == 2:
        ah_coeff_fitted = False #use default for Melbourne
    else:
        ah_coeff_fitted = True #otherwise use fits for Darwin
    #estimate rainfall
    radar = rainrate.conventional(radar, alpha=92, beta=1.7, refl_field='DBZH')
    radar = rainrate.polarimetric(radar, band, refl_field='corrected_reflectivity',
                                  kdp_field=kdp_field_name, phidp_field=phidp_field_name, rhohv_field='RHOHV_CORR',
                                  ah_coeff_fitted=ah_coeff_fitted)
    
    ##################################################################################################
    #
    # Write outputs CF Radial
    #
    ##################################################################################################
    
    #write to cf output
    if VERBOSE2:
        print('Save CFradial')
    #create paths
    if not os.path.exists(cf_path):
        os.makedirs(cf_path)
    if os.path.exists(cf_ffn):
        print('cfradial of same name found, removing')
        os.system('rm -f ' + cf_ffn)
    #write to cf
    pyart.io.write_cfradial(cf_ffn, radar)
    
    ##################################################################################################
    #
    # Create and write grid
    #
    ##################################################################################################
        
    # grid first two sweeps (second sweep used as a fallback where the lower grid has no data)
    sort_idx = np.argsort(radar.fixed_angle['data'])
    data_sweep1 = radar.get_field(sort_idx[0], 'hybrid_rainrate', copy=True).filled(np.nan)
    data_sweep2 = radar.get_field(sort_idx[1], 'hybrid_rainrate', copy=True).filled(np.nan)
    data_combined = np.nanmax(np.stack((data_sweep1, data_sweep2), axis=2), axis=2)
    
    
    #build metadata and grid
    r = radar.range['data']
    th = 450 - radar.get_azimuth(sort_idx[0], copy=False)
    th[th < 0] += 360
    R, A = np.meshgrid(r, th)
    x = R * np.cos(np.pi * A / 180)
    y = R * np.sin(np.pi * A / 180)
    xgrid = np.linspace(-127750,127750,512)
    xgrid, ygrid = np.meshgrid(xgrid, xgrid)
    rain_grid_2d = gridding.KDtree_nn_interp(data_combined, x, y, xgrid, ygrid, nnearest = 16, maxdist = 2500)
    #gatespacing = r[1]-r[0]
    #rain_grid_2d = gridding.grid_data(data_combined, x, y, xgrid, xgrid, gatespacing=gatespacing)
        
    #extract metadata for RF3 grids
    standard_lat_1, standard_lat_2 = file_util.rf_standard_parallel_lookup(RADAR_ID)
    if standard_lat_1 is None:
        print('failed to lookup standard parallels')
    rf_lon0, rf_lat0 = file_util.rf_grid_centre_lookup(RADAR_ID)
    if rf_lon0 is None:
        print('failed to lookup rf grid centre coordinates')
    
    #create paths
    if not os.path.exists(rf_path):
        os.makedirs(rf_path)
    if os.path.exists(rf_ffn):
        print('rf3 of same name found, removing')
        os.system('rm -f ' + rf_ffn)
    #write to nc
    file_util.write_rf_nc(rf_ffn, RADAR_ID, valid_time.timestamp(), rain_grid_2d, rf_lon0, rf_lat0, (standard_lat_1, standard_lat_2))
    
    #create image and save to file
    ###################################################################################################################
    tilt = sort_idx[0] #first tilt
    ylim = [-128, 128]
    xlim = [-128, 128]

    fig = plt.figure(figsize=[16,8])
    display = pyart.graph.RadarDisplay(radar)

    ax = plt.subplot(231)
    display.plot_ppi('DBZH', tilt, vmin=0, vmax=60, cmap='pyart_HomeyerRainbow')
    ax.set_xlabel('')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(232)
    display.plot_ppi(kdp_field_name, tilt, vmin=0, vmax=6, cmap='pyart_HomeyerRainbow')
    ax.set_xlabel('')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(233)
    display.plot_ppi(phidp_field_name, tilt, vmin=0, vmax=90, cmap='pyart_Wild25')
    ax.set_xlabel('')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(234)
    display.plot_ppi('zr_rainrate', tilt, vmin=0.2, vmax=75, cmap='pyart_RRate11', title='SP retrieval')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(235)
    display.plot_ppi('hybrid_rainrate', tilt, vmin=0.2, vmax=75, cmap='pyart_RRate11', title=f'DP retrieval method: {alpha_method} with alpha: {alpha:.3f}')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = plt.subplot(236)
    ax.set_title(f'RF3 grid of DP retrieval')
    rain_grid_2d_plotting = rain_grid_2d.copy()
    rain_grid_2d_plotting[rain_grid_2d_plotting<=0] = np.nan
    img = plt.imshow(np.flipud(rain_grid_2d_plotting), vmin=0.2, vmax=75, cmap=pyart.graph.cm._generate_cmap('RRate11',100))
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('mm/hr')
    
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if os.path.exists(img_ffn):
        print('image of same name found, removing')
        os.system('rm -f ' + img_ffn)
    plt.savefig(img_ffn, dpi=100)
    fig.clf()
    plt.close()
    ##################################################################################################################
    #clean up
    del radar
    del rain_grid_2d
    gc.collect()
    
def manager(date_str):

    #unpack and list daily zip
    vol_zip = f'{vol_root}/{RADAR_ID:02}/{date_str[0:4]}/vol/{RADAR_ID:02}_{date_str}.pvol.zip'
    temp_dir = True
    vol_ffn_list = file_util.unpack_zip(vol_zip)
    
    for arg_slice in file_util.chunks(vol_ffn_list, NCPU):
        with ProcessPool() as pool:
            future = pool.map(buffer, arg_slice, timeout=360)
            iterator = future.result()
            while True:
                try:
                    _ = next(iterator)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print("function took longer than %d seconds" % error.args[1])
                except ProcessExpired as error:
                    print("%s. Exit code: %d" % (error, error.exitcode))
                except TypeError as error:
                    print("%s. Exit code: %d" % (error, error.exitcode))
                except Exception:
                    traceback.print_exc()
            
    
#     import time    
#     for vol_ffn in vol_ffn_list:
#         start = time.time()
#         torrentfields(vol_ffn)
#         end = time.time()
#         print('timer', end - start)

#     #run retrieval
#     i            = 0
#     n_files      = len(vol_ffn_list)   
#     for flist_chunk in file_util.chunks(vol_ffn_list, NCPU): #CUSTOM RANGE USED
#         bag = db.from_sequence(flist_chunk).map(buffer)
#         _ = bag.compute()
#         i += NCPU
#         del bag
#         print('processed: ' + str(round(i/n_files*100,2)))
        
    #clean up
    temp_vol_dir = os.path.dirname(vol_ffn_list[0])
    if '/tmp' in temp_vol_dir:
        os.system('rm -rf ' + temp_vol_dir)
        
def main():
    
    #build list of dates for manager
    dt_list = daterange(DT1, DT2)
    
    for dt in dt_list:
        date_str = dt.strftime('%Y%m%d')
        manager(date_str)
    
    
if __name__ == '__main__':
    """
    Global vars
    """    
    #config
    vol_root = '/g/data/rq0/level_1/odim_pvol'
    cf_root = '/g/data/rq0/admin/dprain-verify-latest/cfradial'
    rf_root = '/g/data/rq0/admin/dprain-verify-latest/rfgrid'
    img_root = '/g/data/rq0/admin/dprain-verify-latest/img'
    VERBOSE = False
    VERBOSE2 = False #more info
    SKIP_EXISTING = False
    
    # Parse arguments
    parser_description = "DP rainfall retrieval"
    parser = argparse.ArgumentParser(description = parser_description)
    parser.add_argument(
        '-j',
        '--cpu',
        dest='ncpu',
        default=16,
        type=int,
        help='Number of process')
    parser.add_argument(
        '-d1',
        '--date1',
        dest='date1',
        default=None,
        type=str,
        help='starting date to process from archive',
        required=True)
    parser.add_argument(
        '-d2',
        '--date2',
        dest='date2',
        default=None,
        type=str,
        help='starting date to process from archive',
        required=True)
    parser.add_argument(
        '-r',
        '--rid',
        dest='rid',
        default=None,
        type=int,
        help='Radar ID',
        required=True)
    
    args = parser.parse_args()
    NCPU         = args.ncpu
    RADAR_ID     = args.rid
    DATE1_STR    = args.date1
    DATE2_STR    = args.date2
    DT1          = datetime.strptime(DATE1_STR,'%Y%m%d')
    DT2          = datetime.strptime(DATE2_STR,'%Y%m%d')
    
    with warnings.catch_warnings():
        # Just ignoring warning messages.
        warnings.simplefilter("ignore")
        main()
        
    #%run op-production.py -r 2 -d1 20181120 -d2 20181120 -j 8