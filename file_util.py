from glob import glob
import zipfile
import tempfile

import numpy as np
import netCDF4

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def unpack_zip(zip_ffn):
    """
    Unpacks zip file in temp directory

    Parameters:
    ===========
        zip_ffn: str
            Full filename to zip file 
            
    Returns:
    ========
        temp_dir: string
            Path to temp directory
    """    
    #build temp dir
    temp_dir = tempfile.mkdtemp()
    #unpack tar
    zip_fd = zipfile.ZipFile(zip_ffn)
    zip_fd.extractall(path=temp_dir)
    zip_fd.close()
    #list files
    file_list = sorted(glob(temp_dir + '/*'))
    return file_list

def write_rf_nc(rf_ffn, rid, valid_time, rain_rate, origin_lon, origin_lat, parallels):

    """
    WHAT: Write a RF3 instaneous rainrate file
    rain_rate grid must be a 512x512 array. With extent 128m.
    """
    
    #create RF3 dims
    x_bounds = np.linspace(-128,128,512)
    x_bounds = np.rot90(np.tile(x_bounds,(2,1)),3)
    y_bounds = x_bounds.copy()
    x = np.linspace(-127.75,127.75,512)
    y = x.copy()
    
    # Write data
    with netCDF4.Dataset(rf_ffn, 'w') as ncid:
        #create dimensions
        ncid.createDimension('n2', 2)
        ncid.createDimension("x", 512)
        ncid.createDimension("y", 512)

        #set global
        ncid.setncattr('licence', 'http://www.bom.gov.au/other/copyright.shtml')
        ncid.setncattr('source', 'dprain testing')
        ncid.setncattr('station_id', np.intc(rid))
        ncid.setncattr('institution', 'Commonwealth of Australia, Bureau of Meteorology (ABN 92 637 533 532)')
        ncid.setncattr('station_name', '')
        ncid.setncattr('Conventions', 'CF-1.7')
        ncid.setncattr('title', 'Radar derived rain rate')

        #create x/y bounds
        ncx_bounds = ncid.createVariable('x_bounds', np.float, ('x','n2'))
        ncy_bounds = ncid.createVariable('y_bounds', np.float, ('y','n2'))    
        ncx_bounds[:] = x_bounds
        ncy_bounds[:] = y_bounds

        #create x/y vars
        ncx = ncid.createVariable('x', np.float, ('x'))
        ncy = ncid.createVariable('y', np.float, ('y'))
        ncx[:] = x
        ncx.units = 'km'
        ncx.bounds = 'x_bounds'
        ncx.standard_name = 'projection_x_coordinate'
        ncy[:] = y
        ncy.units = 'km'
        ncy.bounds = 'y_bounds'
        ncy.standard_name = 'projection_y_coordinate'

        #create time var
        nct = ncid.createVariable('valid_time', np.int_)
        nct[:] = valid_time
        nct.long_name = 'Valid time'
        nct.standard_name = 'time'
        nct.units = 'seconds since 1970-01-01 00:00:00 UTC'
        
        #write rain rate
        scaled_rain = np.short(rain_rate/0.05)
        ncrain = ncid.createVariable('rain_rate', np.short, ("y", "x"), 
                                       fill_value=-1, chunksizes=(512,512))
        ncrain[:] = scaled_rain
        ncrain.grid_mapping = 'proj'
        ncrain.long_name = 'Rainfall rate'
        ncrain.standard_name = 'rainfall_rate'
        ncrain.units = 'mm hr-1'
        ncrain.scale_factor = 0.05
        ncrain.add_offset = 0.0
        
        #write proj
        ncproj = ncid.createVariable('proj', np.byte)
        ncproj[:] = 0
        ncproj.grid_mapping_name = 'albers_conical_equal_area'
        ncproj.false_easting = 0.0
        ncproj.false_northing = 0.0
        ncproj.semi_major_axis = 6378137.0
        ncproj.semi_minor_axis = 6356752.31414
        ncproj.longitude_of_central_meridian = origin_lon
        ncproj.latitude_of_projection_origin = origin_lat
        ncproj.standard_parallel = parallels