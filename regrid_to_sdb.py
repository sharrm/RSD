# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:55:18 2023

@author: Bryan.Eder
"""

import numpy as np
import dask.array as da
import glob
from scipy.interpolate import griddata
import rasterio
import xarray as xr
from shapely.geometry import box
import pyproj
from shapely.ops import transform
from rasterio.vrt import WarpedVRT


r'''
def test_dataset_overlap(src_fp, dest_fp): 

    #TODO include specified AOI for specific test area

    with xr.open_dataset(src_fp,engine="rasterio") as src:
        with xr.open_dataset(dest_fp,engine="rasterio") as dst:
            reproject = pyproj.Transformer.from_crs(
                src.rio.crs.to_string(),
                dst.rio.crs.to_string(),
                always_xy=True
            )
            return transform(reproject.transform, box(*src.rio.bounds())).intersects(box(*dst.rio.bounds()))

def filter_datasets(in_dir, dest_fp):
    datasets = glob.glob(f'{in_dir}/*.tif')
    for i,f in enumerate(datasets):
        if not test_dataset_overlap(f, dest_fp):
            datasets.pop(i)
    return datasets
r'''

def reproject_resample_dataset(src_fp, dest_fp):

    with rasterio.open(dest_fp) as dst:
        with rasterio.open(src_fp) as src:
            with WarpedVRT(src, crs=dst.crs) as vrt:
                #TODO will this work with upscaling too? dest_res = 10 for S2
                scale_factor = 1/(dst.res[0]/vrt.profile['transform'][0]) 
                array=(vrt.read(
                    1,
                    out_shape=(
                    1,
                    int(vrt.height * scale_factor),
                    int(vrt.width * scale_factor))))
                transform_out = vrt.profile['transform'] * vrt.profile['transform'].scale(
                    (vrt.width / array.shape[-1]),
                    (vrt.height / array.shape[-2]))  
                profile = vrt.profile
    return array, transform_out, profile

def create_latlon_grid(array, transform_out):
    width = array.shape[-1]
    height = array.shape[-2]
    cols,rows = da.meshgrid(da.arange(width),da.arange(height))
    xs, ys = rasterio.transform.xy(transform_out, rows.compute(), cols.compute()) # changed transform to transform_out
    lons=np.array(xs)
    lats=np.array(ys)
    return lats,lons

def regrid_dataset(src_lats, src_lons, dest_lats, 
                   dest_lons, src_array, method='linear'):
    # Create tuple of source points
    src_points = np.column_stack((src_lats.ravel(), src_lons.ravel()))
    
    # Interpolate to destination points
    regridded_array = griddata(points=src_points, values=src_array.ravel(), xi=(dest_lats, dest_lons), method=method)
    
    # Print min and max values
    print(f"Min value: {np.nanmin(regridded_array)}")
    print(f"Max value: {np.nanmax(regridded_array)}")
    return regridded_array

def preprocess_src_data(array, dataset_profile, negative_vals=True):

    if negative_vals:
        bad_idx = da.where(da.logical_or(array.ravel()==dataset_profile['nodata'], array.ravel()>0))
    else:
        bad_idx = da.where(da.logical_or(array.ravel()==dataset_profile['nodata']))

    array.ravel()[bad_idx]=np.float32(np.nan)
    array=da.absolute(array)
    return array


def main(src_fp, dest_fp, out_fp):
    
    with rasterio.open(dest_fp) as src:
        dest_array = src.read(1)
        dest_transform = src.transform
    
    array, transform_out, profile = reproject_resample_dataset(src_fp, dest_fp)
    
    src_array = preprocess_src_data(array, profile, negative_vals=True)
    
    src_lats, src_lons = create_latlon_grid(src_array, transform_out)
    
    dest_lats, dest_lons = create_latlon_grid(dest_array, dest_transform)
    
    regridded_array = regrid_dataset(src_lats, src_lons, dest_lats, 
                       dest_lons, src_array, method='linear')

    with open(out_fp, 'wb') as f:
        np.save(f, regridded_array)
        
    
        
    return out_fp

# %% - input


if __name__ == '__main__':
    match_tif = r"P:\_RSD\Data\ETOPO\SDB\KeyLargo_SDB_Output\KeyLargo_RANSACRegressor_SDB.tif"
    tif = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\2019_NGS_FL_topobathy_DEM_Irma_Job778026\Job778026_2019_NGS_FL_topobathy_DEM_Irma.tif"
    out_tif = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\KeyLargo_lidar.npy"
    
    main(tif, match_tif, out_tif)