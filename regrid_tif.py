# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 08:55:18 2023

@author: Bryan.Eder
"""
import os
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
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from pyproj import CRS
from pathlib import Path


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

#TODO fix, not working
r'''
def reproject_resample_dataset(src_fp, dest_fp):
    with rasterio.open(dest_fp) as dst:
        with rasterio.open(src_fp) as src:
            with WarpedVRT(src, crs=dst.crs) as vrt:
                scale_factor_x = 1/(dst.res[0]/vrt.profile['transform'][0]) 
                scale_factor_y = 1/abs((dst.res[1]/vrt.profile['transform'][4]))
                print(vrt.profile)
                print(dst.res)
                array=(vrt.read(
                    1,
                    out_shape=(
                    1,
                    int(vrt.height * scale_factor_y),
                    int(vrt.width * scale_factor_x))))
                print(np.nanmean(array))
                print(array.shape)
                transform_vrt = vrt.profile['transform'] * vrt.profile['transform'].scale(
                    (vrt.width / array.shape[-1]),
                    (vrt.height / array.shape[-2]))  
                profile = vrt.profile
    return array, transform_vrt, profile
r'''

def reproject_dataset(in_fp, out_fp, dst_epsg):
    """

    Parameters
    ----------
    in_fp : str
        file path to geotiff dataset openable by rasterio
    out_fp : str
        file path to saved output file (.tif)
    dst_epsg : int
        EPSG code i.e. 4326

    Returns
    -------
    out_fp : str
        file path to saved output file (.tif)

    """
    with rasterio.open(in_fp) as src: 
        options = src.profile
        array = src.read(1)  
        transform_src = src.transform
        dst_crs=CRS.from_epsg(dst_epsg)
        width=array.shape[-1]
        height=array.shape[-2] 
        reprojected_transform, reprojected_width, reprojected_height = calculate_default_transform(
                src.crs, dst_crs, width, height, *src.bounds)
        
        options.update({'width': reprojected_width, 
                         'height': reprojected_height,
                         'crs': dst_crs,
                         'transform': reprojected_transform,
                         'compress':'lzw',
                         'tiled': "true",
                         'count': 1,
                         'dtype': np.float32})
        
        with rasterio.open(out_fp, 'w', **options) as dst:
            reproject(source=array,
                        destination=rasterio.band(dst, 1),
                        src_transform=transform_src,
                        src_crs=src.crs,
                        dst_transform=reprojected_transform,
                        dst_crs=dst_crs,
                        dst_nodata=options['nodata'],
                        src_nodata=options['nodata'],
                        resampling=Resampling.nearest)
            factors = [2, 4, 8, 16, 32, 64]
            dst.build_overviews(factors, Resampling.nearest)
        return out_fp

def resample_dataset(src_fp, dest_fp, out_fp):
    with rasterio.open(src_fp) as src:
        with rasterio.open(dest_fp) as dst:
            options = dst.profile
            scale_factor_x = 1/(dst.res[0]/src.profile['transform'][0]) 
            scale_factor_y = 1/abs((dst.res[1]/src.profile['transform'][4]))

        # resample data to target shape
        data = src.read(
            out_shape=(
                src.count,
                int(src.height * scale_factor_y),
                int(src.width * scale_factor_x)
            ),
            resampling=Resampling.bilinear #nearest
        )
    
        # scale image transform
        transform_scaled = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )
     
    options.update({'width': data.shape[-1], 
                     'height': data.shape[-2],
                     'nodata': -999999.,
                     'transform': transform_scaled})
    
    with rasterio.open(out_fp, 'w', **options) as out_dst:
        out_dst.write(data.reshape(options['height'],options['width']), 1)
        
    return data, transform_scaled, options

def create_latlon_grid(array, transform_out):
    width = array.shape[-1]
    height = array.shape[-2]
    cols,rows = da.meshgrid(da.arange(width),da.arange(height))
    xs, ys = rasterio.transform.xy(transform_out, rows.compute(), cols.compute())
    lons=np.array(xs)
    lats=np.array(ys)
    return lats,lons

def regrid_dataset(src_lats, src_lons, dest_lats, 
                   dest_lons, src_array, method='linear'):
    # Create tuple of source points
    src_points = np.column_stack((src_lats.ravel(), src_lons.ravel()))
    
    # Interpolate to destination points
    regridded_array = griddata(points=src_points, values=src_array.ravel(), 
                               xi=(dest_lats, dest_lons), method=method)
    
    # Print min and max values
    print(f"Min value: {np.nanmin(regridded_array)}")
    print(f"Max value: {np.nanmax(regridded_array)}")
    
    return regridded_array

def preprocess_data(array, dataset_profile, negative_vals=True):

    if negative_vals:
        bad_idx = da.where(da.logical_or(array.ravel()==dataset_profile['nodata'], array.ravel()>0))
    else:
        bad_idx = da.where(array.ravel()==dataset_profile['nodata'])

    array.ravel()[bad_idx]=np.float32(np.nan)
    array=da.absolute(array)
    if np.all(np.isnan(array)):
        raise ValueError('All nan array created during preprocessing')
    print(np.nanmin(array), np.nanmax(array))
    return array


def main(in_fp, dest_fp, out_reproj_fp, out_resample_fp, out_fp, negative_vals=True, write_tif=True):
    
    """

    Parameters
    ----------
    in_fp : str
        file path to input GeoTIFF to be reprojected, resampled, and regridded 
        to match a destination dataset 
    dest_fp : str
        file path to GeoTIFF destination dataset
    out_reproj_fp : str
        file path for interim product (reprojected file)
    out_resample_fp : str
        file path for interim product (reprojected and resampled file)
    out_fp : str
        file path for regridded .npy file
    negative_vals : bool
        set to true if input GeoTIFF (in_fp) contains negative values for bathymetry 
        and positive for land otherwise set to false
    write_tif : bool
        determines if a GeoTIFF is created for the regridded array
    Returns
    -------
    None

    """
    
    with rasterio.open(dest_fp) as dst:
        dest_array = dst.read(1)
        dest_array.ravel().shape
        dest_transform = dst.transform
        dest_profile = dst.profile
        dst_epsg = dst.crs.to_epsg()

    #array, transform_out, profile = reproject_resample_dataset(src_fp, dest_fp)
    
    out_reproj_fp = reproject_dataset(in_fp, out_reproj_fp, dst_epsg)
    
    array, transform_out, profile = resample_dataset(out_reproj_fp, dest_fp, out_resample_fp)
    
    #TODO array.copy?
    src_array = preprocess_data(array, profile, negative_vals=negative_vals)
    
    src_lats, src_lons = create_latlon_grid(src_array, transform_out)
    
    dest_lats, dest_lons = create_latlon_grid(dest_array, dest_transform)
    
    regridded_array = regrid_dataset(src_lats, src_lons, dest_lats, 
                       dest_lons, src_array, method='linear')

    with open(out_fp, 'wb') as f:
        np.save(f, regridded_array)
    
    if write_tif:
        
        options = {'driver': 'GTiff', 
                   'width': dest_profile['width'], 
                   'height': dest_profile['height'],
                   'crs': dest_profile['crs'],
                   'nodata': np.nan,
                   'transform': dest_transform,
                   'compress':'lzw',
                   'tiled': "true",
                   'count': 1,
                   'dtype': np.float32}
        
        file_path = Path(out_fp)
        out_tif_fp = file_path.with_name(file_path.stem + ".tif").resolve()

        with rasterio.open(out_tif_fp, 'w', **options) as dst:
                dst.write(regridded_array.reshape(options['height'],options['width']), 1)    
                
if __name__ == "__main__":
    in_fp = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\Large\Job780732_2019_NGS_FL_topobathy_DEM_Irma.tif"
    dest_fp = r"P:\_RSD\Data\ETOPO\SDB\KeyLargo_SDB_Output\KeyLargo_LinearRegression_SDB.tif"
    out_reproj_fp = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\KeyLargo_lidar_reproj.tif"
    out_resample_fp = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\KeyLargo_lidar_resample.tif"
    out_fp = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\KeyLargo_lidar.npy"
    
    main(in_fp, dest_fp, out_reproj_fp, out_resample_fp, out_fp, negative_vals=False)
               


