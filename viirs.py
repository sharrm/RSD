# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 09:06:29 2023

@author: sharrm

Created to retrieve VIIRS kd_490 from https://coastwatch.pfeg.noaa.gov/erddap/index.html
Tutorial: https://github.com/coastwatch-training/CoastWatch-Tutorials/blob/main/Tutorial1-basics/Python/Tutorial1-basics.ipynb
East coast: https://eastcoast.coastwatch.noaa.gov/cw_viirs_k490.php
UMD example: https://umd.instructure.com/courses/1336575/pages/color-enhancement-images?module_item_id=11943531
"""

import os
import urllib.request
import xarray as xr
from matplotlib import pyplot as plt
import warnings
import cartopy.crs as ccrs
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio import Affine
import cartopy.feature as cfeature
from scipy.ndimage.morphology import binary_dilation
from scipy.interpolate import griddata
from scipy.ndimage import grey_dilation, grey_closing
import numpy as np


# %% - retrieve, plot, manipulate Kd490

# read data
def retrieve_data(url):
    # retrieve Kd490
    out_kd490 = os.path.join(r'P:\_RSD\Data\VIIRS', "kd_490_epsg4326.tif")
    urllib.request.urlretrieve(url, out_kd490)
    
    print(f'Downloaded {os.path.basename(out_kd490)} from {url.split(".geo")[0]}')
    
    return out_kd490

def dilate_kd490(out_kd490):
    # Open the float GeoTIFF file
    with rasterio.open(out_kd490) as src:
        data = src.read(1)  # Read the data as a NumPy array
        transform = src.transform
        
        # Define a dilation factor (adjust as needed)
        dilation_factor = 2
    
        # Apply dilation operation
        dilated_data = grey_closing(data, size=(dilation_factor, dilation_factor), mode='nearest')
        
        out_trans = Affine(transform[0], 0, transform[2] + transform[0], 0, transform[4], transform[5])
        
        out_dilated = os.path.join(r'P:\_RSD\Data\VIIRS', 'dilated_epsg4326.tif')
        
        # Save the dilated GeoTIFF
        with rasterio.open(out_dilated, 'w', driver='GTiff', 
                           height=data.shape[0], width=data.shape[1], 
                           count=1, dtype=data.dtype, crs=src.crs, transform=out_trans) as dst:
            dst.write(dilated_data, 1)
            
    print(f'Saved dilated and shifted version to {out_dilated}')
    
    return out_dilated

# plot
# def plot_kd490(data, transform, ht, wd):
#     # Create a figure and axis with a specified projection
#     fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
#     # Plot the data
#     im = ax.imshow(data, extent=(transform[2] + 0.075, transform[2] + 0.075 + transform[0]*wd, 
#                                  transform[5] + transform[4]*ht, transform[5]),
#                    origin='upper', cmap='viridis')
    
#     # Add coastline
#     ax.coastlines(resolution='10m', color='black')
    
#     # Add features like country borders, rivers, etc. (optional)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     ax.add_feature(cfeature.RIVERS)
    
#     # Add colorbar
#     cbar = plt.colorbar(im, ax=ax, shrink=0.5)
#     cbar.set_label('Data Value')
    
#     # Set title and show plot
#     plt.title('GeoTIFF with Coastline')
#     plt.show()

#     return None

# %% - reproject

# def save_xarray_to_geotiff_utm(ds, kd_490, filename, data_variable):
    # Get the coordinate values and resolution
    xmin = ds.geospatial_lon_min
    ymax = ds.geospatial_lat_max
    xres = ds.geospatial_lon_resolution
    yres = ds.geospatial_lat_resolution
    transform = from_origin(xmin, ymax, xres, yres)
    
    # Create the raster profile with UTM CRS
    profile = {
        'driver': 'GTiff',
        'height': ds.sizes['latitude'],
        'width': ds.sizes['longitude'],
        'count': 1,
        'dtype': str(ds[data_variable].dtype),
        'crs': CRS.from_epsg(4326),
        'transform': transform,
        'nodata': 0
    }
    
    # Write the data array to a GeoTIFF file
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(kd_490, 1)
        
    dst = None
    
    print(f'Saved Kd490 data to: {filename}')
        
    return filename

# get UTM zone of input area
def get_crs(in_crs):
    with rasterio.open(in_crs) as band:
        utm_crs = band.crs
        
    band = None
    
    return utm_crs

# reproject to UTM
def reproject_viirs(tiff, in_crs):
    # https://rasterio.readthedocs.io/en/latest/topics/reproject.html#estimating-optimal-output-shape
    
    dst_crs = get_crs(in_crs)
    
    with rasterio.open(tiff) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        print(f'Updating {tiff} from {src.crs.data["init"]} to {dst_crs.data["init"]}')
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
           
        reprojected_name = tiff.replace('_epsg4326.tif', '_epsg' + str(dst_crs.data['init'].split(':')[1]) + '.tif')
        
        with rasterio.open(reprojected_name, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
                
    print(f'Saved reprojected version to: {reprojected_name}')


# %% - main

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    url = ''.join(['https://coastwatch.pfeg.noaa.gov/erddap/griddap/nesdisVHNSQkd490Daily.geotif?',
                   'kd_490',
                   '%5B(2023-09-12T12:00:00Z):1:(2023-09-12T12:00:00Z)%5D',
                   '%5B(0.0):1:(0.0)%5D',
                   '%5B(36.0):1:(38.0)%5D%5B(-75.0):1:(-77.0)%5D'
                   ])
    
    in_crs = r"P:\_RSD\Data\Imagery\_turbidTesting_rhos\Cheapeake_20230410\_RGB\Cheapeake_20230410_rgb_composite.tif"
    
    out_kd490 = retrieve_data(url)
    out_dilated = dilate_kd490(out_kd490)
    # plot_kd490(arr, trans, ht, wd)
    reproject_viirs(out_dilated, in_crs)

