# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:29:34 2023

@author: sharrm

https://www.ncei.noaa.gov/products/etopo-global-relief-model

Used to ensure exact overlap between two raster data sets in spatial resolution,
projection, and spatial extent.

"""

import affine
import matplotlib.pyplot as plt
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT


# %% - globals

# interpolation method
resampling_method = Resampling.bilinear


# %% - based on the 'match_tif' extents, will mask, resample, and reproject the 'tif' so the two spatially overlap
# based on https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html
# should not need to modify anything below

def resample(tif, match_tif, out_etopo):
                                                        
    with rasterio.open(match_tif, 'r') as dest:
        dst_crs = dest.crs
        dst_bounds = dest.bounds
        dst_height = dest.height
        dst_width = dest.width
        dst_nodata = dest.nodata
        dst_arr = dest.read()
        options = dest.meta
      
    dest = None
    
    print(f'Using {match_tif} extents...')
    print(f'Matching Shape: {dst_arr.shape}')
      
    # Output image transform based on input match tif
    left, bottom, right, top = dst_bounds
    xres = (right - left) / dst_width
    yres = (top - bottom) / dst_height
    dst_transform = affine.Affine(xres, 0.0, left,
                                  0.0, -yres, top)
    
    vrt_options = {
        'resampling': resampling_method, 
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
        'nodata': dst_nodata
    }
    
    print(f'\nModifying {tif} to overlap...')
    
    with rasterio.open(tif) as src:
        src_height = src.height
        src_width = src.width 
        
        print(f'Input Shape: ({src.height}, {src.width})')
    
        with WarpedVRT(src, **vrt_options) as vrt:
            vrt_height = vrt.height
            vrt_width = vrt.width  
    
            # read all data into memory and mask to match_tif extents
            data = vrt.read()
            data = np.where(dst_arr == dst_nodata, 0, data)
            
            print(f'Output Shape: {data.shape}')
            if data.shape == dst_arr.shape:
                print('Output shape check okay...')

    src, vrt = None, None

    with rasterio.open(out_etopo, 'w', **options) as out:
            out.write(data)   
    
    out = None
    
    return out_etopo

def rmsez(sdb, lidar):
    print('\nComparing SDB with ground truth...')
    with rasterio.open(sdb) as sdb:
        sdb_img = sdb.read(1)
        out_meta = sdb.meta
        sdb_bounds = sdb.bounds
        
    with rasterio.open(lidar) as lidar:
        lidar_img = lidar.read(1)
        out_meta = lidar.meta
        lidar_bounds = lidar.bounds
        
    sdb, lidar = None, None
        
    sdb_img = np.where(np.isnan(sdb_img), 0, sdb_img)
    lidar_img = np.where(lidar_img == -999999., 0, lidar_img)
    
    print(f'Shapes: {sdb_img.shape} {lidar_img.shape}')
    print(f'Bounds: {lidar_bounds} {sdb_bounds}')
    
    if sdb_img.shape == lidar_img.shape and lidar_bounds == sdb_bounds:
        mask1 = (sdb_img != 0)
        mask2 = (lidar_img != 0)
        
        overlap_mask = np.logical_and(mask1, mask2)
        data1 = np.ma.masked_array(sdb_img, mask=~overlap_mask)
        data2 = np.ma.masked_array(lidar_img, mask=~overlap_mask)

        differences = data1 - data2
        differences_flat = differences.flatten()
        mean_diff = np.mean(differences_flat)
        std_dev_diff = np.std(differences_flat)
        rmse = np.sqrt(np.mean(np.square(differences_flat)))
        rmse95 = rmse * 1.96
              
        # plot histogram
        plt.hist(differences_flat, bins=50, color='royalblue', alpha=0.7, label='Differences')
        plt.xlim(mean_diff - 5 * std_dev_diff, mean_diff + 5 * std_dev_diff)
        plt.axvline(mean_diff, color='navy', linestyle='dashed', linewidth=1, label=f'Mean: {mean_diff:.2f}')
        plt.axvline(mean_diff + std_dev_diff, color='slategrey', linestyle='dashed', linewidth=1, label=f'Std. Dev.: {std_dev_diff:.2f}')
        plt.axvline(mean_diff - std_dev_diff, color='slategrey', linestyle='dashed', linewidth=1)
        plt.xlabel('Difference (meters)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        
        # plot difference map
        plt.imshow(differences, vmax=10)
        plt.title('Difference (meters)')
        plt.colorbar()
        plt.show()
        
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlGn
        vmin = -2
        vmax = 2
        boundaries = [vmin, -1, 1, vmax]
        colors = ["yellow", "green", "violet"]
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=False)
        cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=cmap.N)
        plt.imshow(differences, cmap=cmap_custom, vmin=vmin, vmax=vmax)
        plt.title('Difference (meters)')
        plt.colorbar()
        plt.show()
    
        return mean_diff, std_dev_diff, rmse, rmse95
    else: 
        raise Exception('Input ground truth and SDB extents do not match')

if __name__ == '__main__':
    # inputs
    # dest_fp = r"P:\_RSD\Data\ETOPO\SDB\KeyLargo_SDB_Output\KeyLargo_LinearRegression_SDB.tif"
    # in_fp = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\Large\Job780732_2019_NGS_FL_topobathy_DEM_Irma.tif"
    # out_resample_fp = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\KeyLargo_lidar_resample_warp.tif"
    
    dest_fp = r'P:\\_RSD\\Data\\ETOPO\\SDB\\Saipan_SDB_Output\\Saipan_LinearRegression_SDB.tif'
    in_fp = r"P:\_RSD\Data\ETOPO\Ground Truth\Saipan\cnmi2019_islands_dem_J922112.tif"
    out_resample_fp = r"P:\_RSD\Data\ETOPO\Ground Truth\Saipan\Saipan_lidar_resample_warp.tif"
    
    # resample, reproject, mask
    outfile = resample(in_fp, dest_fp, out_resample_fp)
    print(f'\nResampled output: {outfile}')
    
    # grid comparison
    mean_diff, std_dev_diff, rmse, rmse95 = rmsez(dest_fp, outfile)
    print(f'SDB vs Ground Truth (Mean ± Std):\n{mean_diff:.3f} ± {std_dev_diff:.3f}')
    print(f'\nSDB vs Ground Truth (RMSEz, 95% Confidence):\n{rmse:.3f}m ({rmse95:.3f}m @95%)')
    
    