# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:35:10 2023

@author: matthew.sharr
"""

import fiona
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import rasterio
import rasterio.mask
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression


# %% - 

def extract_zs_lidar(lidar_img, zs):
    with rasterio.open(zs) as zs:
        zs_img = zs.read(1)
        
    overlap = np.where(zs_img == 1, lidar_img[0,:,:], np.nan)
    
    print(f'Lidar (med): {np.nanmedian(overlap):.3f}')
    print(f'Lidar (mean): {np.nanmedian(overlap):.3f}')

    overlap = np.reshape(overlap, (-1,1))    

    # plot histogram
    plt.hist(overlap, bins=50, color='royalblue', alpha=0.7, label='pSDBr (less 4m)')
    plt.xlabel('Lidar')
    plt.ylabel('Frequency')
    plt.title('Lidar ZS Values (<4m)')
    plt.legend()
    plt.show()
        
    return overlap

def linear_regression(in492, in665, npy, aoi, lidar_tif, zs):
    with fiona.open(aoi, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(in492) as blue:
        blue_img, out_transform = rasterio.mask.mask(blue, shapes, crop=True)
        blue_meta = blue.meta

    with rasterio.open(in665) as red:
        red_img, out_transform = rasterio.mask.mask(red, shapes, crop=True)
        red_meta = red.meta

    pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)
    lidar = np.load(npy)
    pSDBr_out = pSDBr
    
    extent_meta = red.meta
    
    with rasterio.open(in665) as red:
        red_shape = red.read(1)
        
    with rasterio.open(lidar_tif, 'w', **extent_meta) as tif:
        tif.write(lidar, 1)
        
    with rasterio.open(lidar_tif) as lidar_file:
        lidar_img, out_transform = rasterio.mask.mask(lidar_file, shapes, crop=True)
        out_meta = lidar_file.meta
        
    # extract_zs_lidar(lidar_img, zs)
        
    out_meta.update({"driver": "GTiff",
                      "height": pSDBr_out.shape[1],
                      "width": pSDBr_out.shape[2],
                      "nodata": 0,
                      "transform": out_transform})
    
    pSDBr_fname = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out', 'Nantucket_pSDBr.tif')
        
    with rasterio.open(pSDBr_fname, 'w', **out_meta) as tif:
        tif.write(pSDBr)
    
    with rasterio.open(lidar_tif.replace('.tif','_masked.tif'), 'w', **out_meta) as tif:
        tif.write(lidar_img)
        
    # plt.imshow(lidar_img[0,:,:])
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label('meters', fontsize=8)
    # plt.title('Lidar')
    # plt.show()
    
    # plt.imshow(pSDBr[0,:,:])
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label('pSDBr', fontsize=8)
    # plt.title('pSDBr')
    # plt.show()
    
    pSDBr = np.reshape(pSDBr, (-1,1))
    lidar = np.reshape(lidar_img, (-1,1))
    pSDBr = pSDBr[~np.isnan(lidar)]
    lidar = lidar[~np.isnan(lidar)]
    lidar = lidar[~np.isnan(pSDBr)]    
    pSDBr = pSDBr[~np.isnan(pSDBr)]
    pSDBr = np.reshape(pSDBr, (-1,1))
    lidar = np.reshape(lidar, (-1,1))
    
    limit = 4 # limit to # meters
    pSDBr_4m = np.reshape(np.delete(pSDBr, np.where(lidar > limit)), (-1,1))
    lidar_4m = np.reshape(np.delete(lidar, np.where(lidar > limit)), (-1,1))
    
    # 2d histogram on subset of data
    H, xedges, yedges = np.histogram2d(pSDBr_4m[:,0], lidar_4m[:,0], bins=150)

    # get the bin centers
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    
    # reshape heatmap into two column array
    heatmap_reshaped = np.column_stack((x_grid.ravel(), y_grid.ravel(), H.ravel()))
    
    # extract dense x and y values
    threshold = np.percentile(H, 99)  # percentile to keep
    dense_indices = np.where(H > threshold)
    dense_x_values = x_centers[dense_indices[0]]
    dense_y_values = y_centers[dense_indices[1]]
    
    # linear regression
    model = linregress(dense_x_values, dense_y_values)
    m0 = model.intercept
    m1 = model.slope
    print(f'R2:  {model.rvalue**2:.3f}')
    
    # plot heatmap with fit line for subset
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(H.T, extent=extent, origin='lower', cmap='viridis_r')
    plt.plot(dense_x_values, m1*dense_x_values + m0, color='red', linewidth=1.0)
    plt.xlabel('pSDBr')
    plt.ylabel('Lidar (m)')
    plt.title('Nantucket Subset (20180902)')
    plt.colorbar(label='Density')
    plt.show()
      
    # plot heatmap with fit line for all data
    plt.hist2d(pSDBr[:,0], lidar[:,0], bins=150, cmap='viridis_r')
    plt.plot(pSDBr_4m, m1*pSDBr_4m + m0, color='red', linewidth=0.75)
    plt.xlabel('pSDBr')
    plt.ylabel('Lidar (m)')
    plt.title('Nantucket Subset (20180902)')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Density')
    plt.show()
    
    # plot histogram
    plt.hist(pSDBr_4m, bins=50, color='royalblue', alpha=0.7, label='pSDBr (less 4m)')
    plt.xlabel('pSDBr')
    plt.ylabel('Frequency')
    plt.title('pSDBr Values (<4m)')
    plt.legend()
    plt.show()
       
    return m0, m1, pSDBr_out, out_meta
    
def compare_intercept(pSDBr_zs, pSDBr_in, m0, m1):
    
    m0_zs = -m1 * pSDBr_zs
    intercepts = {'m0':m0, 'm0_zs':m0_zs}
    
    # SDB = m * (psdb - offset)
    
    # sdb = m1 * (pSDBr_in + 5.108)
    # print()
    print(f'\nSlope (LR): {m1:.3f}')
    print(f'm0 (LR):  {m0:.3f}\nm0z (ZS): {m0_zs:.3f}')
    print(f'\nOffset from LR: {m0/m1:.3f}')
    print(f'Offset from ZS: {pSDBr_zs:.3f}\n')

    return m0_zs

def compute_sdb(m0, pSDBr_zs, m1, pSDBr, out_meta, sdb_tif, sdb_zs_tif):
    sdb = m1*pSDBr + m0
    sdb_zs = m1*(pSDBr - pSDBr_zs)

    plt.imshow(sdb[0,:,:])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('meters', fontsize=8)
    plt.title('SDB')
    plt.show()
    
    plt.imshow(sdb_zs[0,:,:])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('meters', fontsize=8)
    plt.title('SDB (from zero shoreline)')
    plt.show()
    
    with rasterio.open(sdb_tif, 'w', **out_meta) as tif:
        tif.write(sdb)
    
    print(f'Wrote: {sdb_tif}')
    tif = None
    
    with rasterio.open(sdb_zs_tif, 'w', **out_meta) as tif:
        tif.write(sdb_zs)
        
    print(f'Wrote: {sdb_zs_tif}')
    
    return None

def rmsez(sdb_tif, lidar_tif):
    with rasterio.open(sdb_tif) as sdb:
        sdb_img = sdb.read(1)
        
    with rasterio.open(lidar_tif.replace('.tif','_masked.tif')) as lidar:
        lidar_img = lidar.read(1)        
    
    differences = sdb_img - lidar_img
    differences_flat = differences.flatten()
    differences_flat = differences_flat[~np.isnan(differences_flat)]
    mean_diff = np.mean(differences_flat)
    median_diff = np.median(differences_flat)
    std_dev_diff = np.std(differences_flat)
    rmse = np.sqrt(np.mean(np.square(differences_flat)))
    rmse95 = rmse * 1.96
    
    print(f'\nRMSE: {rmse:.3f}m; @95%: {rmse95:.3f}m')
    print(f'Mean difference: {np.nanmean(differences):.3f}m')
    print(f'Median difference: {median_diff:.3f}m')
          
    # plot histogram
    plt.hist(differences_flat, bins=100, color='royalblue', alpha=0.7, label='Differences')
    plt.xlim(mean_diff - 5 * std_dev_diff, mean_diff + 5 * std_dev_diff)
    plt.axvline(median_diff, color='blueviolet', linestyle='dotted', linewidth=1, label=f'Median: {median_diff:.2f}m' )
    plt.axvline(mean_diff, color='navy', linestyle='dashed', linewidth=1, label=f'Mean: {mean_diff:.2f}m')
    plt.axvline(mean_diff + std_dev_diff, color='slategrey', linestyle='dashed', linewidth=1, label=f'Std. Dev.: {std_dev_diff:.2f}m')
    plt.axvline(mean_diff - std_dev_diff, color='slategrey', linestyle='dashed', linewidth=1)
    plt.xlabel('Difference (meters)')
    plt.ylabel('Frequency')
    plt.title('Difference Distribution')
    plt.legend()
    plt.show()
    
    # plot difference map
    plt.imshow(differences, vmax=10)
    plt.title('Difference (meters)')
    plt.colorbar()
    plt.show()
    
    return None
    

# %% - main

if __name__ == '__main__':
    # input
    in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\S2A_MSIL1C_20180902T153551_N0206_R111_T19TCF_20180902T204206.SAFE\S2A_MSI_2018_09_02_15_39_09_T19TCF_L2W_Rrs_492.tif"
    in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\S2A_MSIL1C_20180902T153551_N0206_R111_T19TCF_20180902T204206.SAFE\S2A_MSI_2018_09_02_15_39_09_T19TCF_L2W_Rrs_665.tif"
    # aoi = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\Nantucket_Zero2.shp"
    aoi = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\Nantucket_Zero_12Jan2024.shp"
    npy = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\lidar_regrid\usace2018_east_cst_dem_J914372_000_000.npy"
    zs = r'C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out\Nantucket_ZS_RF_int.tif'
    pSDBr_zs = 0.948
    
    # output
    lidar_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\lidar_regrid', 'Nantucket_lidar2.tif')
    sdb_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\lidar_regrid', 'Nantucket_sdb2.tif')
    sdb_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\lidar_regrid', 'Nantucket_sdb_zs2.tif')

    # functions
    m0, m1, pSDBr_in, out_meta = linear_regression(in492, in665, npy, aoi, lidar_tif, zs)
    m0_zs = compare_intercept(pSDBr_zs, pSDBr_in, m0, m1)
    compute_sdb(m0, pSDBr_zs, m1, pSDBr_in, out_meta, sdb_tif, sdb_zs_tif)
    rmsez(sdb_zs_tif, lidar_tif)
    
    
    