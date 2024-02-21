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

def npy_to_tif(npy, lidar_out):
    lidar = np.load(npy)
    
    with fiona.open(aoi, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(in492) as blue:
        blue_img, out_transform = rasterio.mask.mask(blue, shapes, crop=True)
        blue_meta = blue.meta
        
    extent_meta = blue.meta
    
    with rasterio.open(lidar_tif, 'w', **extent_meta) as tif:
        tif.write(lidar, 1)
        
    with rasterio.open(lidar_tif) as lidar_file:
        lidar_img, out_transform = rasterio.mask.mask(lidar_file, shapes, crop=True)
        out_meta = lidar_file.meta
    
    out_meta.update({"driver": "GTiff",
                      "height": blue_img.shape[1],
                      "width": blue_img.shape[2],
                      "nodata": 0,
                      "transform": out_transform})
    
    with rasterio.open(lidar_tif.replace('.tif','_masked.tif'), 'w', **out_meta) as tif:
        tif.write(lidar_img)
    
    return lidar_img

def extract_zs_lidar(lidar_img, zs, pSDBr, pSDBg):
    with rasterio.open(zs) as zs:
        zs_img = zs.read(1)
        
    lidar_zs = np.where(zs_img == 1, lidar_img[0,:,:], np.nan)
    pSDBg_zs = np.where(zs_img == 1, pSDBg[0,:,:], np.nan) # here pSDB is land and cloud masked
    pSDBr_zs = np.where(zs_img == 1, pSDBr[0,:,:], np.nan) # here pSDB is land and cloud masked
    lidar_zs = lidar_zs[~np.isnan(lidar_zs)] 
    
    print(f'Lidar ZS (med): {np.nanmedian(lidar_zs):.3f}')
    print(f'Lidar ZS (mean): {np.nanmean(lidar_zs):.3f}')
    print(f'Count: {lidar_zs.size}')
    # print(f'pSDBr (med): {np.nanmedian(pSDBr_zs):.3f}')
    # print(f'pSDBr (mean): {np.nanmean(pSDBr_zs):.3f}')
    # print(f'pSDBg (med): {np.nanmedian(pSDBg_zs):.3f}')
    # print(f'pSDBg (mean): {np.nanmean(pSDBg_zs):.3f}')

    # plot histogram
    plt.hist(lidar_zs.flatten(), bins=100, color='royalblue', alpha=0.7, label='lidar (zs)')
    plt.xlabel('Lidar')
    plt.ylabel('Frequency')
    plt.title('Lidar ZS Values')
    plt.legend()
    plt.show()
        
    pSDBr_lidar = np.where((lidar_img >= 0) & (lidar_img <=0.25), pSDBr, np.nan)
    mean_diff = np.nanmean(pSDBr_lidar)
    median_diff = np.nanmedian(pSDBr_lidar)
    std_dev_diff = np.nanstd(pSDBr_lidar)
    print(f'pSDBr (0-0.25m) (med): {np.nanmedian(pSDBr_lidar):.3f}')
    print(f'pSDBr (0-0.25m) (mean): {np.nanmean(pSDBr_lidar):.3f}')
    print(f'Count: {pSDBr_lidar.size}')
    
    plt.hist(pSDBr_lidar.flatten(), bins=100, color='royalblue', alpha=0.7, label='lidar (zs)')
    plt.xlim(mean_diff - 5 * std_dev_diff, mean_diff + 5 * std_dev_diff)
    plt.axvline(median_diff, color='blueviolet', linestyle='dotted', linewidth=1.2, label=f'Median: {median_diff:.2f}m' )
    plt.axvline(mean_diff, color='navy', linestyle='dashed', linewidth=1, label=f'Mean: {mean_diff:.2f}m')
    plt.axvline(mean_diff + std_dev_diff, color='slategrey', linestyle='dashed', linewidth=1, label=f'Std. Dev.: {std_dev_diff:.2f}m')
    plt.axvline(mean_diff - std_dev_diff, color='slategrey', linestyle='dashed', linewidth=1)
    plt.xlabel('pSDBr')
    plt.ylabel('Frequency')
    plt.title('pSDBr (0-0.25m lidar)')
    plt.legend()
    plt.show()
    
    # # plot histogram
    # plt.hist(pSDBr_zs, bins=150, color='royalblue', alpha=0.7, label='pSDBr (zs)')
    # plt.xlabel('pSDBr')
    # plt.ylabel('Frequency')
    # plt.title('pSDBg ZS Values')
    # plt.legend()
    # plt.show()
    
    # # plot histogram
    # plt.hist(pSDBg_zs, bins=150, color='royalblue', alpha=0.7, label='pSDBg (zs)')
    # plt.xlabel('pSDBg')
    # plt.ylabel('Frequency')
    # plt.title('pSDBg ZS Values')
    # plt.legend()
    # plt.show()
        
    return lidar_zs

def compare_intercept(pSDBr_zs, pSDBg_zs, pSDBr_in, pSDBg_in, m0red, m1red, m0green, m1green):
    
    m0red_zs = -m1red * pSDBr_zs
    m0green_zs = -m1green * pSDBg_zs

    print(f'\nm1red (LR): {m1red:.3f}')
    print(f'm1green (LR): {m1green:.3f}')
    print(f'\nm0red (LR):  {m0red:.3f}')
    print(f'm0red (ZS): {m0red_zs:.3f}')
    print(f'▲red: {m0red - m0red_zs:.3f}')
    print(f'm0green (LR):  {m0green:.3f}')
    print(f'm0green (ZS): {m0green_zs:.3f}')
    print(f'▲green: {m0green - m0green_zs:.3f}')
    print(f'\nOffset from LRred: {m0red/m1red:.3f}')
    print(f'Offset from ZSred: {pSDBr_zs:.3f}')
    print(f'Offset from LRgreen: {m0green/m1green:.3f}')
    print(f'Offset from ZSgreen: {pSDBg_zs:.3f}\n')

    return m0red_zs, m0green_zs

# def reshape_filter(arr1, arr2, limit, greater):
#     # arr1_flat = arr1.flatten()
#     # arr2_flat = arr2.flatten()
#     # arr1_filtered = arr1_flat[~np.isnan(arr1_flat)]
#     # arr2_filtered = arr2_flat[~np.isnan(arr2_flat)]
    
#     combined_array = np.vstack((arr1.flatten(), arr2.flatten()))
#     nan_rows = np.any(np.isnan(combined_array), axis=1)
#     arr1_filtered = arr1.flatten()[~nan_rows]
#     arr2_filtered = arr2.flatten()[~nan_rows]
    
#     if greater:
#         arr1_filtered = arr1_filtered[arr2_filtered >= limit]
#         arr2_filtered = arr2_filtered[arr2_filtered >= limit]
#     else: 
#         arr1_filtered = arr1_filtered[arr2_filtered <= limit]
#         arr2_filtered = arr2_filtered[arr2_filtered <= limit]

#     return arr1_filtered, arr2_filtered

def linear_regression(in492, in560, in665, npy, aoi, lidar_tif, zs, pSDBr_zs, pSDBg_zs, limit, perc):
    with fiona.open(aoi, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(in492) as blue:
        blue_img, out_transform = rasterio.mask.mask(blue, shapes, crop=True)
        blue_meta = blue.meta
        
    with rasterio.open(in560) as green:
        green_img, out_transform = rasterio.mask.mask(green, shapes, crop=True)
        green_meta = green.meta        

    with rasterio.open(in665) as red:
        red_img, out_transform = rasterio.mask.mask(red, shapes, crop=True)
        red_meta = red.meta

    pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)
    pSDBg = np.log(blue_img * 1000) / np.log(green_img * 1000)
    lidar = np.load(npy)
    pSDBr_out = pSDBr
    pSDBg_out = pSDBg
    
    extent_meta = red.meta
    
    with rasterio.open(in665) as red:
        red_shape = red.read(1)
        
    with rasterio.open(lidar_tif, 'w', **extent_meta) as tif:
        tif.write(lidar, 1)
        
    with rasterio.open(lidar_tif) as lidar_file:
        lidar_img, out_transform = rasterio.mask.mask(lidar_file, shapes, crop=True)
        out_meta = lidar_file.meta
        
    extract_zs_lidar(lidar_img, zs, pSDBr, pSDBg)
        
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
    
    # plt.imshow(pSDBg[0,:,:])
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label('pSDBr', fontsize=8)
    # plt.title('pSDBr')
    # plt.show()
    
    # reshape in a function the will work for both red and green
    # pSDBr_4m, lidarR_4m = reshape_filter(pSDBr, lidar_img, limit, greater=False)
    # pSDBg_4m, lidarG_4m = reshape_filter(pSDBg, lidar_img, limit, greater=True)
    
    pSDBr = pSDBr.flatten()
    pSDBg = pSDBg.flatten()
    lidar = lidar_img.flatten()
    # print('1', pSDBr.shape, pSDBg.shape, lidar.shape)
    
    pSDBr = pSDBr[~np.isnan(lidar)]
    pSDBg = pSDBg[~np.isnan(lidar)]
    lidar = lidar[~np.isnan(lidar)]
    
    # print('2', pSDBr.shape, pSDBg.shape, lidar.shape)
    # print(np.count_nonzero(np.isnan(pSDBr)))
    # print(np.count_nonzero(np.isnan(pSDBg)))
    # print(np.count_nonzero(np.isnan(lidar)))

    lidarR = lidar[~np.isnan(pSDBr)]    
    pSDBr = pSDBr[~np.isnan(pSDBr)]
    lidarG = lidar[~np.isnan(pSDBg)] 
    pSDBg = pSDBg[~np.isnan(pSDBg)]
    
    # print('3', pSDBr.shape, pSDBg.shape, lidarR.shape, lidarG.shape)

    pSDBr = np.reshape(pSDBr, (-1,1))
    pSDBg = np.reshape(pSDBg, (-1,1))
    lidarR = np.reshape(lidarR, (-1,1))
    lidarG = np.reshape(lidarG, (-1,1))
    
    # print(np.count_nonzero(np.isnan(pSDBr)))
    # print(np.count_nonzero(np.isnan(pSDBg)))
    # print(np.count_nonzero(np.isnan(lidarR)))
    # print(np.count_nonzero(np.isnan(lidarG)))
    # print(pSDBr.shape, pSDBg.shape, lidarR.shape, lidarG.shape)
    
    # limit = 4 # limit to # meters
    pSDBr_4m = np.reshape(np.delete(pSDBr, np.where(lidarR > limit)), (-1,1))
    lidarR_4m = np.reshape(np.delete(lidarR, np.where(lidarR > limit)), (-1,1))
    pSDBg_4m = np.reshape(np.delete(pSDBg, np.where(lidarG < limit)), (-1,1))
    lidarG_4m = np.reshape(np.delete(lidarG, np.where(lidarG < limit)), (-1,1))
    
    # print(f'shape check:\n{lidarR_4m.shape} (lidarR_4m)\n{pSDBr_4m.shape} (pSDBr_4m)\n{lidarG_4m.shape} (lidarG_4m)\n{pSDBg_4m.shape} (pSDBg_4m)')

    
    ## pSDBr
    # 2d histogram on subset of data
    H, xedges, yedges = np.histogram2d(pSDBr_4m[:,0], lidarR_4m[:,0], bins=150)
    # get the bin centers
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    threshold = np.percentile(H, perc)  # percentile to keep
    dense_indices = np.where(H > threshold)
    dense_x_values = x_centers[dense_indices[0]]
    dense_y_values = y_centers[dense_indices[1]]
    
    # pSDBr linear regression
    model = linregress(dense_x_values, dense_y_values)
    m0red = model.intercept
    m1red = model.slope
    print(f'R2red:  {model.rvalue**2:.3f}')
    
    ## pSDBg
    # 2d histogram on subset of data
    Hgreen, Gxedges, Gyedges = np.histogram2d(pSDBg_4m[:,0], lidarG_4m[:,0], bins=150)
    # get the bin centers
    xgreen_centers = (Gxedges[:-1] + Gxedges[1:]) / 2
    ygreen_centers = (Gyedges[:-1] + Gyedges[1:]) / 2
    xgreen_grid, ygreen_grid = np.meshgrid(xgreen_centers, ygreen_centers)
    thresholdgreen = np.percentile(Hgreen, perc)  # percentile to keep
    densegreen_indices = np.where(Hgreen > thresholdgreen)
    dense_xgreen_values = xgreen_centers[densegreen_indices[0]]
    dense_ygreen_values = ygreen_centers[densegreen_indices[1]]
    
    # pSDBg linear regression
    modelG = linregress(dense_xgreen_values, dense_ygreen_values)
    m0green = modelG.intercept
    m1green = modelG.slope
    print(f'R2green:  {modelG.rvalue**2:.3f}')
    
    m0red_zs, m0green_zs = compare_intercept(pSDBr_zs, pSDBg_zs, pSDBr_out, pSDBg_out, m0red, m1red, m0green, m1green)
    
    # plots that will work for both red and green
    
    # plot heatmap with fit line for subset
    # extentr = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.hist2d(pSDBr_4m[:,0], lidarR_4m[:,0], bins=150, cmap='viridis_r')
    # plt.imshow(H.T, extent=extentr, origin='lower', cmap='viridis_r')
    plt.plot(dense_x_values, m1red*dense_x_values + m0red, color='red', linewidth=1.0, label='Regression')
    plt.plot(dense_x_values, m1red*dense_x_values - 3.829, color='white', linewidth=1.0, label='Zero Shoreline')
    plt.xlabel('pSDBr')
    plt.ylabel('Lidar (m)')
    plt.title('pSDBr Subset')
    plt.colorbar(label='Density')
    plt.legend(facecolor='grey')
    plt.show()
      
    # plot heatmap with fit line for all data
    plt.hist2d(pSDBr[:,0], lidarR[:,0], bins=150, cmap='viridis_r')
    plt.plot(pSDBr_4m, m1red*pSDBr_4m + m0red, color='red', linewidth=0.75, label='Regression')
    plt.plot(pSDBr_4m, m1red*pSDBr_4m - 3.829, color='white', linewidth=0.75, label='Zero Shoreline')
    plt.xlabel('pSDBr')
    plt.ylabel('Lidar (m)')
    plt.title('pSDBr v. lidar')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Density')
    plt.legend(facecolor='grey')
    plt.show()
    
    # plot histogram
    # plt.hist(pSDBr_4m, bins=50, color='royalblue', alpha=0.7, label='pSDBr (less 4m)')
    # plt.xlabel('pSDBr')
    # plt.ylabel('Frequency')
    # plt.title('pSDBr Values (<4m)')
    # plt.legend()
    # plt.show()
    
    # plot heatmap with fit line for all data
    # extentg = [Gxedges[0], Gxedges[-1], Gyedges[0], Gyedges[-1]]
    plt.hist2d(pSDBg_4m[:,0], lidarG_4m[:,0], bins=150, cmap='viridis_r')
    plt.plot(dense_xgreen_values, m1green*dense_xgreen_values + m0green, color='red', linewidth=1.0, label='Regression')    
    plt.plot(dense_xgreen_values, m1green*dense_xgreen_values - 57.486, color='white', linewidth=1.0, label='Zero Shoreline')    
    plt.xlabel('pSDBg')
    plt.ylabel('Lidar (m)')
    plt.title('pSDBg Subset')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Density')
    plt.legend(facecolor='grey')
    plt.show()
    
    # plot heatmap with fit line for all data
    plt.hist2d(pSDBg[:,0], lidarG[:,0], bins=150, cmap='viridis_r')
    plt.plot(pSDBg_4m, m1green*pSDBg_4m + m0green, color='red', linewidth=0.75, label='Regression')
    plt.plot(pSDBg_4m, m1green*pSDBg_4m - 57.486, color='white', linewidth=0.75, label='Zero Shoreline')
    plt.xlabel('pSDBg')
    plt.ylabel('Lidar (m)')
    plt.title('pSDBg v. lidar')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Density')
    plt.legend(facecolor='grey')
    plt.show()
       
    return m0red, m1red, m0green, m1green, pSDBr_out, pSDBg_out, out_meta

def compute_sdb(m0, pSDB_zs, m1, pSDB, out_meta, sdb_tif, sdb_zs_tif):
    # sdb = m1*pSDB + m0
    sdb_zs = m1*(pSDB - pSDB_zs)

    # plt.imshow(sdb[0,:,:])
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label('meters', fontsize=8)
    # plt.title('SDB')
    # plt.show()
    
    # plt.imshow(sdb_zs[0,:,:])
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label('meters', fontsize=8)
    # plt.title('SDB (from zero shoreline)')
    # plt.show()
    
    # with rasterio.open(sdb_tif, 'w', **out_meta) as tif:
    #     tif.write(sdb)
    
    # print(f'Wrote: {sdb_tif}')
    # tif = None
    
    # with rasterio.open(sdb_zs_tif, 'w', **out_meta) as tif:
    #     tif.write(sdb_zs)
        
    # print(f'Wrote: {sdb_zs_tif}')
    # tif = None
    
    return sdb_zs # we can return here and not actually write?

def rmsez(sdb_tif, lidar_tif):
    # with rasterio.open(sdb_tif) as sdb:
    #     sdb_img = sdb.read(1)
    
    sdb_img = sdb_tif[0,:,:]
        
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
    
    print(f'RMSE (v. lidar): {rmse:.3f}m @95%: {rmse95:.3f}m')
    print(f'Mean difference (v. lidar): {np.nanmean(differences):.3f}m')
    print(f'Median difference (v. lidar): {median_diff:.3f}m\n')
          
    # plot histogram
    plt.hist(differences_flat, bins=100, color='royalblue', alpha=0.7, label='Differences')
    plt.xlim(mean_diff - 5 * std_dev_diff, mean_diff + 5 * std_dev_diff)
    plt.axvline(median_diff, color='blueviolet', linestyle='dotted', linewidth=1.2, label=f'Median: {median_diff:.2f}m' )
    plt.axvline(mean_diff, color='navy', linestyle='dashed', linewidth=1, label=f'Mean: {mean_diff:.2f}m')
    plt.axvline(mean_diff + std_dev_diff, color='slategrey', linestyle='dashed', linewidth=1, label=f'Std. Dev.: {std_dev_diff:.2f}m')
    plt.axvline(mean_diff - std_dev_diff, color='slategrey', linestyle='dashed', linewidth=1)
    plt.xlabel('Difference (meters)')
    plt.ylabel('Frequency')
    plt.title('Difference Distribution')
    plt.legend()
    plt.show()
    
    # plot difference map
    # plt.imshow(differences, vmax=10)
    # plt.title('Difference (meters)')
    # plt.colorbar()
    # plt.show()
    
    return None
    

# %% - main

if __name__ == '__main__':
    ## nantucket input
    in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\S2A_MSIL1C_20180902T153551_N0206_R111_T19TCF_20180902T204206.SAFE\S2A_MSI_2018_09_02_15_39_09_T19TCF_L2W_Rrs_492.tif"
    in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\S2A_MSIL1C_20180902T153551_N0206_R111_T19TCF_20180902T204206.SAFE\S2A_MSI_2018_09_02_15_39_09_T19TCF_L2W_Rrs_560.tif"
    in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\S2A_MSIL1C_20180902T153551_N0206_R111_T19TCF_20180902T204206.SAFE\S2A_MSI_2018_09_02_15_39_09_T19TCF_L2W_Rrs_665.tif"
    aoi = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\Nantucket_Zero_12Jan2024.shp"
    npy = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\lidar_regrid\usace2018_east_cst_dem_J914372_000_000.npy"
    zs = r'C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out\Nantucket_ZS_RF_int_20240206.tif'
    pSDBr_zs = 0.948
    pSDBg_zs = 0.932
    
    # nantucket output
    lidar_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\lidar_regrid', 'Nantucket_lidar.tif')
    sdbr_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\_comparisons', 'Nantucket_sdbRed.tif')
    sdbg_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\_comparisons', 'Nantucket_sdbGreen.tif')
    sdbr_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\_comparisons', 'Nantucket_sdbRed_zs.tif')
    sdbg_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Nantucket\Nantucket\_comparisons', 'Nantucket_sdbGreen_zs.tif')
    
    ## hatteras input
    # in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\S2B_MSIL1C_20191015T154219_N0208_R011_T18SVD_20191015T191348.SAFE\S2B_MSI_2019_10_15_15_53_37_T18SVD_L2W_Rrs_492.tif"
    # in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\S2B_MSIL1C_20191015T154219_N0208_R011_T18SVD_20191015T191348.SAFE\S2B_MSI_2019_10_15_15_53_37_T18SVD_L2W_Rrs_559.tif"
    # in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\S2B_MSIL1C_20191015T154219_N0208_R011_T18SVD_20191015T191348.SAFE\S2B_MSI_2019_10_15_15_53_37_T18SVD_L2W_Rrs_665.tif"
    # aoi = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\_Bryan_SHP\aoi_hatteras20191015_UTM18N.shp"
    # npy = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_2019\lidar_regrid\nc2019_dunex_J849189_hatt2019.npy"
    # zs = r'C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out\Hatteras_ZS_RF_int.tif'
    # pSDBr_zs = 0.956
    # pSDBg_zs = 0.946
    # # pSDBr_zs = 0.986
    # # pSDBg_zs = 0.940
    
    # # # hatteras output
    # lidar_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_2019\lidar_regrid', 'Hatteras_lidar.tif')
    # sdbr_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_2019\_comparisons', 'Hatteras_sdbRed.tif')
    # sdbg_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_2019\_comparisons', 'Hatteras_sdbGreen.tif')
    # sdbr_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_2019\_comparisons', 'Hatteras_sdbRed_zs.tif')
    # sdbg_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_2019\_comparisons', 'Hatteras_sdbGreen_zs.tif')
    
    ## puerto rico input
    # in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\Processed_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_492.tif"
    # in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\Processed_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_559.tif"
    # in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\Processed_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_665.tif"
    # aoi = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\_Bryan_SHP\aoi_puertoRico_UTM19N.shp"
    # npy = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\G_PR_lidar__BLK-e.npy"
    # zs = r'C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out\PuertoRico_ZS_RF_int_20240206.tif'
    # pSDBr_zs = 0.977
    # pSDBg_zs = 0.969

    # ## puerto rico output
    # lidar_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\lidar_regrid', 'PuertoRico_lidar.tif')
    # sdbr_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\_comparisons', 'PuertoRico_sdbRed.tif')
    # sdbg_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\_comparisons', 'PuertoRico_sdbGreen.tif')
    # sdbr_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\_comparisons', 'PuertoRico_sdbRed_zs.tif')
    # sdbg_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\PuertoRico\_comparisons', 'PuertoRico_sdbGreen_zs.tif')
    
    ## hatteras point composite input
    # in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\hatt_point\hatt_point_18\prod\pSDB_composites\blue\bluemaximumpSDBred.tif"
    # in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\hatt_point\hatt_point_18\prod\pSDB_composites\green\greenmaximumpSDBred.tif"
    # in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\hatt_point\hatt_point_18\prod\pSDB_composites\red\redmaximumpSDBred.tif"
    # aoi = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\_Bryan_SHP\aoi_hatterasPt_WGS84_UTM18N.shp"
    # npy = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\hatt_point\hatt_point_18\lidar_regrid\usace2018_east_cst_dem_J925332.npy"
    # zs = r'C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out\HatterasPt_ZS_RF_int_composite.tif'
    # pSDBr_zs = 1.079
    # pSDBg_zs = 0.979

    # # hatteras point composite output
    # lidar_tif = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\hatt_point\hatt_point_18\lidar_regrid", 'HatterasPt_lidar.tif')
    # sdbr_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\_comparisons', 'HatterasPt_sdbRed.tif')
    # sdbg_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\_comparisons', 'HatterasPt_sdbGreen.tif')
    # sdbr_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\_comparisons', 'HatterasPt_sdbRed_zs.tif')
    # sdbg_zs_tif = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Bryan\Hatteras_Point\_comparisons', 'HatterasPt_sdbGreen_zs.tif')

    # functions
    m0red, m1red, m0green, m1green, pSDBr_in, pSDBg_in, out_meta = linear_regression(in492, in560, in665, npy, aoi, 
                                                                                     lidar_tif, zs, pSDBr_zs, pSDBg_zs, limit=4, perc=99)
    # sdbRed_zs = compute_sdb(m0red, pSDBr_zs, m1red, pSDBr_in, out_meta, sdbr_tif, sdbr_zs_tif)
    # sdbGreen_zs = compute_sdb(m0green, pSDBg_zs, m1green, pSDBg_in, out_meta, sdbg_tif, sdbg_zs_tif)
    # rmsez(sdbRed_zs, lidar_tif)
    # rmsez(sdbGreen_zs, lidar_tif)
    
    
    