# -*- coding: utf-8 -*-
"""
October 2023
@author: matthew.sharr


"""

import compare_raster as resample
import datetime
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import pandas as pd
import pickle
import pyproj
import rasterio
import rasterio.mask
# import regrid_to_sdb as regrid
from scipy import spatial
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import lsq_linear
from scipy.stats import linregress
from skimage import feature, filters
# from sklearn import svm
from sklearn.linear_model import LinearRegression, RANSACRegressor, Lasso, Ridge, HuberRegressor, SGDRegressor, TheilSenRegressor, BayesianRidge
from sklearn.metrics import classification_report, jaccard_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import plot_tree
import warnings


# %% - helper functions

current_time = datetime.datetime.now() 
warnings.filterwarnings('ignore')

def list_inputs(rgb_dir, shp_dir, truthiness_dir):
    # img_dirs = []
    # for loc in rgb_dir:
    #     [img_dirs.append(os.path.join(loc, folder)) for folder in os.listdir(loc)]
        
    maskSHP_dir = []
    for loc in shp_dir:
        [maskSHP_dir.append(os.path.join(loc, shp)) for shp in os.listdir(loc) if shp.endswith('.shp')]
        
    # true_bathy = [os.path.join(truthiness_dir, tf) for tf in os.listdir(truthiness_dir) if tf.endswith('.tif')]
    
    return maskSHP_dir

def check_bounds(rgb_dir, shapefile):
    raster = [os.path.join(rgb_dir, r) for r in os.listdir(rgb_dir) if r.endswith('.tif')]
    raster_bounds = rasterio.open(raster[0]).bounds
        
    # check if shapefiles point locations are inside the bounds of the raster
    shp_bounds = fiona.open(shapefile, 'r').bounds
    
    # check bounds
    eastings_within = np.logical_and(shp_bounds[0] > raster_bounds[0], # left
                                     shp_bounds[2] < raster_bounds[2]) # right
    northings_within = np.logical_and(shp_bounds[1] > raster_bounds[1], # bottom
                                      shp_bounds[3] < raster_bounds[3]) # top
    
    if np.all([eastings_within, northings_within]):
        print(f'{os.path.basename(shapefile)} within bounds of {os.path.basename(rgb_dir)} imagery\n')
        return True
    else:
        return False
    
# pair composite image with labels
def pair_with(out_transform, truthiness_dir):
    truthiness_labels = os.listdir(truthiness_dir)
    
    truthiness_mask = None
    
    for label in truthiness_labels:
        label = os.path.join(truthiness_dir, label)
        truthiness_transform = rasterio.open(label).transform
        if out_transform == truthiness_transform:
            truthiness_mask = label
        else:
            continue

    if truthiness_mask == None:
        print('No matching truthiness masks found...')
        return None
    else:    
        return truthiness_mask

# return subsampled training data and corresponding labels
def subsample(array1, array2, adjustment):   
    # get indices of rows containing 0 and 1
    indices_with_zeros = np.where(array1 == 0)[0]
    indices_with_ones = np.where(array1 == 1)[0]
    
    # randomly select a subset of rows containing 0
    num_rows_to_select = np.count_nonzero(array1 == 1) * adjustment # Adjust as needed
    rng = np.random.default_rng(0)
    selected_indices_zeros = rng.choice(indices_with_zeros, size=num_rows_to_select, replace=False)
    
    # include the randomly selected rows
    selected_labels = np.concatenate((array1[indices_with_ones], array1[selected_indices_zeros]))
    selected_training = np.vstack((array2[indices_with_ones], array2[selected_indices_zeros]))
    
    return selected_labels, selected_training

# normalize rgb imagery for plotting
def normalize(band):
    band_min, band_max = (np.nanmin(band), np.nanmax(band))
    return ((band-band_min)/((band_max - band_min)))

# plot the rgb image and nearshore pixels overlaid on ndwi
def plot_rgb_poi(cropped_red, cropped_green, cropped_blue, prediction, pSDBg, out_transform, w, h, brightness, aspect):
    # normalize rgb image
    red_n = normalize(cropped_red[0,:,:])
    green_n = normalize(cropped_green[0,:,:])
    blue_n = normalize(cropped_blue[0,:,:])
    
    # brighten rgb image
    rgb_composite_n = np.dstack((red_n, green_n, blue_n))
    # brightened_image = np.clip(rgb_composite_n * brightness, 0, 255)
    brightened_image = rgb_composite_n * 255.0 * brightness
    brightened_image = np.where(np.isnan(brightened_image), 255, brightened_image.astype(np.uint8))
    
    # set extent for plotting
    x_min, y_max = out_transform * (0, 0)  
    x_max, y_min = out_transform * (w, h)  
    extent = [x_min, x_max, y_min, y_max]
        
    f, ax = plt.subplots(1,2, figsize=(12, 6), dpi=200)
    
    ax[0].imshow(brightened_image, extent=extent)
    ax[0].set_title('RGB Composite', fontsize=10)
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)
    ax[0].set_xlabel('UTM Easting (m)')
    ax[0].set_ylabel('UTM Northing (m)')
    ax[0].set_aspect(aspect)  # adjust the value as needed
    # ax[0].grid(True)
    
    ax[1].imshow(prediction, cmap='viridis', extent=extent)
    ax[1].set_title('Predicted Nearshore Pixels Overlaid on NDWI', fontsize=10)
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)
    ax[1].set_xlabel('UTM Easting (m)')
    ax[1].set_ylabel('UTM Northing (m)')
    ax[1].set_aspect(aspect)  # adjust the value as needed
    
    # ax[1].colorbar()
    plt.tight_layout()
    plt.show()
    
    return None


# %% - data prep

def wavelengths(rgb_dir):
    rgb_list = os.listdir(rgb_dir)
    
    for band in rgb_list:
        if '492' in band and '.aux.xml' not in band:
            in492 = os.path.join(rgb_dir, band)
        elif '560' in band and '.aux.xml' not in band or '559' in band and '.aux.xml' not in band:
            in560 = os.path.join(rgb_dir, band)
        elif '665' in band and '.aux.xml' not in band:
            in665 = os.path.join(rgb_dir, band)
        
    return in492, in560, in665
    
def apply_truthiness(pSDBg, mask):
       
    with rasterio.open(mask) as tf:
        truthiness_tif = tf.read(1)
        out_meta = tf.meta
            
    nans = np.logical_or(np.isnan(pSDBg), np.logical_or(np.logical_or(truthiness_tif == 0, truthiness_tif == 1), pSDBg == 0))
    
    pSDBg[nans] = np.nan
    dims = pSDBg[0,:,:].shape
    
    # num_zeros = np.count_nonzero(pSDBg == 0)
    
    return pSDBg, nans[0,:,:], dims

# linear equation y = mx + b
def linear_eq(params, x):
    m, b = params
    return m * x + b

def constrained_lstsq(x_train, y_train):
    x = x_train.reshape(-1).astype(np.float64)
    y = y_train.reshape(-1).astype(np.float64)
    
    # Initial guess for parameters (slope and intercept)
    initial_guess = [1, 0]  # <-- Initial guess of [1, 0]
    
    # Perform least squares linear regression with constraints
    result = lsq_linear(A=np.vstack([x, np.ones_like(x)]).T, 
                        b=y, bounds=([45, -np.inf], [120, np.inf]), 
                        method='trf', #'bvls', 
                        verbose=1) #'trf'
    m1, m0 = result.x
    print(f"Slope (m1): {m1:.3f}")
    print(f"Intercept (m0): {m0:.3f}")
    
    y_pred = m1 * x + m0
    residuals = y - y_pred
    tss = np.sum((y - np.mean(y))**2)
    ssr = np.sum(residuals**2)
    r2 = 1 - (ssr / tss)
    print(f"R2: {r2:.3f}")
    
    # Plot the data and least squares fit
    plt.scatter(x, y, s=0.8, label='Data')
    plt.plot(x, linear_eq(result.x, x), color='red', label='Least Squares Fit')
    plt.xlabel('pSDBg', fontsize=8)
    plt.ylabel('ETOPO (EGM08 [m])', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('ETOPO vs. pSDBg - Constrained Lsq', fontsize=10)
    plt.legend()
    plt.show()
    
    return m1, m0

# %% - pSDB prep
def compute_pSDB(in492, in560, in665, land, truthiness_dir, truthiness):
    with fiona.open(land, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    
    with rasterio.open(in492) as blue:
        blue_img, out_transform = rasterio.mask.mask(blue, shapes, crop=True)
        out_meta = blue.meta

    with rasterio.open(in560) as green:
        green_img, out_transform = rasterio.mask.mask(green, shapes, crop=True)
        out_meta = green.meta
        w = green.width
        h = green.height
        
    with rasterio.open(in665) as red:
        red_img, out_transform = rasterio.mask.mask(red, shapes, crop=True)
        out_meta = red.meta
    
    print(f'Using the following in pSDBgreen calculation:\n{in492}\n{in560}\n')
    pSDBg = np.log(blue_img * 1000) / np.log(green_img * 1000)
    
    print(f'Using the following in pSDBred calculation:\n{in492}\n{in560}\n')
    pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)
    
    if truthiness:
        truthiness_mask = pair_with(out_transform, truthiness_dir)   
        pSDBg, nans, dims = apply_truthiness(pSDBg, truthiness_mask)
    else:
        nans = np.where(np.isnan(pSDBg[0,:,:]))
        dims = pSDBg.shape               
    
    # pSDBg = gaussian_filter(pSDBg, sigma=1)
    # pSDBg = median_filter(pSDBg, size=3)
    
    pSDBg_output = os.path.join(os.path.dirname(in492), 'pSDBg.tif')
    pSDBr_output = os.path.join(os.path.dirname(in665), 'pSDBr.tif')
    
    out_meta.update({"driver": "GTiff",
                      "height": pSDBg.shape[1],
                      "width": pSDBg.shape[2],
                      "nodata": 0,
                      "transform": out_transform})
    
    with rasterio.open(pSDBg_output, 'w', **out_meta) as dstg:
        dstg.write(pSDBg)   
        
    print(f'Saved temp pSDBg file: {pSDBg_output}')
    
    with rasterio.open(pSDBr_output, 'w', **out_meta) as dstr:
        dstr.write(pSDBr)   
        
    print(f'Saved temp pSDBr file: {pSDBr_output}')
    blue, green, dstg, dstr = None, None, None, None
    
    return pSDBg, pSDBr, pSDBg_output, pSDBr_output, nans, shapes, dims, w, h, out_transform, out_meta

def reproject_toWGS84(land, shapes):
    # Load the shapefile
    gdf = gpd.read_file(land)
    source_crs = gdf.crs
    target_crs = pyproj.CRS.from_epsg('4326')
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    # Define the list of geometries (in your provided format)
    geometries = shapes
    
    # Reproject each geometry
    reprojected_geometries = []
    for geometry in geometries:
        coordinates = geometry['coordinates'][0]  # Get the coordinates
        reprojected_coordinates = [transformer.transform(x, y) for x, y in coordinates]  # Reproject
        reprojected_geometry = {'type': 'Polygon', 'coordinates': [reprojected_coordinates]}
        reprojected_geometries.append(reprojected_geometry)

    # print(f'Reprojected shapefile geometry from {source_crs} to WGS84 (EPSG 4326)')
    
    return reprojected_geometries

# shapes the feature inputs into a 2d array where each column is a 1d version of the 2d array/raster image
# returns the training data (x-inputs) and training labels (y-inputs) for fitting the model
# the 'points of interest' (poi) are simply nearshore points from the edge detection and ndwi filtering
def training_data_prep(pSDB_v, rgb_dir, truthiness_dir, etopo_full, land, truthiness):        
    # compute pSDBg
    in492, in560, in665 = wavelengths(rgb_dir)
    pSDBg, pSDBr, pSDBg_output, pSDBr_output, nans, shapes, dims, w, h, pSDB_transform, pSDB_meta = compute_pSDB(in492, in560, in665, land, truthiness_dir, truthiness)

    reprojected_geometries = reproject_toWGS84(land, shapes)
    
    print('Masking ETOPO to input shapefile geometry...')
    
    # do this before resampling
    with rasterio.open(etopo_full) as etopo:
        etopo_img, out_transform = rasterio.mask.mask(etopo, reprojected_geometries, crop=True)
        out_meta = etopo.meta
    
    # writing information
    out_meta.update({"driver": "GTiff",
                      "dtype": 'float32',
                      "height": etopo_img.shape[1],
                      "width": etopo_img.shape[2],
                      "nodata": 0,
                      "count": 1,
                      "transform": out_transform})
    
    etopo_img[np.isclose(etopo_img, 3.4e+38, atol=1e-6)] = 0
    
    cropped_etopo = etopo_full.replace('.tif', '_temp.tif')
    
    with rasterio.open(cropped_etopo, 'w', **out_meta) as dst:
        dst.write(etopo_img)   
        
    print(f'Saved temp cropped ETOPO file: {cropped_etopo}')
    etopo, dst = None, None
    
    etopo_dir, etopo_name = os.path.split(cropped_etopo)

    out_downsampledpSDBg = os.path.join(etopo_dir, 'dsampled_pSDBg.tif')
    out_downsampledpSDBr = os.path.join(etopo_dir, 'dsampled_pSDBr.tif')
    
    if not os.path.isfile(out_downsampledpSDBg):
        resampled_pSDBg = resample.resample(pSDBg_output, cropped_etopo, out_downsampledpSDBg)
        print(f'Resampled pSDBg: {resampled_pSDBg}')
        
        resampled_pSDBr = resample.resample(pSDBr_output, cropped_etopo, out_downsampledpSDBr)
        print(f'Resampled pSDBr: {resampled_pSDBr}')
    else:
        resampled_pSDBg = out_downsampledpSDBg
        print(f'Using pSDBg: {resampled_pSDBg}')
        resampled_pSDBr = out_downsampledpSDBr
        print(f'Using pSDBr: {resampled_pSDBr}')
    
    x_min, y_max = out_transform * (0, 0)  
    x_max, y_min = out_transform * (w, h)  
    extent = [x_min, x_max, y_min, y_max]
    
    plt.imshow(etopo_img[0,:,:], extent=extent, vmin=-20., vmax=0.)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('UTM Easting (m)', fontsize=8)
    plt.ylabel('UTM Northing (m)', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('ETOPO', fontsize=10)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('(EGM08 [m; heights])', fontsize=8)
    plt.show()
    
    plt.imshow(pSDBg[0,:,:], cmap='viridis_r', extent=extent)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('UTM Easting (m)', fontsize=8)
    plt.ylabel('UTM Northing (m)', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('pSDBg', fontsize=10)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('(Unitless)', fontsize=8)
    plt.show()
    
    if pSDB_v == 'green':
        pSDB = pSDBg
        with rasterio.open(resampled_pSDBg, 'r') as src:
            pSDB_resampled = src.read()
    elif pSDB_v == 'red':
        pSDB = pSDBr
        with rasterio.open(resampled_pSDBr, 'r') as src:
            pSDB_resampled = src.read()
    
    src = None 
    
    training_arr = np.reshape(pSDB_resampled, (-1, 1))
    training_labels = np.reshape(etopo_img, (-1, 1))
    
    training_arr = np.reshape(np.delete(training_arr, np.where(training_labels < -30)), (-1,1))
    training_labels = np.reshape(np.delete(training_labels, np.where(training_labels < -30)), (-1,1))
        
    extents = (x_min, y_min, x_max, y_max, extent)
    
    return pSDB, np.nan_to_num(training_arr), np.nan_to_num(training_labels), dims, extents, nans, pSDB_transform, pSDB_meta

    
# %% - regression

def distribution(data1, data2):
    plt.hist(data1, bins=50, alpha=0.5, label='pSDBg')
    plt.show()
    
    plt.hist(data2, bins=50, alpha=0.5, label='ETOPO')
    plt.show()    
    
    return None

def regression(pSDB_v, rgb, truthiness_dir, etopo, maskSHP, truthiness, reg_options, save_sdb, sdb_out_dir):
    # uses training_data_prep function above to shape the feature inputs
    # for training labels, 1s are nearshore pixels, and 0s are everything else   
    pSDB, training_arr, training_labels, dims, extents, nans, pSDB_transform, pSDB_meta = training_data_prep(pSDB_v, rgb, truthiness_dir, etopo, maskSHP, truthiness)
    
    x_train = np.reshape(np.delete(training_arr, np.where((training_labels == 0) | (training_arr == 0))), (-1, 1))
    y_train = np.reshape(np.delete(training_labels, np.where((training_labels == 0) | (training_arr == 0))), (-1, 1))
    y_train = y_train * -1 # heights to depths
    
    # distribution(x_train, y_train)    
    
    sdb_output = []
    
    if constrained_lsq:
        m1, m0 = constrained_lstsq(x_train, y_train)
        sdb_reg_method = 'Constrained Lsq'
    else:
        for reg in reg_options:
            print(f'\nTraining: {reg}')
            
            if 'SDGReg' in str(reg):
                reg = make_pipeline(StandardScaler(),reg)
            
            model = reg.fit(x_train, y_train)
            print('Metrics:')
            
            if 'RANSAC' in str(reg):
                m0 = model.estimator_.intercept_
                m1 = model.estimator_.coef_[0]
            else:
                m0 = model.intercept_
                m1 = model.coef_
            r2 = model.score(x_train, y_train)
            print(f'm0: {m0}')
            print(f'm1: {m1}')
            print(f'R2: {r2:.3f}')
            
            # print(f'Size: {x_train.size}, {y_train.size}')
            
            plt.scatter(x_train, y_train, s=0.8)
            plt.plot(x_train, m1*x_train + m0, color='red')
            # plt.text(1.1, -5., 'y = ' + '{:.2f}'.format(m0) + ' + {:.2f}'.format(m1) + 'x' + '\nr2='+'{:.2f}'.format(r2), size=8)
            plt.xlabel('pSDBg', fontsize=8)
            plt.ylabel('ETOPO (EGM08 [m])', fontsize=8)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.title('ETOPO vs. pSDBg - ' + str(reg).split('(')[0], fontsize=10)
            plt.show()
            
            sdb_reg_method = str(reg).split('(')[0]
                  
    pSDB = np.nan_to_num(np.reshape(pSDB, (-1, 1)))
    sdb = m1*pSDB + m0
    sdb = np.reshape(sdb, dims)[0,:,:]
    sdb[nans] = np.nan
    
    x_min, y_min, x_max, y_max, extent = extents
    
    # plot heatmap with fit line for all data
    ## pSDBg
    # 2d histogram on subset of data
    Hgreen, Gxedges, Gyedges = np.histogram2d(x_train[:,0], y_train[:,0], bins=150)
    # get the bin centers
    xgreen_centers = (Gxedges[:-1] + Gxedges[1:]) / 2
    ygreen_centers = (Gyedges[:-1] + Gyedges[1:]) / 2
    xgreen_grid, ygreen_grid = np.meshgrid(xgreen_centers, ygreen_centers)
    thresholdgreen = np.percentile(Hgreen, 98)  # percentile to keep
    densegreen_indices = np.where(Hgreen > thresholdgreen)
    dense_xgreen_values = xgreen_centers[densegreen_indices[0]]
    dense_ygreen_values = ygreen_centers[densegreen_indices[1]]
    
    # pSDBg linear regression
    modelG = linregress(dense_xgreen_values, dense_ygreen_values)
    m0green = modelG.intercept
    m1green = modelG.slope
    print(f'\nm0 heat: {m0green:.3f}')
    print(f'm1 heat: {m1green:.3f}')
    print(f'R2 heat:  {modelG.rvalue**2:.3f}')
    
    plt.hist2d(x_train[:,0], y_train[:,0], bins=150, cmap='viridis_r')
    plt.plot(x_train[:,0], m1green*x_train[:,0] + m0green, color='red', linewidth=0.75, label='Regression')
    # plt.plot(pSDBg_4m, m1green*pSDBg_4m - 57.486, color='white', linewidth=0.75, label='Zero Shoreline')
    plt.xlabel('pSDBg')
    plt.ylabel('ETOPO (m)')
    plt.title('pSDBg v. ETOPO')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Density')
    plt.legend(facecolor='grey')
    plt.show()
       
    plt.imshow(sdb, extent=extent, vmin=-20., vmax=0.)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('UTM Easting (m)', fontsize=8)
    plt.ylabel('UTM Northing (m)', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(sdb_reg_method + ' SDB', fontsize=10)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('(EGM08 [m; heights])', fontsize=8)
    plt.show()
    
    out_image = sdb
    pSDB_meta.update({"driver": "GTiff",
                      "height": out_image.shape[0],
                      "width": out_image.shape[1],
                      "nodata": 0,
                      "transform": pSDB_transform})
    
    sdb[nans] = 0
    
    if save_sdb:
        # location = os.path.basename(etopo_name).split('_')[0]
        sdb_dir = sdb_out_dir + '\\_SDB_Output'
        if not os.path.exists(sdb_dir):
            os.makedirs(sdb_dir)
        
        sdb_output_name = os.path.join(sdb_dir, '_' + sdb_reg_method + '_SDB_' + current_time.strftime('%Y%m%d_%H%M') + '.tif')
        
        with rasterio.open(sdb_output_name, 'w', **pSDB_meta) as dst:
            dst.write(out_image, 1)   
            
        print(f'Saved: {sdb_output_name}')
            
        dst = None
        
        sdb_output.append(sdb_output_name)
        
    if not sdb_output:
        return '\nNo output ...'
    else:
        return sdb_output


# %% - main

if __name__ == '__main__':
    
    # rgb_dir = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\_ETOPO\Nantucket_20180902"
    rgb_dir = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\_ETOPO\Matthews3_Composite_Green"
    # rgb_dir = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\_ETOPO\Matthews_20190818"
    shp_dir = [r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\_Bryan_SHP"]
    etopo = r"C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\ETOPO\All_ETOPO2022_15s_IceSurf_EXT_01_m100_LZW.tif"
    etopo_out = r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\ETOPO\_Out'
    truthiness_dir = r'P:\_RSD\Data\ETOPO\Truthiness'
    ground_truth = r'P:\_RSD\Data\ETOPO\Ground Truth'
    
    truthiness = False
    # truthiness = True
    compute_sdb = True
    # compute_sdb = False
    
    constrained_lsq = False
    # constrained_lsq = True
    # save_sdb = True
    save_sdb = False
    sdb_out_dir = r'C:\Users\Matthew.Sharr\Documents\NGS\SatBathy\Data\Testing\_ETOPO\_Out'
    # compare = True
    compare = False
    # pSDB_v = 'red'
    pSDB_v = 'green'
    
    # regression options
    # https://scikit-learn.org/stable/modules/linear_model.html#ransac-random-sample-consensus
    reg_options = [
                    LinearRegression(), 
                    # RANSACRegressor(random_state=42),
                    # TheilSenRegressor(random_state=42),
                    # HuberRegressor(),
                    # Ridge(alpha=1.0),
                    # BayesianRidge(),
                    # SGDRegressor(max_iter=500, tol=1e-3)
                   ]
    
    if compute_sdb:
        maskSHP_dir = list_inputs(rgb_dir, shp_dir, truthiness_dir)
        
        # for rgb in img_dirs:
            # rgb_composite(rgbnir_dir)
        for maskSHP in maskSHP_dir:
            if check_bounds(rgb_dir, maskSHP):
                # location = os.path.basename(rgb_dir)
                # etopo_name = etopo_out + '\\' + location + '_pSDBg_bilinear.tif'
                # print(etopo_name)
                
                sdb_output_names = regression(pSDB_v, rgb_dir, truthiness_dir, etopo, maskSHP, truthiness, reg_options, save_sdb, sdb_out_dir)
                
                print(sdb_output_names)
        
    if compare:
        # inlidar = r"P:\_RSD\Data\ETOPO\Ground Truth\CapeCod\usace2018_east_cst_dem_Job922111\usace2018_east_cst_dem_J922111.tif"
        # sdb = r"D:\ML\ETOPO\SDB\CapeCod_SDB_Output\CapeCod_RANSACRegressor_SDB.tif"
        # out_gt = r"P:\_RSD\Data\ETOPO\Ground Truth\CapeCod\CapeCod_lidar.tif"
        
        # sdb = r"P:\_RSD\Data\ETOPO\SDB\KeyWest_SDB_Output\KeyWest_TheilSenRegressor_SDB.tif"
        # inlidar = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyWest\2019_NGS_FL_topobathy_DEM_Irma_Job896369\2019_NGS_FL_topobathy_DEM_Irma_J896369.tif"
        # out_gt = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyWest\WestWest_lidar.tif"
        
        # sdb = r"P:\_RSD\Data\ETOPO\SDB\StCroix_SDB_Output\StCroix_LinearRegression_SDB.tif"
        # sdb = r"D:\ML\ETOPO\SDB\StCroix_SDB_Output\StCroix_RANSACRegressor_SDB.tif"
        # inlidar = r"P:\_RSD\Data\ETOPO\Ground Truth\StCroix\2019_ngs_topobathy_dem_usvi_Job921533\2019_ngs_topobathy_dem_usvi_J921533.tif"
        # out_gt = r"P:\_RSD\Data\ETOPO\Ground Truth\StCroix\StCroix_lidar.tif"
        
        # sdb = r"P:\_RSD\Data\ETOPO\SDB\KeyLargo_SDB_Output\KeyLargo_LinearRegression_SDB.tif"
        # inlidar = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\Large\Job780732_2019_NGS_FL_topobathy_DEM_Irma.tif"
        # out_gt = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\KeyLargo_lidar_resample_warp.tif"
        
        # sdb = r"P:\_RSD\Data\ETOPO\SDB\Ponce_SDB_Output\Ponce_RANSACRegressor_SDB.tif"
        # inlidar = r"P:\_RSD\Data\ETOPO\Ground Truth\Ponce\2019_ngs_topobathy_dem_pr_Job922086\2019_ngs_topobathy_dem_pr_J922086.tif"
        # out_gt = r"P:\_RSD\Data\ETOPO\Ground Truth\Ponce\Ponce_lidar.tif"
        
        # sdb = r"P:\_RSD\Data\ETOPO\SDB\KeyLargo_SDB_Output\KeyLargo_LinearRegression_SDB.tif"
        # inlidar = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\Large\Job780732_2019_NGS_FL_topobathy_DEM_Irma.tif"
        # out_gt = r"P:\_RSD\Data\ETOPO\Ground Truth\KeyLargo\KeyLargo_lidar_resample_warp.tif"
        
        sdb = r'P:\\_RSD\\Data\\ETOPO\\SDB\\Saipan_SDB_Output\\Saipan_LinearRegression_SDB.tif'
        inlidar = r"P:\_RSD\Data\ETOPO\Ground Truth\Saipan\cnmi2019_islands_dem_J922112.tif"
        out_gt = r"P:\_RSD\Data\ETOPO\Ground Truth\Saipan\Saipan_lidar_resample_warp.tif"
        
        if not os.path.isfile(out_gt):                
            outlidar = resample.resample(inlidar, sdb, out_gt)
        else:
            outlidar = out_gt
        
        mean_diff, std_dev_diff, rmse, rmse95 = resample.rmsez(sdb, outlidar)
        print(f'\nSDB vs Ground Truth Mean ± Std:\n{mean_diff:.3f} // {std_dev_diff:.3f}')
        print(f'\nSDB vs Ground Truth RMSEz // 95% Confidence:\n{rmse:.3f}m // {rmse95:.3f}m')
