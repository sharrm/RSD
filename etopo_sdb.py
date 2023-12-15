# -*- coding: utf-8 -*-
"""
September 2023
@author: matthew.sharr

Zero Shoreline:
Objective is to identify nearshore pixels, adjacent to land, that can be used
to determine the m0 (y-intercept) for band ratio SDB.
"""

import fiona
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
import rasterio.mask
from scipy import spatial
from scipy import ndimage
from scipy.optimize import lsq_linear
from skimage import feature, filters
# from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, jaccard_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree
import warnings


# %% - globals and functions

warnings.filterwarnings('ignore')

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


# %% - Training

# shapes the feature inputs into a 2d array where each column is a 1d version of the 2d array/raster image
# returns the training data (x-inputs) and training labels (y-inputs) for fitting the model
# the 'points of interest' (poi) are simply nearshore points from the edge detection and ndwi filtering
def training_data_prep(in492, in560, in665, in833, etopo, land):
    with fiona.open(land, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(in492) as blue:
        blue_img, out_transform = rasterio.mask.mask(blue, shapes, crop=True)
        out_meta = blue.meta

    with rasterio.open(in560) as green:
        green_img, out_transform = rasterio.mask.mask(green, shapes, crop=True)
        out_meta = green.meta
        
    with rasterio.open(in665) as red:
        red_img, out_transform = rasterio.mask.mask(red, shapes, crop=True)
        out_meta = red.meta
        
    with rasterio.open(in833) as nir:
        nir_img, out_transform = rasterio.mask.mask(nir, shapes, crop=True)
        out_meta = nir.meta
        w = nir.width
        h = nir.height
        
    with rasterio.open(etopo) as etopo:
        etopo_img, out_transform = rasterio.mask.mask(etopo, shapes, crop=True)
        out_meta = etopo.meta
        
    pSDBg = np.log(blue_img * 1000) / np.log(green_img * 1000)
    pSDBg = np.where(pSDBg == 0, np.nan, pSDBg)
    shape = pSDBg[0,:,:].shape

    # shape feature array
    # training_arr = np.vstack((
    #                         pSDBg.flatten(),
    #                         red_img.flatten(), 
    #                         green_img.flatten(), 
    #                         blue_img.flatten()
    #                         )).transpose()
    
    nans = np.where(etopo_img[0,:,:] == 0)
    etopo_img = np.where(etopo_img == 0, np.nan, etopo_img)
    
    x_min, y_max = out_transform * (0, 0)  
    x_max, y_min = out_transform * (w, h)  
    extent = [x_min, x_max, y_min, y_max]
    
    plt.imshow(etopo_img[0,:,:], extent=extent)
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
    
    training_arr = np.reshape(pSDBg, (-1, 1))
    training_labels = np.reshape(etopo_img, (-1, 1))
    
    extents = (x_min, y_min, x_max, y_max, extent)
    
    return np.nan_to_num(training_arr), np.nan_to_num(training_labels), shape, extents, nans

# trains classifier
def model_training(in492, in560, in665, in833, etopo, land, save_model, out_model):
    # uses training_data_prep function above to shape the feature inputs
    # for training labels, 1s are nearshore pixels, and 0s are everything else   
    training_arr, training_labels, shape, extents, nans = training_data_prep(in492, in560, in665, in833, etopo, land)
       
    # scale data between 0 and 1, per column
    scaler = MinMaxScaler().fit(training_arr)
    scaled_water = scaler.transform(training_arr)
    x_train, x_test, y_train, y_test = train_test_split(training_arr, training_labels, test_size=0.3, random_state=42)
    scaled_X_train = scaler.transform(x_train)
    scaled_X_test = scaler.transform(x_test)
    
    # classifier options
    # reg = RandomForestRegressor(n_jobs=4, random_state=42) # 20 trees seemed to work well for me and is fast
    reg = LinearRegression()
    
    # fit/train classifier/model
    print(f'Training: {reg}\n')
    x_train = np.reshape(np.delete(x_train, np.where(x_train == 0)), (-1, 1))
    y_train = np.reshape(np.delete(y_train, np.where(y_train == 0)), (-1, 1))
    x_test = np.reshape(np.delete(x_test, np.where(x_test == 0)), (-1, 1))
    y_test = np.reshape(np.delete(y_test, np.where(y_test == 0)), (-1, 1))
    
    lb, ub = -47., -45.
    
    print(x_train.shape, y_train.shape)
    x_train = x_train.reshape(-1)
    y_train = y_train.reshape(-1)
    print(x_train.shape, y_train.shape)
          
    model = reg.fit(x_train, y_train)
    # model = reg.fit(scaled_X_train, y_train)
    
    print('model:')
    m0 = model.intercept_
    m1 = model.coef_
    r2 = model.score(x_train, y_train)
    print(f'm0: {m0[0]:.3f}')
    print(f'm1: {m1[0][0]:.3f}')
    print(f'R2: {r2:.3f}')
    
    res = lsq_linear(x_train, y_train, m1, bounds=(lb, ub), lsmr_tol='auto', verbose=1)

        
    plt.scatter(x_train, y_train, s=0.8)
    plt.plot(x_train, m1*x_train + m0, color='red')
    plt.text(1.1, -5., 'y = ' + '{:.2f}'.format(m0[0]) + ' + {:.2f}'.format(m1[0][0]) + 'x' + '\n[r2='+'{:.2f}'.format(r2)+']', size=8)
    plt.xlabel('pSDBg', fontsize=8)
    plt.ylabel('ETOPO (EGM08 [m])', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('ETOPO vs. pSDBg', fontsize=10)
    plt.show()
    
    # training metrics
    predictions = model.predict(x_test) 
    print('\nprediction metrics:')
    print(f'mean squared_error : {mean_squared_error(y_test, predictions):.3f}') 
    print(f'mean absolute_error : {mean_absolute_error(y_test, predictions):.3f}') 
    print(f'R2: {r2_score(y_test, predictions):.3f}')
    
    # plt.scatter(x_test, predictions, s=0.8)
    # plt.plot(x_test, m1*x_test + m0, color='red')
    # plt.xlabel('pSDBg (test)', fontsize=8)
    # plt.ylabel('ETOPO (EGM08 [m]) (test)', fontsize=8)
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)
    # plt.title('ETOPO vs. pSDBg (test)', fontsize=10)
    # plt.show()
    
    sdb = m1*training_arr + m0
    sdb = np.reshape(sdb, shape)
    sdb[nans] = np.nan
    
    x_min, y_min, x_max, y_max, extent = extents
    
    plt.imshow(sdb, extent=extent)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('UTM Easting (m)', fontsize=8)
    plt.ylabel('UTM Northing (m)', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('SDB', fontsize=10)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('(EGM08 [m; heights])', fontsize=8)
    plt.show()
    
    # save trained model
    if save_model:
        with open(out_model, 'wb') as f:
            pickle.dump(model, f)
            
        print(f'\nSaved model to {out_model}')
        
    return model


# %% - Predicting

# predicts nearshore pixels and plot/print results
# can write prediction to raster
def prediction(in492, in560, in665, in833, land, model, out_prediction, write_prediction):
    # could have reused the same training function but kept everything here for plotting purposes
    with fiona.open(land, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(in492) as blue:
        blue_img, out_transform = rasterio.mask.mask(blue, shapes, crop=True)
        out_meta = blue.meta

    with rasterio.open(in560) as green:
        green_img, out_transform = rasterio.mask.mask(green, shapes, crop=True)
        out_meta = green.meta
        
    with rasterio.open(in665) as red:
        red_img, out_transform = rasterio.mask.mask(red, shapes, crop=True)
        out_meta = red.meta
        
    with rasterio.open(in833) as nir:
        nir_img, out_transform = rasterio.mask.mask(nir, shapes, crop=True)
        out_meta = nir.meta
        w = nir.width
        h = nir.height
        
    # compute pSDBg
    pSDBg = np.log(blue_img * 1000) / np.log(green_img * 1000)
    
    test_arr = np.reshape(pSDBg, (-1, 1))
    
    # test_arr = np.vstack((
    #                         pSDBg.flatten(),
    #                         red_img.flatten(), 
    #                         green_img.flatten(), 
    #                         blue_img.flatten()
    #                         )).transpose()
    
    test_arr = np.nan_to_num(test_arr)  
    
    print('\nPredicting...')
    prediction = model.predict(test_arr).reshape((pSDBg[0,:,:].shape))
    
    print(prediction.shape)
       
    # plot rgb and results
    plot_rgb_poi(red_img, green_img, blue_img, prediction, pSDBg[0,:,:], out_transform, w, h, brightness=3, aspect=1.)
    
    out_image = prediction
    out_meta.update({"driver": "GTiff",
                      "height": out_image.shape[0],
                      "width": out_image.shape[1],
                      "transform": out_transform})
    
    if write_prediction:
        with rasterio.open(out_prediction, 'w', **out_meta) as dst:
            dst.write(out_image, 1)   
            
        dst = None
        
    return None


# %% - main

if __name__ == '__main__':
    
    # input training data
    in492 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
    in560 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_560.tif"
    in665 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_665.tif"
    in704 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_704.tif"
    in833 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_833.tif"
    etopo = r"P:\_RSD\Data\ETOPO\aligned-ETOPO_StCroix_bilinear.tif"
    land = r"P:\_RSD\Data\ETOPO\SHP\etopo_stcroix_test.shp"
    
    # train
    out_model = os.path.join(r'C:\_ZeroShoreline\Model', 'RF_20trees_TrainedStCroix.pkl')
    model = model_training(in492, in560, in665, in833, etopo, land, save_model=False, out_model=out_model)
    
    # test data
    # in492 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_704.tif"    
    # in833 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet_FocusedExtent.shp"
    # land = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_559.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet_FocusedExtent.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\StCroix_Zero.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\Halfmoon_Zero.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\FL_Zero2.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20211201\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20211201\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20211201\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_665.tif"
    # # in704 = r
    # in833 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20211201\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\FL_Zero2.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\Saipan_Zero.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_559.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\Ponce_Zero.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\CapeLookout.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\WakeIsland_20221223\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\WakeIsland_20221223\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_559.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\WakeIsland_20221223\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_665.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\WakeIsland_20221223\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\WakeIsland_Zero.shp"

    # in492 = r"C:\_ZeroShoreline\Imagery\Nihau\S2B_MSI_2022_01_28_21_19_22_T04QCK_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\Nihau\S2B_MSI_2022_01_28_21_19_22_T04QCK_L2R_rhos_559.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\Nihau\S2B_MSI_2022_01_28_21_19_22_T04QCK_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\Nihau\S2B_MSI_2022_01_28_21_19_22_T04QCK_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\Nihau\S2B_MSI_2022_01_28_21_19_22_T04QCK_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\Niihua4.shp"
    
    # to load a saved model, 
    # in_model = r"C:\_ZeroShoreline\Model\RF_20trees_TrainedStCroix.pkl"
    # with open(in_model, 'rb') as f:
    #     model = pickle.load(f)
    
    # predict 
    # prediction(in492, in560, in665, in833, land, model, out_prediction=None, write_prediction=False)