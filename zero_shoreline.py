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
from skimage import feature, filters
# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier #, HistGradientBoostingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, jaccard_score
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
def plot_rgb_poi(cropped_red, cropped_green, cropped_blue, prediction, ndwi, out_transform, w, h, brightness, aspect):
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
    
    ax[1].imshow(ndwi, cmap='Greys', extent=extent)
    ax[1].imshow(prediction, vmax=0.2, cmap='Reds', alpha=0.5, extent=extent)
    custom_legend = [Line2D([0], [0], color='maroon', lw=2, label='Prediction')]
    ax[1].legend(handles=custom_legend, loc='lower right')
    ax[1].set_title('Predicted Nearshore Pixels Overlaid on NDWI', fontsize=10)
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)
    ax[1].set_xlabel('UTM Easting (m)')
    ax[1].set_ylabel('UTM Northing (m)')
    ax[1].set_aspect(aspect)  # adjust the value as needed
    # ax[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    return None

# can give you a sense of decision tree
def plot_decision_tree():
    plt.figure(figsize=(12, 8))
    plot_tree(model.estimators_[0], feature_names=feature_list, class_names=['0','1'], filled=True, rounded=True)
    plt.show()


# %% - Training

# shapes the feature inputs into a 2d array where each column is a 1d version of the 2d array/raster image
# returns the training data (x-inputs) and training labels (y-inputs) for fitting the model
# the 'points of interest' (poi) are simply nearshore points from the edge detection and ndwi filtering
def training_data_prep(in492, in560, in665, in833, land):
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
        
    # compute ndwi and pSDBr
    ndwi = (green_img - nir_img) / (green_img + nir_img)
    pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)

    # perform edge detection on ndwi, and mask out areas (targeting land)
    canny_feat = feature.canny(ndwi[0,:,:], sigma=3)
    poi = np.where((canny_feat == 1) & (ndwi[0,:,:] > 0.1), 1, 0 )

    # shape feature array
    training_arr = np.vstack((ndwi.flatten(),
                            pSDBr.flatten(),
                            nir_img.flatten(),
                            red_img.flatten(), 
                            green_img.flatten(), 
                            blue_img.flatten()
                            )).transpose()
    
    return np.nan_to_num(training_arr), poi.flatten()

# trains classifier
def model_training(in492, in560, in665, in833, land, save_model, out_model):
    # uses training_data_prep function above to shape the feature inputs
    # for training labels, 1s are nearshore pixels, and 0s are everything else
    training_arr, poi_labels = training_data_prep(in492, in560, in665, in833, land)
    
    # because the data is so imbalanced (far fewer nearshore pixels than not), 'randomly' subsample the data
    # using 20 times the amount of non-nearshore pixels
    # using the same 'random' pixels every time for repeatability
    training_labels, training_selection = subsample(poi_labels, training_arr, 20)
    
    # scale data between 0 and 1, per column
    scaler = MinMaxScaler().fit(training_selection)
    scaled_water = scaler.transform(training_arr)
    x_train, x_test, y_train, y_test = train_test_split(training_selection, training_labels, test_size=0.33, random_state=42, stratify=training_labels)
    scaled_X_train = scaler.transform(x_train)
    scaled_X_test = scaler.transform(x_test)
    
    # classifier options
    clf = RandomForestClassifier(n_jobs=4, n_estimators=20, random_state=42) # 20 trees seemed to work well for me and is fast
    # clf = AdaBoostClassifier(random_state=42, learning_rate=1., base_estimator=RandomForestClassifier(n_jobs=4, n_estimators=20, random_state=42)) # can boost another classifier
    # clf = MLPClassifier(random_state=42, max_iter=300, hidden_layer_sizes=(10,10,10,10,10)) # hidden layers worked well for me
    # clf = HistGradientBoostingClassifier(random_state=42, max_iter=500, learning_rate=0.2, l2_regularization=0.2) # seems to overpredict
    # clf = GradientBoostingClassifier(n_estimators=50, random_state=42) # comperable to Hist above
    # clf = svm.SVC(random_state=42, kernel='rbf', gamma='auto') # underpredicts and is much slower -- probably not useful operationally
    
    # fit/train classifier/model
    print(f'Training: {clf}\n')
    model = clf.fit(scaled_X_train, y_train)
    
    # training metrics
    print(classification_report(y_test, model.predict(scaled_X_test)))
    print(f'Jaccard (IOU) Score: {jaccard_score(y_test, model.predict(scaled_X_test), average="macro"):.3f}')
    
    # this is to get a sense of feature importance from random forest.
    # ****comment out code if using a model other than random forest
    feature_list = ['ndwi', 'pSDBr', 'nir', 'red', 'green', 'blue']
    feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False).round(3)
    print(f'\nFeature Importance:\n{feature_importance}')
    
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
        
    # compute ndwi and pSDBr
    ndwi = (green_img - nir_img) / (green_img + nir_img)
    pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)

    # perform edge detection on ndwi, and mask out areas (targeting land)
    canny_feat = feature.canny(ndwi[0,:,:], sigma=3)
    poi = np.where((canny_feat == 1) & (ndwi[0,:,:] > 0.1), 1, 0 )

    # shape feature array
    test_arr = np.vstack((ndwi.flatten(),
                            pSDBr.flatten(),
                            nir_img.flatten(),
                            red_img.flatten(), 
                            green_img.flatten(), 
                            blue_img.flatten()
                            )).transpose()
    
    test_arr = np.nan_to_num(test_arr)
    # scale data between 0 and 1
    test_scaler = MinMaxScaler().fit(test_arr)
    test_scaled = test_scaler.transform(test_arr)
    
    print('\nPredicting...')
    prediction = model.predict(test_scaled).reshape((ndwi[0,:,:].shape))
    
    # print median results
    print('\nPlotting results...')
    nearshore_pixels = pSDBr[0,:,:][prediction == 1]
    median_prediction = np.median(nearshore_pixels)
    # median_ndwi = np.median(ndwi[0,:,:][prediction == 1])
    print(f'\nMedian (pSDBr): {median_prediction:.3f}')
    # print(f'Median (ndwi): {median_ndwi:.3f}')
    print(f'Count: {nearshore_pixels.size:,}')
    
    # plot rgb and results
    plot_rgb_poi(red_img, green_img, blue_img, prediction, ndwi[0,:,:], out_transform, w, h, brightness=3, aspect=1.)
    
    # plot histogram
    plt.hist(pSDBr[0,:,:][prediction == 1], bins=50)
    plt.title('Distribution of pSDBr in Predicted Nearshore Area')
    plt.xlabel('pSDBr Value')
    plt.ylabel('Count')
    plt.show()
    
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
    # in492 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_704.tif"
    # in833 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\StCroix_Zero.shp"
    
    # train
    # out_model = os.path.join(r'C:\_ZeroShoreline\Model', 'RF_20trees_TrainedStCroix.pkl')
    # model = model_training(in492, in560, in665, in833, land, save_model=True, out_model=out_model)
    
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
    in_model = r"C:\_ZeroShoreline\Model\RF_20trees_TrainedStCroix.pkl"
    with open(in_model, 'rb') as f:
        model = pickle.load(f)
    
    # predict 
    prediction(in492, in560, in665, in833, land, model, out_prediction=None, write_prediction=False)