# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:19:30 2023

@author: sharrm


"""

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
from rasterio.mask import mask
from scipy import spatial
from scipy import ndimage
from skimage import feature, filters
from skimage.morphology import binary_dilation, binary_erosion
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier 
from sklearn.metrics import accuracy_score#, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score 
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# %% - morphology

def normalize(band):
    band_min, band_max = (np.nanmin(band), np.nanmax(band))
    return ((band-band_min)/((band_max - band_min)))

def near_land(input_blue, input_green, input_red, input_704, input_nir, shapefile, out_dir, write):
    
    # Open the geotiff file
    with rasterio.open(input_green) as green:
        # Read the green band metadata
        out_meta = green.meta
        
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_green, transform = mask(green, gdf.geometry, crop=True)
    
    # Open the geotiff file
    with rasterio.open(input_nir) as nir:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_nir, transform = mask(nir, gdf.geometry, crop=True)
        
    # Open the geotiff file
    with rasterio.open(input_704) as b704:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_704, transform = mask(b704, gdf.geometry, crop=True)        
        
    # Open the geotiff file
    with rasterio.open(input_blue) as blue:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_blue, transform = mask(blue, gdf.geometry, crop=True)
            
    # Open the geotiff file
    with rasterio.open(input_red) as red:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_red, transform = mask(red, gdf.geometry, crop=True)            
    
    # compute ndwi
    ndwi = (cropped_green - cropped_nir) / (cropped_green + cropped_nir)
    cropped = np.moveaxis(ndwi, 0, -1)[:,:,0]
    
    # compute pSDBr
    pSDBr = np.log(cropped_blue * 1000) / np.log(cropped_red * 1000)
    pSDBg = np.log(cropped_blue * 1000) / np.log(cropped_green * 1000)  
    
    # create binary array for land and water pixels
    nan_vals = np.where(np.isnan(cropped))
    cropped_land_water = np.where(cropped < 0.1, 1, 0)
    
    # morphological operation to grow land pixels
    morphed_land = binary_dilation(cropped_land_water) #.astype(cropped_land_water.dtype))
    erode_land = binary_erosion(morphed_land) #.astype(cropped_land_water.dtype))
    
    # pixels adjacent to land
    zero_mask = np.logical_and(morphed_land, ~erode_land)
    land_adjacent_ndwi = np.where(zero_mask, cropped, 0)    
    # land_adjacent_ndwi = np.where(land_adjacent_ndwi < 0.15, 0, land_adjacent_ndwi)
    # land_adjacent_percentile = np.where(np.percentile(land_adjacent_ndwi, 90), land_adjacent_ndwi, 0)
    percentile10 = np.nanpercentile(cropped[zero_mask == 1], 10)
    print(f'Precentile 10: {percentile10:.3f}')
    percentile10 = np.where(land_adjacent_ndwi < percentile10, land_adjacent_ndwi, 0)
    
    percentile90 = np.nanpercentile(cropped[zero_mask == 1], 90)
    print(f'Precentile 90: {percentile90:.3f}')
    percentile90 = np.where(land_adjacent_ndwi > percentile90, land_adjacent_ndwi, 0)
    
    # ndwi values for pixels adjacent to land for histogram
    ndwi_adjacent = cropped[zero_mask == 1]
    print(f'Average land adjacent NDWI value: {np.nanmean(ndwi_adjacent):.3f} ± {np.nanstd(ndwi_adjacent):.3f}')
    print(f'Median land adjacent NDWI value: {np.nanmedian(ndwi_adjacent):.3f}')
    land_adjacent_ndwi[nan_vals] = np.nan
    percentile10[nan_vals] = np.nan
    percentile90[nan_vals] = np.nan

    red_n = normalize(cropped_red[0,:,:])
    green_n = normalize(cropped_green[0,:,:])
    blue_n = normalize(cropped_blue[0,:,:])
    
    rgb_composite_n = np.dstack((red_n, green_n, blue_n))
    
    # Stack the bands to create an RGB image
    rgb_image = np.dstack((cropped_red[0,:,:], cropped_green[0,:,:], cropped_blue[0,:,:]))
    brightened_image = np.clip(rgb_composite_n * 3, 0, 255)#.astype(np.uint8)
    brightened_image[nan_vals] = 255
    m = np.ma.masked_where(np.isnan(brightened_image),brightened_image)
    
    # plt.figure(figsize=(10, 10))
    f, ax = plt.subplots(2,2, figsize=(10, 6), dpi=200)
    ax[0,0].imshow(brightened_image)
    ax[0,0].set_title('RGB', fontsize=10)
    ax[0,1].imshow(land_adjacent_ndwi, vmax=0.1, cmap='cividis')
    ax[0,1].set_title('Land Adjacent Pixels', fontsize=10)
    ax[1,0].imshow(percentile10, vmax=0.01, cmap='cividis')
    ax[1,0].set_title('10th Percentile', fontsize=10)
    ax[1,1].imshow(percentile90, vmax=0.05, cmap='cividis')
    ax[1,1].set_title('90th Percentile', fontsize=10)
    # plt.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.show()
    
    # ndwi values for pixels adjacent to land for histogram
    ndwi_adjacent = cropped[zero_mask == 1]
    print(f'Average land adjacent NDWI value: {np.nanmean(ndwi_adjacent):.3f} ± {np.nanstd(ndwi_adjacent):.3f}')
    print(f'Median land adjacent NDWI value: {np.nanmedian(ndwi_adjacent):.3f}')
    land_adjacent_ndwi[nan_vals] = np.nan
        
    training_data = np.vstack((cropped_blue.flatten(), 
                               cropped_green.flatten(), 
                               cropped_red.flatten(),
                               cropped_704.flatten(),
                               cropped_nir.flatten(), 
                               ndwi.flatten(), 
                               # pSDBg.flatten(),
                               pSDBr.flatten())).transpose()
    training_data[np.isnan(training_data)] = 2
    
    # Plot the masked image
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cropped, cmap='gray', vmin=0.2)
    plt.title('Land Adjacent Pixels')
    plt.imshow(zero_mask, cmap='Reds', alpha=0.3, vmax=0.2)
    plt.colorbar()
    
    # Plot histogram of values
    plt.subplot(2, 2, 2)
    plt.hist(ndwi_adjacent, bins=50, edgecolor='k')
    plt.xlabel('NDWI Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of NDWI Values at Land Adjacent Pixels')
    plt.tight_layout()
    plt.show()
    
    land_adjacent_ndwi[nan_vals] = 2
    
    # raster meta
    out_meta.update({"driver": "GTiff",
                      "height": cropped_nir.shape[1],
                      "width": cropped_nir.shape[2],
                      "count": cropped_nir.shape[0],
                      "nodata": 2,
                      "transform": transform})
    
    # save rasters    
    if write:
        morph_name = os.path.join(out_dir, 'morphed.tif')
        with rasterio.open(morph_name, "w", **out_meta) as dest:
            dest.write(morphed_land, 1)
        
        dest = None
        
        ndwi_name = os.path.join(out_dir, 'ndwi.tif')
        with rasterio.open(ndwi_name, "w", **out_meta) as dest:
            dest.write(cropped, 1)
        
        dest = None
        
        print(f'Wrote: {ndwi_name}')
        
        water_name = os.path.join(out_dir, 'land_adjacent.tif')
        with rasterio.open(water_name, "w", **out_meta) as dest:
            dest.write(land_adjacent_ndwi, 1)
        
        dest = None
        
        percentile10_name = os.path.join(out_dir, 'percentile10.tif')
        with rasterio.open(percentile10_name, "w", **out_meta) as dest:
            dest.write(percentile10, 1)
        
        dest = None
        
        print(f'Wrote: {percentile10_name}')
        
        percentile90_name = os.path.join(out_dir, 'percentile90.tif')
        with rasterio.open(percentile90_name, "w", **out_meta) as dest:
            dest.write(percentile90, 1)
        
        dest = None
        
        print(f'Wrote: {percentile90_name}')
    
    return land_adjacent_ndwi, training_data


# %% - training

# plots the learning curve -- relationship between prediction accuracy and data size
def plotLearningCurve(train_size_abs, train_mean, train_std, test_mean, test_std, curve_title):
    plt.plot(train_size_abs, train_mean, color='forestgreen', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_size_abs, train_mean + train_std, train_mean - train_std, alpha=0.3, color='forestgreen')
    plt.plot(train_size_abs, test_mean, color='royalblue', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
    plt.fill_between(train_size_abs, test_mean + test_std, test_mean - test_std, alpha=0.3, color='royalblue')
    plt.title(curve_title)
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy (f1-score)')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    
    return None

# computes and plots learning curve
def compute_learning_curve(clf, x_train, y_train):

    # start_time = time.time() # start time for process timing
    cv = StratifiedKFold(n_splits=5)
    print(f'\nComputing learning curve for {clf}.')
    
    train_size_abs, train_scores, test_scores = learning_curve(
    clf, x_train, y_train, cv=cv, scoring='f1_macro', 
    train_sizes=np.linspace(0.1, 1., 10), random_state=42)
        
    # Calculate training and test mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the learning curve
    print(f'--Plotting learning curve for {clf}.')
    plotLearningCurve(train_size_abs, train_mean, train_std, test_mean, test_std, curve_title=clf)
    print(f'Test accuracy:\n{test_mean}')
    print(f'Test standard deviation:\n{test_std}')
        
    return None

def subsample(array1, array2, adjustment):   
    # get indices of rows containing 0 and 1
    indices_with_zeros = np.where(array1 == 0)[0]
    indices_with_ones = np.where(array1 == 1)[0]
    
    # randomly select a subset of rows containing 0
    num_rows_to_select = np.count_nonzero(array1 == 1) * adjustment # Adjust as needed
    rng = np.random.default_rng(0)
    selected_indices_zeros = rng.choice(indices_with_zeros, size=num_rows_to_select, replace=False)
    
    # include the randomly selected rows
    selected_rows_array1 = np.concatenate((array1[indices_with_ones], array1[selected_indices_zeros]))
    selected_rows_array2 = np.vstack((array2[indices_with_ones], array2[selected_indices_zeros]))
    
    return selected_rows_array1, selected_rows_array2

def train_model(water_vals, training_data):
    labels = np.where((water_vals != 0) & (water_vals != 2), 1, water_vals)
    
    water_vals_1d = training_data
    labels_1d = labels.flatten()
    
    print(f'\nTraining else values: {np.count_nonzero(labels_1d == 0)}')
    print(f'Water labels: {np.count_nonzero(labels_1d == 1)}')
    print(f'Nan labels: {np.count_nonzero(labels_1d == 2)}')
    
    # water_vals_1d = np.delete(water_vals_1d, np.where(training_data == 2), axis = 0)
    # labels_1d = np.delete(labels_1d, np.where(training_data == 2), axis=0)
    
    # subsample()
    
    print(f'\nTrainData Shape: {water_vals_1d.shape}\nLabels Shape: {labels_1d.shape}')    
    
    X_train, X_test, Y_train, Y_test = train_test_split(water_vals_1d, labels_1d, 
                                                        test_size=0.3, random_state=40, stratify=labels_1d)
    
    scaler = MinMaxScaler().fit(water_vals_1d)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'\nX Train Shape: {X_train_scaled.shape}\nY_train Shape: {Y_train.shape}')
    print(f'Water labels: {np.count_nonzero(Y_train == 1)}\n')
    
    clf = RandomForestClassifier(random_state=42, n_jobs=4, n_estimators=50)
    # clf = HistGradientBoostingClassifier(random_state=42, max_iter=500, learning_rate=0.1, max_depth=5)
    # clf = MLPClassifier(random_state=42, max_iter=300, hidden_layer_sizes=(30,30,30))
    # clf = svm.SVC(C=1.0, class_weight='balanced', random_state=42)
    
    # X_learn_scaled = scaler.transform(water_vals_1d)
    # compute_learning_curve(clf, X_learn_scaled, labels_1d)
    
    print(f'Training {clf}')
    model = clf.fit(X_train_scaled, Y_train)
    
    # feature_list = ['blue', 'green', 'red', '704', 'nir', 'ndwi', 'pSDBr']
    
    # feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False).round(3)
    # print(f'\nFeature Importance:\n{feature_importance}\n')
    
    print('--Computing Precision, Recall, F1-Score...')
    classification = classification_report(Y_test, model.predict(X_test_scaled), labels=model.classes_)
    print(f'--Classification Report:\n{classification}')
    
    return water_vals_1d, labels_1d, model

def save_model(model_dir, model_name, model):
    model_name = os.path.join(model_dir, model_name)
    pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
    
    print(f'Saved model: {model_name}')
    
    return None

# %% - prediction

def predict(test_blue, test_green, test_red, test_704, test_nir, shapefile, model):
    # Open the geotiff file
    with rasterio.open(test_green) as green:
        # Read the green band metadata
        prediction_meta = green.meta
        
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_green, transform = mask(green, gdf.geometry, crop=True)
    
    # Open the geotiff file
    with rasterio.open(test_nir) as nir:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_nir, transform = mask(nir, gdf.geometry, crop=True)
        
    # Open the geotiff file
    with rasterio.open(test_704) as b704:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_704, transform = mask(b704, gdf.geometry, crop=True)          
        
    # Open the geotiff file
    with rasterio.open(test_blue) as blue:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_blue, transform = mask(blue, gdf.geometry, crop=True)
            
    # Open the geotiff file
    with rasterio.open(test_red) as red:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_red, out_transform = mask(red, gdf.geometry, crop=True)           
    
    # compute ndwi
    ndwi = (cropped_green - cropped_nir) / (cropped_green + cropped_nir)
    
    # compute pSDBr
    pSDBr = np.log(cropped_blue * 1000) / np.log(cropped_red * 1000)  
    pSDBg = np.log(cropped_blue * 1000) / np.log(cropped_green * 1000)  
    
    # shape prediction data
    test_data = np.vstack((cropped_blue.flatten(), 
                               cropped_green.flatten(), 
                               cropped_red.flatten(),
                               cropped_704.flatten(),
                               cropped_nir.flatten(), 
                               ndwi.flatten(), 
                               # pSDBg.flatten(),
                               pSDBr.flatten())).transpose()
    
    scaler = MinMaxScaler().fit(test_data)
    # scaler = StandardScaler().fit(test_data)
    scaled = scaler.transform(test_data)
    scaled[np.isnan(scaled)] = 2
        
    prediction = model.predict(scaled)
    prediction_shape = cropped_red.shape
    
    print(f'\nPrediction (0) values: {np.count_nonzero(prediction == 0)}')
    print(f'Prediction (1) values: {np.count_nonzero(prediction == 1)}')
    
    return prediction_shape, prediction, prediction_meta, pSDBr, out_transform

def plot_prediction(prediction, prediction_shape, pSDBr):
    # reshape
    img = np.reshape(prediction, prediction_shape)
    img = np.moveaxis(img, 0, -1)[:,:,0]
    
    # pSDBr = np.moveaxis(pSDBr, 0, -1)[:,:,0]
    # mask = np.ma.masked_where(img != 1, img)
    img = np.where(img == 2, np.nan, img)
    
    fig = plt.figure()
    # plt.imshow(pSDBr, cmap='gray')
    # plt.imshow(mask, cmap='hot', alpha=0.7)
    plt.imshow(img, cmap='viridis')
    plt.title('Prediction')
    plt.colorbar()
    plt.show()
    
    return img.shape

def save_prediction(prediction, pSDBr, prediction_shape, prediction_meta, out_dir, out_transform):
        
    prediction_name = os.path.join(out_dir, '_prediction.tif')
    pSDBr_name = os.path.join(out_dir, '_pSDBr.tif')
    img = np.reshape(prediction, prediction_shape)
    # img = np.ma.masked_where(img == 1, img)
    
    # raster meta
    prediction_meta.update({"driver": "GTiff", 
                            "height": prediction_shape[1],
                            "width": prediction_shape[2],
                            "count": prediction_shape[0],
                            "nodata": 2, 
                            "transform": out_transform})
    
    # save rasters    
    with rasterio.open(prediction_name, "w", **prediction_meta) as dest:
        dest.write(img) # had to specify '1' here for some reason
        dest = None
        
    print(f'\nSaved prediction to: {prediction_name}')
    
    # save rasters    
    with rasterio.open(pSDBr_name, "w", **prediction_meta) as dest:
        dest.write(pSDBr) # had to specify '1' here for some reason
        dest = None
        
    print(f'Saved pSDBr to: {pSDBr_name}')

    return None


# %% - main

if __name__ == '__main__':   
    out_dir = r"C:\_ZeroShoreline\Out\Testing"
    model_dir = r'C:\_ZeroShoreline\Model'
    model_name = 'RF_BGR7NWpR_StCroix.pkl'
    
    # in492 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_704.tif"    
    # in833 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_833.tif"
    # land = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet_FocusedExtent.shp"

    in492 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
    in560 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_560.tif"
    in665 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_665.tif"
    in704 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_704.tif"
    in833 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_833.tif"
    land = r"C:\_ZeroShoreline\Extent\StCroix_Zero.shp"

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
    # land = r"C:\_ZeroShoreline\Extent\FL_Zero.shp"
    
    water_vals, training_data = near_land(in492, in560, in665, 
                                          in704, in833, land, out_dir, write=False)
    water_vals_1d, labels_1d, model = train_model(water_vals, training_data)
    # save_model(model_dir, model_name, model)
    
    # in492 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_492.tif"
    # in560 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_560.tif"
    # in665 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_665.tif"
    # in704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_704.tif"    
    # in833 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_833.tif"
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
    # land = r"C:\_ZeroShoreline\Extent\FL_Zero.shp"
    
    prediction_shape, prediction, prediction_meta, pSDBr, out_transform = predict(in492, in560, in665, 
                                          in704, in833, land, model)
    img_shape = plot_prediction(prediction, prediction_shape, pSDBr)
    # save_prediction(prediction, pSDBr, prediction_shape, prediction_meta, out_dir, out_transform)