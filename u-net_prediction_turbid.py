# -*- coding: utf-8 -*-
"""
Last updated Nov/03/2022

@ author: Sreenivas Bhattiprolu, ZEISS
@ modified by: Jaehoon Jung, PhD, OSU

Semantic segmentation using U-Net architecture (prediction)

Modified by Matt Sharr
"""

import datetime
import keras
from matplotlib import pyplot as plt
import numpy as np
import os
from osgeo import gdal
from patchify import patchify, unpatchify
# import tifffile as tiff
import time


# %% - functions

def plotPatches(im,row):
    plt.figure(figsize=(9, 9))
    square = im.shape[1]
    ix = 1
    for i in range(square):
    	for j in range(square):
    		ax = plt.subplot(square, square, ix)
    		ax.set_xticks([])
    		ax.set_yticks([])
    		plt.imshow(im[i+row, j, :, :], cmap='jet')
    		ix += 1
    
def padding(image,s_patch):
    h,w = np.shape(image)
    pad_row = (0, s_patch - (h % s_patch))
    pad_col = (0, s_patch - (w % s_patch))
    image = np.pad(image, [pad_row, pad_col], mode='constant', constant_values=0)
    return image,h,w

# def padding3D(image,s_patch):
#     h,w,d = np.shape(image)
#     pad_row = (0, s_patch - (h % s_patch))
#     pad_col = (0, s_patch - (w % s_patch))
#     image = np.pad(image, [pad_row, pad_col, (0,0)], mode='constant', constant_values=0)
#     return image,h,w

def padding3D(image,s_patch):
    # https://stackoverflow.com/questions/50008587/zero-padding-a-3d-numpy-array
    h,w,d = np.shape(image)
    pad_row = (0, s_patch - (h % s_patch))
    pad_col = (0, s_patch - (w % s_patch))
    dim = [pad_row, pad_col]
    for i in range(2,len(image.shape)):
        dim.append((0,0))    
    image = np.pad(image, dim, mode='constant', constant_values=0)
    return image,h,w

def plotImage(zi,t_cmap, filename):
    plt.figure()
    plt.clf()
    plt.imshow(zi, cmap=t_cmap)
    plt.colorbar()
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    # plt.axis('off')
    plt.title(filename, fontsize=11)
    # plt.savefig('Z:\\CE560\\HW3\\Report\\Images\\' + filename + '.png', dpi=300)
    plt.show()

def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def saveGTiff(im,gt,proj,i_ft,fileName):   
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    outds = driver.Create(fileName, 
                          xsize = im.shape[1],
                          ysize = im.shape[0], 
                          bands = 1, 
                          eType = gdal.GDT_UInt16
                          )
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outds.GetRasterBand(i_ft).WriteArray(im)
    outds.GetRasterBand(i_ft).SetNoDataValue(0) # np.nan
    outds.FlushCache()
    outds = None

# %% hyper parameters
s_patch = 128
s_step = 32 # 64


# %% - options
# write_prediction = True
write_prediction = False

# more turbid areas
predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\Chesapeake_20230316\\_Features_6Bands\\_Composite\\ChesapeakeBay_vCompositeTest_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\Florida_20230115\\_Features_6Bands\\_Composite\\FL_Keys_20230115Ex4C_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\Hatteras_20230127\\_Features_6Bands\\_Composite\\Hatteras_Inlet_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\Hatteras_20230206\\_Features_6Bands\\_Composite\\Hatteras_Inlet_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\Hatteras_20230507\\_Features_6Bands\\_Composite\\Hatteras_Inlet_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\Lookout_20230306\\_Features_6Bands\\_Composite\\CapeLookout_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\Lookout_20230507\\_Features_6Bands\\_Composite\\CapeLookout_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\Lookout_20230726\\_Features_6Bands\\_Composite\\CapeLookout_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\PuertoReal_20211203\\_Features_6Bands\\_Composite\\Puerto_Real_Smaller_6Bands_composite_20231213_1004.tif'
# predict_raster = 'P:\\Thesis\\Test Data\\_Turbid_Tests\\StCroix_20220129\\_Features_6Bands\\_Composite\\StCroix_20220129Ex3C_6Bands_composite_20231213_1004.tif'

input_model = r"P:\_RSD\Models\UNet\UNet_6bands_50epoch.hdf5"

IOU = False
# IOU = True

# test_mask = r"C:\_Thesis\Masks\Test\GreatLakes_Mask_NoLand_TF.tif"
# test_mask = r"C:\_Thesis\Masks\Test\Niihua_Mask_TF.tif"
test_mask = r"C:\_Thesis\Masks\Test\PuertoReal_Mask_TF.tif"
# test_mask = r"C:\_Thesis\Masks\Test\Saipan_Mask_NoIsland_TF.tif"


# %% load data and model
start_time = time.time()
print(f'Predicting on: {predict_raster}')
print(f'Using model: {input_model}')

o_image = gdal.Open(predict_raster)

#-- affine trasform coefficients 
gt = o_image.GetGeoTransform()
#-- projection of raster data
proj = o_image.GetProjection()

o_image = o_image.ReadAsArray().transpose((1,2,0))

print(f'\nInput image shape: {o_image.shape}')

o_image = np.nan_to_num(o_image)
o_image[o_image == -9999] = 0
# image = image.astype(np.float) / 255.  # scale dataset
# image = scaleData(o_image.astype(np.float)) # scale data

image = np.zeros(np.shape(o_image)) # default float64 (scaled array)
for i in range(0, np.shape(image)[2]):
    ft = o_image[:,:,i]
    # ft[(ft < np.mean(ft) - 3 * np.std(ft)) | (ft > np.mean(ft) + 3 * np.std(ft))] = 0
    image[:,:,i] = scaleData(ft.astype(np.float64))

image, h, w = padding3D(image,s_patch)
# image, h, w = padding(image,s_patch)

print(f'Padded image shape: {image.shape}')

# Trained U-Net model
model = keras.models.load_model(input_model, compile=False)

# %% prediction, patch by patch 
# patches = patchify(image, (s_patch, s_patch), step = s_step)#-- split image into small patches with overlap  
#-- plotPatches(patches,20) # plot patches
patches = patchify(image, (s_patch, s_patch, image.shape[2]), step=s_step) 
row, col, dep, hi, wi, d = patches.shape
patches = patches.reshape(row*col*dep, hi, wi, d)  

print(f'Number of patches: {patches.shape[0]}')

print('\nPredicting...')

patches_predicted = [] # store predicted images (196,128,128)
for i in range(patches.shape[0]): # loop through all patches
    if not i % 100:
        print("Now predicting on patch: ", i)
    patch1 = patches[i,:,:,:] # all rows, cols, dep of each patch (128,128,10)
    patch1 = np.expand_dims(np.array(patch1), axis=[0]) # expand first dimension to fit into model (1,128,128,10)
    patch1_prediction = model.predict(patch1) # predict on patch (1,128,128,3)
    patch1_predicted_img = np.argmax(patch1_prediction, axis=3)[0,:,:] # along class axis store maximum values (128,128)
    patches_predicted.append(patch1_predicted_img) # store patch prediction in list (196 in total)

patches_predicted = np.array(patches_predicted) # create array of each patch in list (196,128,128)
# need to figure out reshaping to 3D, this code works for reshaping from 2D
# reshaped size (196, 128, 128, 1)
patches_predicted_reshaped = np.reshape(patches_predicted, (row, col, s_patch, s_patch) ) #-- Gives a new shape to an array without changing its data
image_predicted = unpatchify(patches_predicted_reshaped, image.shape[0:2]) #-- merge patches into original image
image_predicted = image_predicted[:h,:w] #-- recover original image size
# #-- plot segmented image
# plotImage(o_image[0:2],'Greys_r', 'Original')
plotImage(image_predicted,'viridis', 'Classified')

print(f'\n--Completed U-Net prediction on {patches.shape[0]} patches in {(time.time() - start_time):.1f} seconds / {(time.time() - start_time)/60:.1f} minutes\n')

# classified image vs point cloud intensity

# patches_predicted_reshaped = np.reshape(patches_predicted, (image.shape[0], -1))
# plt.imshow(patches_predicted_reshaped)
# plt.colorbar()
# plt.show()

# image_height, image_width, channel_count = image.shape # (1734,1730,10)
# output_height = image_height - (image_height - s_patch) % s_step # [1734 - (1734 - 128) % 128] = 1664
# output_width = image_width - (image_width - s_patch) % s_step # [1730 - (1730 - 128) % 128] = 1664
# output_shape = (output_height, output_width) # (1664, 1664, 10)

if IOU:
    print('Performing intersection over union analysis...')
    # bandmask = tiff.imread(test_mask)
    bandmask = gdal.Open(test_mask).ReadAsArray()
    truth = np.nan_to_num(bandmask)
    truth[truth == -9999] = 0
    
    prediction1 = np.reshape(image_predicted, (image_predicted.shape[0] * image_predicted.shape[1]))
    truth1 = np.reshape(truth, (truth.shape[0] * truth.shape[1]))
    m = keras.metrics.MeanIoU(num_classes=3)
    m.update_state(truth1, prediction1)
    print('\--Mean IOU:', m.result().numpy())


# %% - Save output

if write_prediction:
    current_time = datetime.datetime.now()
    prediction_path = os.path.abspath(os.path.join(os.path.dirname(predict_raster), '..', '_UNet_Prediction'))
    
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    
    prediction_path = prediction_path + '\\' + os.path.basename(predict_raster).replace('composite.tif', 'prediction_') + current_time.strftime('%Y%m%d_%H%M') + '.tif'
    
    saveGTiff(image_predicted.astype(np.uint8), gt, proj, 1, prediction_path)
    
    print(f'\nWrote prediction to {prediction_path}')

#%%

# https://stackoverflow.com/questions/68249421/how-to-modify-patches-made-by-patchify

# for row in range(patches.shape[0]):
#     # for col in range(patches.shape[1]):
#     print("Now predicting on patch", row, col)
#     patch1 = patches[row,:,:,:]
#     patch1 = np.expand_dims(np.array(patch1), axis=[0])
#     patch1_prediction = model.predict(patch1)
#     patch1_predicted_img = np.argmax(patch1_prediction, axis=3)[0,:,:]
#     patches_predicted.append(patch1_predicted_img)

# for row in range(patches.shape[0]):
#     for col in range(patches.shape[1]):
#         print("Now predicting on patch", row, col)
#         patch1 = patches[row,col,:,:]
#         patch1 = np.expand_dims(np.array(patch1), axis=[0,3])
#         patch1_prediction = model.predict(patch1)
#         patch1_predicted_img = np.argmax(patch1_prediction, axis=3)[0,:,:]
#         patches_predicted.append(patch1_predicted_img)



# https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
# image_height, image_width, channel_count = image.shape
# patch_height, patch_width, step = 128, 128, 128
# patch_shape = (patch_height, patch_width, channel_count)
# patches = patchify(image, patch_shape, step=step)

# output_patches = np.empty(patches.shape).astype(np.uint8)
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         patch = patches[i, j, 0]
#         output_patch = model.predict(patch)  # process the patch
#         output_patches[i, j, 0] = output_patch

# image_height, image_width, channel_count = o_image.shape # (1734,1730,10)
# output_height = image_height - (image_height - s_patch) % s_step # [1734 - (1734 - 128) % 128] = 1664
# output_width = image_width - (image_width - s_patch) % s_step # [1730 - (1730 - 128) % 128] = 1664
# output_shape = (output_height, output_width, channel_count) # (1664, 1664, 10)
# output_image = unpatchify(output_patches, output_shape) 

# o_image = tiff.imread(r"P:\Thesis\Test Data\Puerto Real\_10Band\_Composite\Puerto_Real_Smaller_composite.tif")
# o_image = tiff.imread(r"P:\Thesis\Test Data\TinianSaipan\_10Band\_Composite\Saipan_Extents_composite.tif")
# o_image = tiff.imread(r"P:\Thesis\Test Data\TinianSaipan\_8Band_MaskChk\_Composite\Saipan_Mask_composite.tif")
# o_image = tiff.imread(r"P:\Thesis\Test Data\Puerto Real\_8Band\_Composite\Puerto_Real_Smaller_composite.tif")
# o_image = tiff.imread(r"P:\Thesis\Test Data\NWHI\_8Band\_Composite\NWHI_Extents_composite.tif")
# o_image = tiff.imread(r"P:\Thesis\Training\PuertoReal\_7Band\_Composite\Puerto_Real_Smaller_composite.tif")
# o_image = tiff.imread(r"P:\Thesis\Test Data\GreatLakes\_7Band_NoLand\_Composite\GreatLakes_Mask_NoLand_composite.tif")
# o_image = tiff.imread(r"P:\Thesis\Test Data\TinianSaipan\_7Band\_Composite\Saipan_Extents_composite.tif")
# o_image = tiff.imread(r'P:\Thesis\Test Data\A_Samoa\_7Band\_Composite\A_Samoa_Harbor_composite.tif')
# o_image = tiff.imread(r"P:\Thesis\Test Data\Niihua\_7Band\_Composite\Niihua_Mask_composite.tif")








