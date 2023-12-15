# -*- coding: utf-8 -*-
"""
Satellite image classification 

@author: Jaehoon Jung, PhD, OSU

last updated: Oct 10, 2022
"""

from osgeo import gdal 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import numpy as np
import time
import pickle
import tkinter as tk
import shapefile as shp
import pandas as pd
from tkinter import filedialog
from yellowbrick.classifier import ROCAUC
from datetime import datetime
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow import config
from tensorflow.keras import regularizers
from pytictoc import TicToc 
tt = TicToc()
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # TODO: uncomment this and restart kernel to not use GPU # check before shipping 

# from tensorflow import random
# random.set_seed(2)

print('GPU:', config.list_physical_devices('GPU'))

# %% functions 
def assessAccuracy(features, model, classifier, Y, row, col, s_window=2):
    acc = 0
    #-- iterate the evaluation over the number of testing data (Y)
    for i in range(len(Y)):
        X_new = features[row[i]-s_window:row[i]+s_window+1, col[i]-s_window:col[i]+s_window+1,:]
        X_new = np.reshape(X_new,(-1,8))
        if classifier == 'NN':
            #-- argmax identifies the maximum value for every row and return its index
            Y_new = np.argmax(model.predict(X_new),axis=1) 
        elif classifier == 'RF':
            Y_new = model.predict(X_new)            
        if Y[i] in Y_new:
            acc += 1 
    return acc/len(Y)
            

def findGeoIntersection(raster1,raster2,band1_id,band2_id):
    # https://sciience.tumblr.com/post/101722591382/finding-the-georeferenced-intersection-between-two
    
    # load data
    band1 = raster1.GetRasterBand(band1_id)
    band2 = raster2.GetRasterBand(band2_id)
    gt1 = raster1.GetGeoTransform()
    gt2 = raster2.GetGeoTransform()
    
    # find each image's bounding box
    # r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * raster1.RasterXSize), gt1[3] + (gt1[5] * raster1.RasterYSize)]
    r2 = [gt2[0], gt2[3], gt2[0] + (gt2[1] * raster2.RasterXSize), gt2[3] + (gt2[5] * raster2.RasterYSize)]
    # print('\t1 bounding box: %s' % str(r1))
    # print('\t2 bounding box: %s' % str(r2))
    
    # find intersection between bounding boxes
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]
    if r1 != r2:
        # print('\t** different bounding boxes **')
        # check for any overlap at all...
        if (intersection[2] <= intersection[0]) or (intersection[1] <= intersection[3]):
            intersection = None
            print('\t***no overlap***')
            return
        else:
            # print('\tintersection:',intersection)
            left1 = int(round((intersection[0]-r1[0])/gt1[1])) # difference divided by pixel dimension
            top1 = int(round((intersection[1]-r1[1])/gt1[5]))
            col1 = int(round((intersection[2]-r1[0])/gt1[1])) - left1 # difference minus offset left
            row1 = int(round((intersection[3]-r1[1])/gt1[5])) - top1
            
            left2 = int(round((intersection[0]-r2[0])/gt2[1])) # difference divided by pixel dimension
            top2 = int(round((intersection[1]-r2[1])/gt2[5]))
            col2 = int(round((intersection[2]-r2[0])/gt2[1])) - left2 # difference minus new left offset
            row2 = int(round((intersection[3]-r2[1])/gt2[5])) - top2
            
            #print '\tcol1:',col1,'row1:',row1,'col2:',col2,'row2:',row2
            if col1 != col2 or row1 != row2:
                print("*** MEGA ERROR *** COLS and ROWS DO NOT MATCH ***")
            # these arrays should now have the same spatial geometry though NaNs may differ
            array1 = band1.ReadAsArray(left1,top1,col1,row1)
            array2 = band2.ReadAsArray(left2,top2,col2,row2)

    else: # same dimensions from the get go
        col1 = raster1.RasterXSize # = col2
        row1 = raster1.RasterYSize # = row2
        array1 = band1.ReadAsArray()
        array2 = band2.ReadAsArray()
        
    return array1, array2, col1, row1, intersection

def getTime():
    now = datetime.now()
    return str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second) 
    
def getSHPattribute(sf,name):
    i = 0
    print(name)
    for f in sf.fields:  
        
        # print(f[0])
        if name == f[0]:
            break
        i+=1
    records = sf.records()
    # print(records)
    atr = []
    for r in records:
        atr.append(r[i-1]) 
        
    # print(atr)
    return np.array(atr)

def getData(fileName,im,gt,n_ft):
    sf = shp.Reader(fileName)
    Lat_m = getSHPattribute(sf,'Lat_m')
    Lon_m = getSHPattribute(sf,'Long_m')
    Y_train = getSHPattribute(sf,'Habitat')
    Y_train = LabelEncoder().fit_transform(Y_train) #Encode Y values to 0, 1, 2, 3,... 
    # print(Y_train)
    s_cell = gt[1]    # cellsize 
    col = np.floor((Lon_m - gt[0])/s_cell) # + 1
    row = np.floor((gt[3] - Lat_m)/s_cell) # + 1
    col = col.astype(int)
    row = row.astype(int)
    X_train = []
    for i in range(n_ft):
        im_tmp = im[:,:,i]
        X_train.append(im_tmp[row,col])
    X_train = np.transpose(X_train)
    # print(X_train)
    return X_train, Y_train, s_cell, row, col

def plotImage(zi,labels,cmap):
    #-- add legend: https://bitcoden.com/answers/how-to-add-legend-to-imshow-in-matplotlib
    plt.figure()
    plt.imshow(zi)
    plt.grid(False)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
    plt.legend(handles=patches, loc=4)
    plt.show()

def scaleData(data):
    print(np.min(data), np.max(data))
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
    outds.GetRasterBand(i_ft).SetNoDataValue(np.nan)
    outds.FlushCache()
    outds = None
    
def saveImage(img,featureName,fileName,labels,cmap,p_dpi):
    #-- bbox_to_anchor: https://stackoverflow.com/questions/40908983/arguements-of-bbox-to-anchor-function
    plt.figure()
    plt.imshow(img)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title(featureName)
    plt.grid(False)
    patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
    plt.legend(handles=patches, loc=4, bbox_to_anchor=(1.5, 0))
    plt.savefig(fileName + '.png', dpi=p_dpi)
    plt.clf()
    plt.close()


# %% feature list    
feature_list = ["CoastalBlue",      #1
                "Blue",             #2   
                "Green",            #3  
                "Yellow",           #4
                "Red",              #5
                "RedEdge",          #6
                "Near_Infrared_1",  #7  
                "Near_Infrared_2"]  #8

Habitat_class = ["Mixed Algae and Seagrass on Sand, ", #1
                 "Seagrass on Sand (Halodule uninervis)", #2
                 "Bare Sand", #3
                 "Seagrass on Sand (Enhalus acoroides)" #4
                 "Pavement Colonized with Mixed Algae", #5
                 "Coral Rubble Colonized with Mixed Algae", #6
                 "Live and Upright Dead Coral with Mixed Algae", #7
                 ]

labels = {1:'Mixed Algae',
        2:'Seagrass (HU)',
        3:'Bare Sand',
        4:'Seagrass (EA)',
        5:'Pavement',
        6:'Coral Rubble',
        7:'Live Coral'
        }

cmap = {1:[205/255, 205/255, 102/255, 1],
        2:[76/255, 127/255, 0/255, 1], 
        3:[255/255, 255/255, 190/255, 1], 
        4:[38/255, 82/255, 0/255, 1],  
        5:[0/255, 112/255, 255/255, 1], 
        6:[115/255, 76/255, 0/255, 1], 
        7:[230/255, 152/255, 0/255, 1] 
        }

def colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [255/255, 255/255, 255/255, 1] # unclassified 
    rgba[data==1, :] = [205/255, 205/255, 102/255, 1] 
    rgba[data==2, :] = [76/255, 127/255, 0/255, 1] 
    rgba[data==3, :] = [255/255, 255/255, 190/255, 1] 
    rgba[data==4, :] = [38/255, 82/255, 0/255, 1]  
    rgba[data==5, :] = [0/255, 112/255, 255/255, 1] 
    rgba[data==6, :] = [115/255, 76/255, 0/255, 1] 
    rgba[data==7, :] = [230/255, 152/255, 0/255, 1] 
    return rgba

# %% read data 
root = tk.Tk()
filePath = filedialog.askopenfilename() 
filePath = os.path.split(filePath)[0]
root.withdraw()   

start_time = time.time()

ds_mask = gdal.Open(filePath + "/Saipan_Mask_2m.tif")
#-- affine trasform coefficients 
#-- In case of north up images, the GT(2) and GT(4) coefficients are zero, and the GT(1) is pixel width, and GT(5) is pixel height. 
#-- The (GT(0),GT(3)) position is the top left corner of the top left pixel of the raster.
gt = ds_mask.GetGeoTransform()
#-- projection of raster data
proj = ds_mask.GetProjection()
ds_feature = gdal.Open(filePath + "/Satellite_Image_WV2_2016Feb05_resample_2m.tif")

# %% convert to feature array  
features = []
#-- Every cell location in a raster has a value assigned to it. 
#-- When information is unavailable for a cell location, the location will be assigned as NoData. 
upper_limit = np.finfo(np.float32).max/10
lower_limit = np.finfo(np.float32).min/10
for i, __ in enumerate(feature_list):
    ft, mask, __, __, bb = findGeoIntersection(ds_feature, ds_mask, i+1, 1)
    ft[(ft < lower_limit) | (ft > upper_limit)] = 0
    ft[mask==0] = 0 
    features.append(scaleData(ft))
features = np.array(features)
features = np.moveaxis(features,0,-1) # np.ndstack is slow  
gt_intesec = (bb[0],gt[1],gt[2],bb[1],gt[4],gt[5])

# %% read ground truth data
#-- shapefile is a geospatial vector data format for GIS software
X_train, Y_train, __, row_train, col_train = getData(r'Z:\CE560\HW2\Input Data\GV_Points_Presence.shp', features, gt_intesec, features.shape[2])
X_test, Y_test, __, row_test, col_test = getData(r'Z:\CE560\HW2\Input Data\AA_Points_Presence.shp', features, gt_intesec, features.shape[2])

# %% Hyper-parameters  
# classifier = 'NN'
classifier = 'RF' 
training = True 
# training = False
s_window = 2


# %% classification with Neural Network

depth = 256 # 128

#-- model training
if training:
    model = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dense(depth, input_dim=8, activation="relu"), # kernel_initializer='he_uniform',
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(depth, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(7, activation="softmax")
        ])

    #-- the output values of softmax are predicted probabilities between 0 - 1, their sum is 1   
    #-- The word ‘stochastic‘ means a system or process is based on a random probability.
    opt1 = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9) 
    opt2 = keras.optimizers.Adam(learning_rate=0.01)
    #-- A metric is a function that is used to judge the performance of your model.
    #-- Metric functions are similar to loss functions, but they are not used when training the model.
    model.compile(optimizer=opt2, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #-- "validation_data" is to check if the model is overfitting. It doesn't afftect the training 
    #-- "validation_split=0.2" allows you to automatically reserve part of your training data (20% in this example) for validation. 
    history = model.fit(X_train,Y_train, epochs=100, batch_size=15, validation_data=(X_test,Y_test), verbose=2)
   
    #-- accuracy assessment 
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    test_acc *= 100
    print ("\nValidation Accuracy= %.3f %%" % test_acc)  
   
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy: Nodes = {depth}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title(f'Model loss: Nodes = {depth}') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()

else:
    root = tk.Tk()
    ModelPath = filedialog.askopenfilename() 
    ModelPath = os.path.split(ModelPath)[0]
    root.withdraw()  
    model = keras.models.load_model(ModelPath + "/NN_model.hdf5", compile=False) # use pre-trained model    
    
#-- accuracy assessment 
acc2 = assessAccuracy(features, model, classifier, Y_test, row_test, col_test, s_window)*100    
print("Validation Accuracy= %.3f %% within %d pixels" % (acc2, s_window))

# -- prediction 
tt.tic()
rc = np.argwhere(mask>0) # return row and column of aoi 
print('rc:', np.shape(rc))
X_new = features[rc[:,0],rc[:,1],:]
print('X New:', np.shape(X_new))
im_predicted = np.zeros((mask.shape))
im_predicted[rc[:,0],rc[:,1]] = np.argmax(model.predict(X_new),axis=1)+1
print('im:',np.shape(im_predicted))
# plotImage(colorMap(im_predicted),labels,cmap)  
tt.toc()

#-- save outputs
filePath = filePath + '/NN_' + getTime() 
os.makedirs(filePath) # create a new folder to save outputs  
# model.save(filePath + '/NN_model.hdf5')
# saveImage(colorMap(im_predicted),"NN_habitat_classification",filePath + '/NN_prediction',labels,cmap,3000) 
# saveGTiff(im_predicted.astype(np.uint8), gt_intesec, proj, 1, filePath + "/NN_prediction.tif")

print("--- Total elapsed time: %.3f seconds ---" % (time.time() - start_time))























