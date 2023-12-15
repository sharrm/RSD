# -*- coding: utf-8 -*-
"""
Last updated Nov/03/2022

@ author: Sreenivas Bhattiprolu, ZEISS
@ modified by: Jaehoon Jung, PhD, OSU

Semantic segmentation using U-Net architecture (training)

Modified by Matt Sharr
"""

import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from patchify import patchify
import datetime


#-- Building Unet by dividing encoder and decoder into blocks
def conv_block(input, num_filters):
    x = keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    x = keras.layers.BatchNormalization()(x)    
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)  
    x = keras.layers.Activation("relu")(x)
    return x

#-- Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    m = keras.layers.MaxPool2D((2, 2))(x)
    return x, m   

#-- Decoder block: skip features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
    x = keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#-- Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = keras.layers.Input(input_shape)
    e1, m1 = encoder_block(inputs, 64)
    e2, m2 = encoder_block(m1, 128)
    e3, m3 = encoder_block(m2, 256)
    e4, m4 = encoder_block(m3, 512)
    b1 = conv_block(m4, 1024) #Bridge
    d1 = decoder_block(b1, e4, 512)
    d2 = decoder_block(d1, e3, 256)
    d3 = decoder_block(d2, e2, 128)
    d4 = decoder_block(d3, e1, 64)
    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'
    outputs = keras.layers.Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)
    model = keras.models.Model(inputs, outputs, name="U-Net")
    return model

def plotAccuracy(loss,acc,title): 
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'y', label= title + 'loss')
    plt.title(title + 'loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(epochs, acc, 'r', label= title + 'accuracy')
    plt.title(title + 'accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

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

def plotImage(zi,t_cmap):
    plt.figure()
    # plt.clf()
    plt.imshow(zi, cmap=t_cmap)
    plt.colorbar()
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.show()
    
def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
            
def padding(image,s_patch):
    h,w = np.shape(image)
    pad_row = (0, s_patch - (h % s_patch))
    pad_col = (0, s_patch - (w % s_patch))
    image = np.pad(image, [pad_row, pad_col], mode='constant', constant_values=0)
    return image,h,w

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


# %% hyper parameters
n_classes = 8 #-- number of classes for segmentation (no data, false, true, turbid) # changed from 3 to 4
n_images = 330  #-- number of subset  # 330
s_patch = 128 #-- patch size # 128
s_step = 32 #-- step size, use smaller step size for overlap # 64
s_batch = 16 # batch size 16
eps = 500 # epochs


# %% - data


composite_rasters = [
                        'P:\\Thesis\\Training\\_Turbid_Training\\Lookout_20230306\\_Features_6Bands\\_Composite\\Lookout_UNet_8C_6Bands_composite_20231212_1602.tif'
                       ]

training_rasters = [
                        r'P:\Thesis\Test Data\_Manuscript_Test\Masks\Lookout_UNet_8C_TF.tif'
    
                      ]

model_name = r'P:\_RSD\Models\UNet\UNet_6bands_50epoch.hdf5'


# %% load data  
#-- read image data
all_patches_list = []

for comp in composite_rasters:
    # print(comp)
    o_image = tiff.imread(comp)    
    o_image = np.nan_to_num(o_image)
    o_image[o_image == -9999] = 0
    
    #-- scale, pad, and patchify
    image = np.zeros(np.shape(o_image)) # default float64 (scaled array)
    for i in range(0, np.shape(image)[2]):
        ft = o_image[:,:,i]
        image[:,:,i] = scaleData(ft.astype(np.float64))
    
    image, __, __, = padding3D(image,s_patch) #-- to divide image by s_patch 
    patches = patchify(image, (s_patch, s_patch, image.shape[2]), step=s_step) 
    row, col, dep, h, w, d = patches.shape
    patches = patches.reshape(row*col*dep, h, w, d) 
    patches = patches[:n_images,:,:] #-- use subset if input data is too large
    all_patches_list.append(patches) 
    # patches = patches # n_images, row, col, bands
    # image = patches[:n_images,:,:] #-- use subset if input data is too large
    # image = np.expand_dims(image, axis = 3) #-- expand dimension to make the right format for neural network 

all_patches = np.vstack(all_patches_list)

all_masks_list = []

for tf_mask in training_rasters:
    #-- read mask data (ground truth)
    o_mask = tiff.imread(tf_mask)
    mask, _, _ = padding(o_mask,s_patch)
    mask = patchify(mask, (s_patch, s_patch), step=s_step) 
    r, c, h, w = mask.shape
    mask = mask.reshape(r*c, h, w)
    mask = mask[:n_images,:,:] #-- use subset if input data is too large
    # mask = np.expand_dims(mask, axis = 3)
    all_masks_list.append(mask)
    
all_masks = np.vstack(all_masks_list)


# %% training 
#-- split training data
# X_train, X_test, Y_train, Y_test = train_test_split(patches, mask, test_size = 0.2)
X_train, X_test, Y_train, Y_test = train_test_split(all_patches, all_masks, test_size = 0.3)
#-- one-hot encoding
#-- deep learning reads a higher number as more important than a lower number
#-- this can lead to issues if data do not have any ranking for category values
#-- alternatively, one hot encoder performs “binarization” of the category  
train_masks_cat = to_categorical(Y_train, num_classes=n_classes)
Y_train_cat = train_masks_cat.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], n_classes))
test_masks_cat = to_categorical(Y_test, num_classes=n_classes)
Y_test_cat = test_masks_cat.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], n_classes))
print(Y_train_cat.shape)
#-- train Model
model = build_unet((X_train.shape[1],X_train.shape[2],X_train.shape[3]), n_classes) #-- input_shape (img height, img width, img channels)
#-- report metrics during the training of model
#-- unlike the loss, metrics is not used when training the model
#-- use 'categorical crossentropy' for the one-hot encoder 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
model.summary()
start_time = time.time()
history = model.fit(X_train, Y_train_cat, 
                    batch_size = s_batch, #-- large batch size can casue memory issue
                    verbose=1, 
                    epochs=eps, 
                    validation_data=(X_test, Y_test_cat))
print("\n--- %.3f seconds ---\n" % (time.time() - start_time))
#-- save the model for future use
current_time = datetime.datetime.now()
model.save(model_name)
#-- plot the training and validation accuracy and loss at each epoch
#-- note that your results may vary given the stochastic nature of the algorithm
#-- run the code a few times and compare the results
plotAccuracy(history.history['loss'], history.history['accuracy'], 'Training ')
plotAccuracy(history.history['val_loss'], history.history['val_accuracy'], 'Validation ')

print(f'patches: {s_patch}\nstep: {s_step}\nbatch size: {s_batch}')
print(f'\nSaved U-Net model here: {model_name}')


# %% original code pieces
# image = tiff.imread(r"Z:\CE560\HW3\HWY081_intensity.tiff")

#-- test array
# image = np.array([[[0,1,2],[3,4,5],[6,7,8]],
# [[20,21,22],[23,24,25],[26,27,28]],
# [[30,31,32],[33,34,35],[36,37,38]],
# [[59,58,57],[56,55,54],[53,52,51]],
# [[9,10,11],[12,13,14],[15,16,17]]])
# image = image.reshape(-1, image.shape[-1])

#-- original code
# image = image.astype(np.float) /255.  #-- normalize dataset
# image = patchify(image, (s_patch, s_patch), step=s_step) #-- split image into small overlappable patches
# plotPatches(image,10) # plot patches 
# r, c, h, w = image.shape 
# image = image.reshape(r*c, h, w) #-- stack patches 
# image = image[:n_images,:,:] #-- use subset if input data is too large







