# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:43:02 2023

@author: sharrm

Updated: 20Mar2023
"""

# %% - pkgs

import datetime
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
# from pytictoc import TicToc
from scipy.ndimage import median_filter #, gaussian_filter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, jaccard_score, roc_curve, f1_score, recall_score, precision_score 
from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from xgboost import XGBClassifier
import sys
import time
# from yellowbrick.classifier import ROCAUC


# %% - globals

rcParams['figure.dpi'] = 600 # matplotlib figure dpi
np.random.seed(42)
random_state = 42
n_jobs = 5 # number of cores to use in sklearn processes

current_time = datetime.datetime.now() # current time for output file names
log_file = r"P:\Thesis\Training\_Logs\training_" + current_time.strftime('%Y%m%d_%H%M') + '.txt' # log file

#-- When information is unavailable for a cell location, the location will be assigned as NoData. 
upper_limit = np.finfo(np.float32).max/10
lower_limit = np.finfo(np.float32).min/10


# %% - functions

# plots correlation matrix for feature inputs
def correlation_matrix(correlation, df):
    # print(correlation)
    
    plt.matshow(correlation, cmap='cividis') # viridis cividis
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=-60)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=30)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=12);
    plt.show()
    return None

# scales data between 0 and 1
def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# logs print statements to an output file
def log_output(in_string):
    f = open(log_file, 'a')
    f.write(in_string)
    f.close()
    return None

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

# pairs composite images with their labels; returns a tuple of pairs
def pair_composite_with_labels(training_rasters, training_labels):
    training_list = []
    for comp in training_rasters:
        for label in training_labels:
            if rasterio.open(comp).bounds == rasterio.open(label).bounds:
                training_list.append((comp,label))
            else:
                continue
        # [print(f'Did not find matching label for: {comp}') for i in training_list if comp not in i[0]]
    return training_list

# reads a .tif image and returns the image, metadata, and data boundary
def read_image(image):
    img = rasterio.open(image)
    metadata = img.meta
    bounds = img.bounds
    
    composite_arr = img.read().transpose((1,2,0))
    img = None
    return composite_arr, metadata, bounds

# shapes multi-band image array and returns training data ready for sklearn
def shape_feature_array(composite_arr):
    features = []
    
    # for i, __ in enumerate(feature_list):
    for i, __ in enumerate(range(0, composite_arr.shape[2], 1)):
         ft = composite_arr[:,:,i] # take ith band
         ft[ft == -9999.] = 0. # set '-9999.' values to 0.
         ft[ft == -99999.] = 0. # set '-9999.' values to 0.
         ft = np.nan_to_num(ft) # set nan values to 0.
         ft[(ft < lower_limit) | (ft > upper_limit)] = 0. # value precision
         features.append(scaleData(ft)) # scale data and append to features list
         
    x_train = np.array(features).transpose((1,2,0)) # shape features list into array
    x_train = [x_train[:,:,i].ravel() for i in range(x_train.shape[2])] # stack features into columns
    return x_train    

# shapes a single composite image for sklearn training input
def shape_single_composite(composite_string):
    print('Preparing training data...')
    x_img, x_metadata, x_bounds = read_image(composite_string)
    x_train = np.vstack(shape_feature_array(x_img)).transpose()
    return x_train, x_metadata, x_bounds

# shapes multiple composites for sklearn training input
def shape_multiple_composites(composite_list):
    all_features = []
    x_metadata = []
    x_bounds = []
    
    for comp in composite_list:
        features, metadata, bounds = read_image(comp[0])
        feature_arr = shape_feature_array(features)
    
        all_features.append(np.array(feature_arr).transpose())
        x_metadata.append(metadata)
        x_bounds.append(bounds)
        
        print(f'Added {comp[0]} to X_train truthiness training data set. Shape: {features.shape}')
        log_output(f'\nAdded {comp[0]} to X_train truthiness training data set.')

        features = None
        
    x_train = np.vstack(all_features)
    return x_train, x_metadata, x_bounds

# shapes the label data array for sklearn
def shape_single_label(label_arr):
    labels, y_metadata, y_bounds = read_image(label_arr)
    y_train = labels.ravel()
    return y_train, y_metadata, y_bounds

# shapes multiple labels for sklearn
def shape_multiple_labels(label_list):
    all_labels = []
    y_metadata = []
    y_bounds = []
    
    for label in label_list:
        labels, metadata, bounds = read_image(label[1])
        label_arr = labels.ravel()
    
        all_labels.append(np.array(label_arr).transpose())
        y_metadata.append(metadata)
        y_bounds.append(bounds)
        
        print(f'Added {label[1]} to Y_train truthiness training data set. Shape: {labels.shape}')
        log_output(f'\nAdded {label[1]} to Y_train truthiness training data set.')

        labels = None
        
    y_train = np.concatenate(all_labels)
    return y_train, y_metadata, y_bounds

# checks training data input sizes match
def check_data_sizes(x_train, y_train, X_train, X_test, Y_train, Y_test):
    print('\nData summary:')
    print(f'--X_train pixels: {x_train.size:,}\n--Y_train pixels: {y_train.size:,}')
    print(f'--X_train shape: {X_train.shape}\n--Y_train shape: {Y_train.shape}\n--X_test shape: {X_test.shape}\n--Y_test shape: {Y_test.shape}')
    
    print('\nVerifying number of rows in training data and labels match')
    log_output('\nVerifying number of rows in training data and labels match')
    row_check = (X_train.shape[0] + X_test.shape[0]) - (Y_train.shape[0] + Y_test.shape[0])

    if row_check != 0:
        print('X and Y training/test row number mismatch...')
        log_output('\n\nX and Y training/test row number mismatch...')
        
        return False
    else:
        print(f'--X_train + X_test (row check): {X_train.shape[0] + X_test.shape[0]:,}')
        print(f'--Y_train + Y_test (row check): {Y_train.shape[0] + Y_test.shape[0]:,}')
        print('--Rows match...')
        
        log_output(f'--X_train + X_test (row check): {X_train.shape[0] + X_test.shape[0]}'
                   f'--Y_train + Y_test (row check): {Y_train.shape[0] + Y_test.shape[0]}'
                   '--Rows match...'
                   f'\n')
        return True
    
# verifies boundaries of training data and labels overlap
def check_metadata(x_bounds, y_bounds, x_metadata, y_metadata):
    if x_bounds != y_bounds:
        print('Mismatch between training data and labels boundaries...')
        return False
        if x_metadata['crs'] != y_metadata['crs'] or x_metadata['transform'] != y_metadata['transform']:
            print('Mismatch between training data and training label metadata...')
            return False
            if x_metadata['nodata'] != y_metadata['nodata']:
                print('Mismatch between training data and training labels no data values...')
                return False
    else:
        return True

# training label statistics
def label_stats(y_):
    no_data_value = np.count_nonzero(y_ == 0)
    true_negatives = np.count_nonzero(y_ == 1)
    true_positives = np.count_nonzero(y_ == 2)
    vessels = np.count_nonzero(y_ == 3)
    turbid = np.count_nonzero(y_ == 4)

    print('\nTraining label breakdown:')
    print(f'--Percent No Data: {no_data_value / y_.size:1f} ({no_data_value:,} No Data Values)')
    print(f'--Percent False: {true_negatives / y_.size:1f} ({true_negatives:,} False Values)')
    print(f'--Percent True: {true_positives / y_.size:1f} ({true_positives:,} True Values)')
    print(f'--Percent Vessels: {vessels / y_.size:1f} ({vessels:,} Vessel Values)')
    print(f'--Percent Turbid: {turbid / y_.size:1f} ({turbid:,} Turbid Values)')
    
    
    log_output(f'--Percent True: {true_positives / y_.size:1f} ({true_positives:,} True Values)')
    log_output(f'--Percent False: {true_negatives / y_.size:1f} ({true_negatives:,} False Values)')
    log_output(f'--Percent No Data: {no_data_value / y_.size:1f} ({no_data_value:,} No Data Values)')
    return None

# training accuracy
def assess_accuracy(model, X_train, Y_train, X_test, Y_test, RF, feature_list):
    print('\nAssessing accuracy...') 
    
    # print('--Computing Precision, Recall, F1-Score...')
    # classification = classification_report(Y_test, model.predict(X_test), labels=model.classes_)
    # print(f'--Classification Report:\n{classification}')
    
    # print('\nCreating confusion matrix...')
    cm = confusion_matrix(Y_test, model.predict(X_test), labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cividis')
    plt.title('Confusion Matrix')
    plt.show()
    
    acc1 = accuracy_score(Y_test, model.predict(X_test)) * 100.0
    print (f'--Validation Accuracy: {acc1:.2f} %') 
    prc = precision_score(Y_test, model.predict(X_test), average='macro')
    print (f'--Precision: {prc:.3f}') 
    rcll = recall_score(Y_test, model.predict(X_test), average='macro')
    print (f'--Recall: {rcll:.3f}') 
    f1 = f1_score(Y_test, model.predict(X_test), average='macro')
    print (f'--F1-Score: {f1:.3f}') 
    
    iou_score = jaccard_score(Y_test, model.predict(X_test), average=None) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
    print(f'--IOU: {iou_score}')
    # log_output(f'--Precision: {prc:.3f}\n--Recall: {rcll:.3f}\n--F1-Score: {f1:.3f}\n--Mean IOU: {iou_score:.3f}')
    
    # print('\nPlotting ROC AUC...')
    # roc_auc=ROCAUC(model, classes=np.unique(Y_train))
    # roc_auc.fit(X_train, Y_train)
    # roc_auc.score(X_test, Y_test)
    # roc_auc.show()
    
    if RF:
        oob_error = 1 - model.oob_score_
        print(f'--Out-of-Bag Error: {oob_error*100:.2f} %')
        
        feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False).round(3)
        print(f'\nFeature Importance:\n{feature_importance}')
        # log_output(f'\n--Random forest Validation Accuracy= {acc1:.2f} %'
                    # f'\nFeature Importance:\n{feature_importance}')
        
        df = pd.DataFrame(X_train, columns=feature_list)
        correlation = df.corr()
        # print(df.corr()['Chl_a'])
        # print(df.corrwith(df['Chl_a']))
        correlation_matrix(correlation, df)
    return None

# trains model
def train_model(model_options, test_size, x_train, y_train, num_inputs, data_stats, exclude_zeros,
                model_accuracy, feature_list, write_model, model_dir):
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, 
                                                        test_size=test_size, random_state=random_state)
    print('\nPrepared training data...')
    
    # chk = check_data_sizes(x_train, y_train, X_train, X_test, Y_train, Y_test)
    chk = True
    
    if exclude_zeros:
        X_train = np.delete(X_train, np.where(Y_train == 0), axis = 0)
        Y_train = np.delete(Y_train, np.where(Y_train == 0), axis=0)
        X_test = np.delete(X_test, np.where(Y_test == 0), axis = 0)
        Y_test = np.delete(Y_test, np.where(Y_test == 0), axis=0)
    
    if data_stats:
        label_stats(Y_test)
        
    model_names = []
    
    if chk:
        for clf in model_options:
            print(f'\n\nTraining: {clf}')
            log_output(f'\n\nTraining: {clf}')
            start_time = time.time() # start time for process timing
            model = clf.fit(X_train, Y_train)
            print(f'--Trained {clf} in {(time.time() - start_time):.1f} seconds / {(time.time() - start_time)/60:.1f} minutes\n')
            log_output(f'--Trained {clf} in {(time.time() - start_time):.1f} seconds / {(time.time() - start_time)/60:.1f} minutes\n')
        
            if model_accuracy:
                if 'RandomForestClassifier' in str(clf) and 'Ada' not in str(clf):
                    assess_accuracy(clf, X_train, Y_train, X_test, Y_test, True, feature_list)
                else:
                    assess_accuracy(clf, X_train, Y_train, X_test, Y_test, False, feature_list)
        
            if write_model and 'RandomForestClassifier' in str(clf) and 'Ada' not in str(clf): # {X_train.shape[1]}B_{num_inputs}In
                model_name = model_dir + f'\RF_{X_train.shape[1]}B_{model.n_estimators}trees_{model.min_samples_leaf}leaf_{model.min_samples_split}split_{model.max_depth}depth_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
                pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
                # print(f'\nSaved model: {model_name}\n')
                model_names.append(model_name)
            elif write_model and 'Ada' in str(clf): # {X_train.shape[1]}B_{num_inputs}In
                model_name = model_dir + f'\Ada_{X_train.shape[1]}B__' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
                pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
                # print(f'\nSaved model: {model_name}\n')
                model_names.append(model_name)
            elif write_model and 'HistGradient' in str(clf):
                model_name = model_dir + f'\Hist_{X_train.shape[1]}B_{num_inputs}In_LR{model.learning_rate}_L2{model.l2_regularization}_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
                pickle.dump(model, open(model_name, 'wb'))
                # print(f'\nSaved model: {model_name}\n')
                model_names.append(model_name)
            elif write_model and 'XGB' in str(clf):
                model_name = model_dir + f'\XGB_{X_train.shape[1]}B_{num_inputs}In_NumEst{model.n_estimators}_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
                pickle.dump(model, open(model_name, 'wb'))
                # print(f'\nSaved model: {model_name}\n')
                model_names.append(model_name)
            elif write_model and 'MLP' in str(clf):
                model_name = model_dir + f'\MLP_{X_train.shape[1]}B_{num_inputs}In_{model.hidden_layer_sizes}Layers_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
                pickle.dump(model, open(model_name, 'wb'))
                # print(f'\nSaved model: {model_name}\n')
                model_names.append(model_name)
            elif write_model and 'GradientBoosting' in str(clf):
                model_name = model_dir + f'\GBC_{X_train.shape[1]}B_{num_inputs}In_NumEst{model.n_estimators}_Depth{model.max_depth}_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
                pickle.dump(model, open(model_name, 'wb'))
                # print(f'\nSaved model: {model_name}\n')
                model_names.append(model_name)                
        return model_names
    else:
        print('Issue with data sizes')
        return None

# performs k-fold cross validation of training data
def compute_kfold(model_options, x_train, y_train, n_splits, stratified):  
    start_time = time.time() # start time for process timing

    for clf in model_options:
        if stratified:
            cv = StratifiedKFold(n_splits=n_splits)
            print(f'\nPerforming stratified {n_splits}-fold cross validation...')
            log_output(f'\n--Performing {n_splits}-fold cross validation...')
        else:
            cv = n_splits
            print(f'\nPerforming {n_splits}-fold cross validation...')
            log_output(f'\n--Performing {n_splits}-fold cross validation...')
        
        scores = cross_val_score(clf, x_train, y_train, cv=cv, scoring='f1_macro') # ***** check this
        print(f'\n--{n_splits}-fold cross validation results:\n--{scores}')
        print("--%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        log_output(f'\n{n_splits}-fold cross validation results:\n--{scores}'
                   f'\n{scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}')
    
    print(f'\n--Completed {n_splits}-fold cross validation in {(time.time() - start_time):.1f} seconds / {(time.time() - start_time)/60:.1f} minutes\n')
    return None

# computes and plots learning curve
def compute_learning_curve(model_options, x_train, y_train, n_splits, stratified):
    
    for clf in model_options:
        start_time = time.time() # start time for process timing

        if stratified:
            cv = StratifiedKFold(n_splits=n_splits)
            print(f'\nUsing stratified {n_splits}-fold cross validation...')
            log_output(f'\n--Performing {n_splits}-fold cross validation...')
        else:
            cv = n_splits
            print(f'\nUsing {n_splits}-fold cross validation...')
            log_output(f'\n--Performing {n_splits}-fold cross validation...')
            
        print(f'\nComputing learning curve for {clf}. Time: {datetime.datetime.now().time()}')
        log_output(f'--Computing learning curve for {clf}. Time: {datetime.datetime.now().time()}')
        
        train_size_abs, train_scores, test_scores = learning_curve(
        clf, x_train, y_train, cv=cv, scoring='f1_macro', 
        train_sizes=np.linspace(0.1, 1., 10), random_state=random_state)
            
        # Calculate training and test mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot the learning curve
        print(f'--Plotting learning curve for {clf}. Time: {datetime.datetime.now().time()}')
        plotLearningCurve(train_size_abs, train_mean, train_std, test_mean, test_std, curve_title=clf)
        print(f'Test accuracy:\n{test_mean}')
        print(f'Test standard deviation:\n{test_std}')
        log_output(f'Test accuracy:{test_mean}\nTest stdev:{test_std}')
        
        print(f'\n--Completed learning curve in {(time.time() - start_time):.1f} seconds / {(time.time() - start_time)/60:.1f} minutes\n')
    return None


# %% -- main

def main(): 
    feature_list = [
# 'Blue',
# 'Green',
# 'Red',
# 'NIR',
# 'OSI',
# 'pSDBg',
# 'pSDBgStandardDeviationSlope',
# 'pSDBgRoughness',
# 'Chl_a',
# 'TSM',
# 'Secchi',
        
'RedEdge704',
'Blue',
'Green',
'Red',
'NIR',
'OSI',
'pSDBgStandardDeviationSlope',
'pSDBgRoughness',
'Chl_a',
'TSM',




    ]

    # inputs 
    training_composites = [
                            # turbid training
'P:\\Thesis\\Training\\_Turbid_Training\\Hatteras_20230102\\_Features_10Bands\\_Composite\\Hatteras_20230102Ex5C_10Bands_composite_20231214_1521.tif', 'P:\\Thesis\\Training\\_Turbid_Training\\Lookout_20230306\\_Features_10Bands\\_Composite\\Lookout_20230306_5C_10Bands_composite_20231214_1521.tif'



]
    
    training_labels = [
                        # turbid training                       
                        r'P:\Thesis\Test Data\_Manuscript_Test\Masks\Lookout_20230306_5C_TF.tif',
                        r'P:\Thesis\Test Data\_Manuscript_Test\Masks\Hatteras_20230102Ex5C_TF.tif'
                        
                      ]
    
    training_list = pair_composite_with_labels(training_composites, training_labels)
    
    # models
    model_options = [
                    # RandomForestClassifier(n_estimators=50, min_samples_leaf=10,  min_samples_split=2,  max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    
                    RandomForestClassifier(n_estimators=100, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=80, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_
                    #                        leaf=160, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=200, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=400, random_state=random_state,n_jobs=n_jobs,oob_score=True),


                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=40, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # MLPClassifier(hidden_layer_sizes=(5,5,5,6,6,6,6), random_state=random_state),
                    # MLPClassifier(hidden_layer_sizes=(7,7,7,6,6,6,6), random_state=random_state),
                    # RandomForestClassifier(n_estimators=20, random_state=random_state,n_jobs=n_jobs,oob_score=True),


                    
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=40, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=60, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=80, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=100, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=160, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=320, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=640, random_state=random_state,n_jobs=n_jobs,oob_score=True),

                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=200, min_samples_leaf=40, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=60, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=200, min_samples_leaf=80, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=100, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=200, min_samples_leaf=160, random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=100, min_samples_leaf=320, random_state=random_state,n_jobs=n_jobs,oob_score=True),

                    # RandomForestClassifier(n_estimators=1000,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
                    # RandomForestClassifier(n_estimators=2000,  random_state=random_state,n_jobs=n_jobs,oob_score=True),

                    # HistGradientBoostingClassifier(random_state=random_state,max_depth=3),
                    # HistGradientBoostingClassifier(learning_rate=0.2, l2_regularization=0.2, random_state=random_state, max_iter=500),
                    
                    # GradientBoostingClassifier(random_state=random_state,max_depth=3,n_estimators=100),
                    # GradientBoostingClassifier(random_state=random_state,max_depth=3,n_estimators=200),
                    # GradientBoostingClassifier(random_state=random_state,max_depth=6,n_estimators=100),
                    # GradientBoostingClassifier(random_state=random_state,max_depth=6,n_estimators=200),
                    # GradientBoostingClassifier(random_state=random_state,max_depth=None,n_estimators=100),
                    # GradientBoostingClassifier(random_state=random_state,max_depth=None,n_estimators=200),
                    
                    # MLPClassifier(hidden_layer_sizes=(4,5,5,6,6,6), random_state=random_state),
                    # MLPClassifier(hidden_layer_sizes=(4,4,5,5,6,6,6), random_state=random_state),
                    # MLPClassifier(hidden_layer_sizes=(4,4,5,5,5,6,6,6), random_state=random_state),
                    # MLPClassifier(hidden_layer_sizes=(4,4,5,5,5,6,6,6,6), random_state=random_state),
                    # MLPClassifier(hidden_layer_sizes=(5,5,5,6,6,6,6), random_state=random_state),
                    # MLPClassifier(hidden_layer_sizes=(5,5,5,5,6,6,6,6,6), random_state=random_state),
                    # MLPClassifier(hidden_layer_sizes=(5,5,5,6,6,6,6,6), random_state=random_state),
                    # MLPClassifier(hidden_layer_sizes=(5,5,5,5,5,6,6,6,6,6), random_state=random_state),
                    
                    # AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=n_jobs),random_state=random_state),

                    # XGBClassifier(
                    #                 learning_rate=0.1,
                    #                 n_estimators=100,
                    #                 max_depth=3,
                    #                 subsample=0.8,
                    #                 colsample_bytree=0.8,
                    #                 objective='binary:logistic'
                    #                 )
                    ]

    
    # output
    model_dir = r"C:\_Turbidity\Models"

    # train model(s) -- either one or multiple with various hyperparameters options
    num_inputs = len(training_list)
    if num_inputs > 1: # multiple training inputs
        x_train, x_metadata, x_bounds = shape_multiple_composites(training_list)
        y_train, y_metadata, y_bounds = shape_multiple_labels(training_list)
        
        # compute_learning_curve(model_options, x_train, y_train, n_splits=5, stratified=True)
        # compute_kfold(model_options, x_train, y_train, n_splits=10, stratified=True)
    
        model_names = train_model(model_options, test_size=0.3, x_train=x_train, y_train=y_train, num_inputs=num_inputs, data_stats=True, exclude_zeros=True,
                    model_accuracy=True, feature_list=feature_list, write_model=True, model_dir=model_dir)
        print(f'\nModels trained: {model_names}')
    elif num_inputs == 1: # single training input
        x_train, x_metadata, x_bounds = shape_single_composite(training_list[0][0])
        y_train, y_metadata, y_bounds = shape_single_label(training_list[0][1])

        model_names = train_model(model_options, test_size=0.3, x_train=x_train, y_train=y_train, num_inputs=num_inputs, data_stats=True, exclude_zeros=True,
                    model_accuracy=True, feature_list=feature_list, write_model=True, model_dir=model_dir)
        print(f'Models trained: {model_names}')
    return None

if __name__ == '__main__':
    total_time = time.time() # start time for process timing

    start = current_time.strftime('%H:%M:%S')
    print(f'Starting at {start}\n')
    main()
    runtime = time.time() - total_time
    print(f'\nTotal elapsed time: {runtime:.1f} seconds / {(runtime/60):.1f} minutes')

        
## %% - Sensitivity analysis

# RandomForestClassifier(n_estimators=50, min_samples_leaf=1,   min_samples_split=2,  max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=10,  min_samples_split=2,  max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=100, min_samples_split=2,  max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=1,   min_samples_split=10, max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=1,   min_samples_split=20, max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=1,   min_samples_split=2,  max_depth=5,   random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=1,   min_samples_split=2,  max_depth=10,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=1,   min_samples_split=2,  max_depth=20,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=100, min_samples_split=2,  max_depth=10,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=100, min_samples_split=2,  max_depth=20,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=100, min_samples_split=20, max_depth=20,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=50, min_samples_leaf=10,  min_samples_split=20, max_depth=20,  random_state=random_state,n_jobs=n_jobs,oob_score=True),

# RandomForestClassifier(n_estimators=100, min_samples_leaf=1,   min_samples_split=2,  max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=10,  min_samples_split=2,  max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=100, min_samples_split=2,  max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=1,   min_samples_split=10, max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=1,   min_samples_split=20, max_depth=None,random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=1,   min_samples_split=2,  max_depth=5,   random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=1,   min_samples_split=2,  max_depth=10,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=1,   min_samples_split=2,  max_depth=20,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=100, min_samples_split=2,  max_depth=10,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=100, min_samples_split=2,  max_depth=20,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=100, min_samples_split=20, max_depth=20,  random_state=random_state,n_jobs=n_jobs,oob_score=True),
# RandomForestClassifier(n_estimators=100, min_samples_leaf=10,  min_samples_split=20, max_depth=20,  random_state=random_state,n_jobs=n_jobs,oob_score=True)

