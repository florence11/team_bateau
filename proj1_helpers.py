# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
#import matplotlib.pyplot as plt
#import datetime

def load_csv_data(data_path, sub_sample=False, add_outlier = False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids) and remove aberant values"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]   

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]
    
    #final pre-processing of data
    # data = input_data
    data= remove_single_data(input_data)
    # data_int = remove_useless_col(input_data)
    # data = remove_single_data(data_int)
    # data = remove_outlier(data, 1)
    data = remove_outlier(data, 2)
    # data = remove_outlier(data, 3)
    
    return yb, data, ids

def remove_useless_col(data):
    """Remove a col of the data matrix if the proportion of -999 is above 10 or 20%"""
    ind_to_del = []
    nb_col = data.shape[1]
    data=data.T

    for k in range(nb_col):
        col = np.array(data[k])
        number = col.tolist().count(-999)
        proportion = number / len(col)
        if proportion > 0.2 :
            ind_to_del.append(k)
            
    data = np.delete(data, ind_to_del, 0)
    data=data.T
    return data


def remove_single_data(data):
    """Replace aberrant of the data matrix if it is -999"""
    nb_col = data.shape[1]
    data=data.T
    ind_to_modify = []
    mean_of_modify = []
    for k in range(nb_col):
        col = np.array(data[k])
        number = col.tolist().count(-999)

        if number > 0 :
            sum_col = np.sum(col) - number * (-999)
            len_col = len(col) - number
            col_mean = sum_col/len_col

            data[k , np.where(col==-999)] = col_mean
    data=data.T
    
    return data

def remove_outlier(data,level):
    data_without_outlier = data
    for c in range(0, data.shape[1]-1):
        mean = np.mean(data[:,c])
        std = np.std(data[:, c])

        max = mean + level*std
        min = mean - level*std
        
        for i in range(0, data.shape[0]-1):
            if data_without_outlier[i, c]>max or data_without_outlier[i, c] < min :
                data_without_outlier[i, c] = mean
    return data_without_outlier        



def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 0.5*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))



def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def compare_result(y_final, y_ini):
    """Compare the final and initial result vector and return the proportion of good result"""
    l = len(y_ini)
    if l != len(y_final): raise NotGoodSizeError
    ratio = np.zeros(l)
    for k in range(l):
        if y_ini[k] == y_final[k]:
            ratio[k] = 1  
            
    return np.sum(ratio) / l


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

