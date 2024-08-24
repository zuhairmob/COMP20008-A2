import os
import sys
import json
import pandas as pd
import numpy as np
import re 
import nltk
from numpy import nan
from enum import Enum
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
def main():
    # load dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
    dataframe = pd.read_csv(url, header=None, na_values='?')
    
    # # summarize the first few rows
    # print(dataframe.head())
    
    # # summarize the number of rows with missing values for each column
    # for i in range(dataframe.shape[1]):
    #     # count number of rows with missing values
    #     n_miss = dataframe[[i]].isnull().sum()
    #     perc = n_miss / dataframe.shape[0] * 100
    #     print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

    # split into input and output elements
    data = dataframe.values
    ix = [i for i in range(data.shape[1]) if i != 23]
    X, y = data[:, ix], data[:, 23]
    # # print total missing
    # print('Missing: %d' % sum(np.isnan(X).flatten()))
    # # define imputer
    # imputer = KNNImputer()
    # # fit on the dataset
    # imputer.fit(X)
    # # transform the dataset
    # Xtrans = imputer.transform(X)
    # # print total missing
    # print('Missing: %d' % sum(np.isnan(Xtrans).flatten()))
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=21)), ('m', RandomForestClassifier())])
    # fit the model
    pipeline.fit(X, y)
    # define new data
    row = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00, 8.40, nan, nan, 2, 11300, 00000, 00000, 2]
    # make a prediction
    yhat = pipeline.predict([row])
    # summarize prediction
    print('Predicted Class: %d' % yhat)
if __name__ == "__main__":
    main()