#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:13:27 2018

@author: Alex
"""

"""
This script is for searching hyper parameters for KNN model (sci-kit learn)
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('mushrooms.csv')

# feature enginnering

# split data into train and test

# k-fold cross validation (whithin training data) also, decide how many times I would do this

# randomly select indicies training of training data

# test all possible (or given) KNN parameters set

# choose the best set of parameters 

# plot the result or research the way to select those parameters