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

def ConvCharVec2NumbVec(CharVec):
    # this function will take a vector of characters and convert it to a correspon
    # ding int vector. A range of int value will be determined by the length of 
    # unique function value of given character vector.
    
    IntVec = np.zeros(CharVec.shape)
    CharVecUnique = np.unique(CharVec)
    
    # now, create the random vectors to assign values
    FirstRand = np.random.uniform(0,10,1)[0] # mean
    SecondRand = np.random.uniform(0.5,3,1)[0] # increment
    RandVal = np.zeros(CharVecUnique.shape)
    
    for i in range(0,CharVecUnique.shape[0]):
        RandVal[i] = FirstRand + (i*SecondRand)
        
    for i in range(0,CharVec.shape[0]):
        ThisChar = CharVec[i]
        CorrespInd = np.where(CharVecUnique==ThisChar)[0][0]
        IntVec[i] = RandVal[CorrespInd]
              
    return IntVec

def FeatEng_Mushrooms(data):
    # today, I would like to populate a given DataFrame with positive
    # integer values. This particular data frame is popualated with
    # strings. Now, I will have to get the unique string values and 
    # convert it to corresponding integer values
    
    # save the label in somewhere
    Label = data['class']
    data = data.drop('class', axis=1)
    
    for col in data:
        data[col] = ConvCharVec2NumbVec(data[col].values)
        
    data = data.assign(classes=Label.values) # If you use class, it throws the syntax error!
    data.rename(columns={'classes': 'class'}, inplace=True)
    
    return data

# feature enginnering
data = FeatEng_Mushrooms(data)

# split data into train and test

# k-fold cross validation (whithin training data) also, decide how many times I would do this

# randomly select indicies training of training data

# test all possible (or given) KNN parameters set

# choose the best set of parameters 

# plot the result or research the way to select those parameters
