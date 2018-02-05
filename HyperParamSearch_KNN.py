#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:54:14 2018

@author: Alex
"""
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

train = pd.read_csv('../Titanic/train.csv')
test = pd.read_csv('../Titanic/test.csv')

# get title information
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

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
    
def FeatEng(train):
    # title might be important
    train['Title'] = train['Name'].apply(get_title)
    train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    train['Title'] = train['Title'].replace('Mlle', 'Miss')
    train['Title'] = train['Title'].replace('Ms', 'Miss')
    train['Title'] = train['Title'].replace('Mme', 'Mrs')

    # Convert Title value to random integers
    TempCharArr = train['Title'].values
    TempIntArr = ConvCharVec2NumbVec(TempCharArr)
    train = train.drop('Title', axis=1)
    train['Title'] = TempIntArr

    # sex of each person is definitely important
    train['Sex'] = train['Sex'].map({'female':0, 'male':1}).astype(int)

    # drop unimportant features
    train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    # fill in the age (null) list
    age_avg = train['Age'].mean()
    age_std = train['Age'].std()
    age_null_count = train['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    train['Age'][np.isnan(train['Age'])] = age_null_random_list
    train['Age'] = train['Age'].astype(int)

    # replace NaN value in the Cabin
    # train['Cabin'][train['Cabin'].isnull()==True] = 'Z' # 'Z' for 'NaN'
    train.Cabin.fillna('Z', inplace=True) # 'Z' for 'NaN'

    # this is inefficient
    #for i, row in train.iterrows():
    #    if len(row['Cabin']) > 1:
    #        train['Cabin'][i] = row['Cabin'][0] # cabin category

    # this is efficient way
    train.loc[:,"Cabin"] = train.Cabin.apply(lambda x: x[0])

    # Convert cabin value to random integers
    TempCharArr = train['Cabin'].values
    TempIntArr = ConvCharVec2NumbVec(TempCharArr)
    train = train.drop('Cabin', axis=1)
    train['Cabin'] = TempIntArr

    # replace NaN value in Embarked
    train.Embarked.fillna('P', inplace=True) # 'P' for 'NaN'

    # Convert Embarked value to random integers
    TempCharArr = train['Embarked'].values
    TempIntArr = ConvCharVec2NumbVec(TempCharArr)
    train = train.drop('Embarked', axis=1)
    train['Embarked'] = TempIntArr
    
    # replace NaN value in Fare
    train.Fare.fillna(np.mean(np.asanyarray(train['Fare'][:20])), inplace=True)
    HowMany = np.where(np.isnan(np.asanyarray(train['Fare'])))[0].shape[0] # number of NaN
    

    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    
    # Create new feature IsAlone from FamilySize
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
    
    return train

# train: original training data
# TempTrain: temporary training data for manipulation
# Train: final training dataset
TempTrain = train.copy() # make a copy of it
TempTrain = TempTrain.drop(['Survived'], axis=1) # take survive column out
Data = pd.concat([TempTrain,test]) # concatanation
Data = FeatEng(Data) # feature engineering
TempTrain = Data.iloc[:TempTrain.shape[0],:]
Train = TempTrain.copy()
Train['Survived'] = train['Survived']

# test: original test data
# Test: final test dataset
Test = Data.iloc[TempTrain.shape[0]:,:]

X = Train.iloc[:,0:Data.shape[1]] # data
y = Train.iloc[:,Data.shape[1]] # label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
KNNNumbSamples = 5 # get 5 best mean_test_score
KNN_N_iter_search = KNNNumbSamples*4
KNN_N_splits = 5
KNNparam_dist = {"n_neighbors": [1, 3, 5, 7, 9],
                  "weights": ["uniform", "distance"],
                  "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                  "p": [1, 2]}
KNNss = ShuffleSplit(n_splits=KNN_N_splits, test_size=0.2, random_state=0)

def KNNRandomParamSearch(X_train, y_train, N_iter_search, NumbSamples,
                         param_dist, ss, FigFlag):
    # specify parameters and distributions to sample from
    clf = KNeighborsClassifier()
    n_iter_search = N_iter_search
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search,scoring='roc_auc',
                                       cv=ss)
    
    NotDone = 1
    while NotDone:
        random_search.fit(X_train, y_train)
        
        MeanTestScore = random_search.cv_results_['mean_test_score']
        MeanTrainScore = random_search.cv_results_['mean_train_score']
        MeanTestScoreSortedDesInd = (-MeanTestScore).argsort()[:MeanTestScore.shape[0]]
        ResultInd = []
        for i in range(0,MeanTestScore.shape[0]):
            if MeanTrainScore[MeanTestScoreSortedDesInd[i]] < 0.97:
                ResultInd.append(MeanTestScoreSortedDesInd[i])
        ResultInd = np.array(ResultInd)[:NumbSamples]
        if ResultInd.shape[0]>=NumbSamples:
            NotDone = 0
            
    CVResult = random_search.cv_results_
    ResultDict = []
    for i in range(0,NumbSamples):
        ResultDict.append(CVResult['params'][ResultInd[i]])
    
    if FigFlag:
        fig, ax = plt.subplots()
        bar_width = 0.35
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        
        plt.bar(np.arange(1,(ss.n_splits)+1), MeanTestScore[ResultInd], bar_width, alpha=opacity, color='b', error_kw=error_config, label='Mean test score')
        plt.bar(np.arange(1,(ss.n_splits)+1)+bar_width, MeanTrainScore[ResultInd], bar_width, alpha=opacity, color='r', error_kw=error_config, label='Mean train score')
        plt.xlabel('Splits')
        plt.ylabel('Scores')
        plt.title('KNN scores by splits')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return ResultDict

KNNParamDict = KNNRandomParamSearch(X_train, y_train, KNN_N_iter_search, 
                                    KNNNumbSamples, KNNparam_dist, KNNss, 
                                    1)    


