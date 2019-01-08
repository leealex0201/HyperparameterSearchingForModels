#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:11:04 2018

@author: Alex
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from polylearn import FactorizationMachineClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
train = pd.read_csv(cwd + '/Titanic/train.csv')
test = pd.read_csv(cwd + '/Titanic/test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId']

# get title information
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def RemoveCorrFeat(data,CorrThresh,DesiredDataSize):
    # Removing highly correlated features. Features pair with pearson correlation
    # value greater than CorrThresh will be removed from the data
    NotDone = 1
    while NotDone:
            
        IndArray = data.astype(float).corr().index.values
        CorrMat = data.astype(float).corr().values
        CorrIndArray = []

        for i in range(0,CorrMat.shape[0]-1):
            for j in range(i+1,CorrMat.shape[0]-1):
                if np.absolute(CorrMat[j,i]) > CorrThresh:
                    CorrIndArray.append([IndArray[j],IndArray[i]])
    
        CorrIndArray = np.array(CorrIndArray)
        if (CorrIndArray.size == 0) or data.shape[1] == DesiredDataSize:
            # no pair of feature that has correlation threshold
            NotDone = 0
        else:
            unique, counts = np.unique(CorrIndArray, return_counts=True)
            i = np.unravel_index(counts.argmax(), counts.shape)
            data.drop([unique[i[0]]], axis = 1, inplace = True)
        
    return data

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
    train.Cabin.fillna('Z', inplace=True) # 'Z' for 'NaN'

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

    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    
    # Create new feature IsAlone from FamilySize
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
    
    return train

def RandomParamSearch(clf, X_train, y_train, N_iter_search, NumbSamples,
                         param_dist, ss, FigFlag, ModelName):
    # specify parameters and distributions to sample from
    n_iter_search = N_iter_search
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search,scoring='roc_auc',
                                       cv=ss)
    if ModelName == 'RF': # to avoid overfitting
        Thresh = 0.99 # Random forest overfitting issue?
    else:
        Thresh = 0.95 
        
    NotDone = 1
    ResultInd = []
    Counter = 0
    while NotDone:
        random_search.fit(X_train, y_train)
        Counter = Counter + 1
        MeanTestScore = random_search.cv_results_['mean_test_score']
        MeanTrainScore = random_search.cv_results_['mean_train_score']
        MeanTestScoreSortedDesInd = (-MeanTestScore).argsort()[:MeanTestScore.shape[0]]

        for i in range(0,MeanTestScore.shape[0]):
            if MeanTrainScore[MeanTestScoreSortedDesInd[i]] < Thresh:
                ResultInd.append(MeanTestScoreSortedDesInd[i])
        
        if len(ResultInd)>=NumbSamples:
            NotDone = 0
            
        if Counter > 10:
            Thresh = Thresh + 0.01 # as stage progressed, overfitting occurrs frequently.
            Counter = 1
    
    ResultInd = np.array(ResultInd)[:NumbSamples]     
    CVResult = random_search.cv_results_
    ResultParamList = []
    for i in range(0,NumbSamples):
        ResultParamList.append(CVResult['params'][ResultInd[i]])
    
    if FigFlag:
        fig, ax = plt.subplots()
        bar_width = 0.35
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        
        plt.bar(np.arange(1,NumbSamples+1), MeanTestScore[ResultInd], bar_width, alpha=opacity, color='b', error_kw=error_config, label='Mean test score')
        plt.bar(np.arange(1,NumbSamples+1)+bar_width, MeanTrainScore[ResultInd], bar_width, alpha=opacity, color='r', error_kw=error_config, label='Mean train score')
        plt.xlabel('Splits')
        plt.ylabel('Scores')
        plt.title(ModelName + ' scores by splits')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return ResultParamList

class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)

def CreateNNHiddenLayer(MIN, MAX, LOW, HIGH, NL):
    # this code randomly creates tuples
    #
    # MIN: minimum number of tuple indices
    # MAX: maximum number of tuple indices
    # LOW: low-end of range of number
    # HIGH: high-end of range of number
    # NL: number of layers
    import random
    
    HL = []
    for i in range(NL):
        current = []
        # add a random number of indices to the tuple
        for j in range(random.randint(MIN, MAX)):
            # add another number to the current list
            current.append(random.randint(LOW, HIGH))
        # convert current list into a tuple and add to resulting list
        HL.append(tuple(current))
        
    return HL

def ModelParamSearch(Param, Model, NumbModel, N_splits, X_train, y_train):
    N_iter_search = np.maximum(10, NumbModel*4)
    ss = ShuffleSplit(n_splits=N_splits, test_size=0.2, random_state=0)
    if Model == 'XGB':
        clf = XGBoostClassifier(eval_metric = 'auc', num_class = 2, 
                                nthread = 4, silent = 1)
    elif Model == 'NN':
        clf = MLPClassifier(max_iter=500)
    elif Model == 'FM':
        clf = FactorizationMachineClassifier(max_iter=500)
    elif Model == 'LR':
        clf = LogisticRegression(max_iter=500)
    elif Model == 'KNN':
        clf = KNeighborsClassifier()
    elif Model == 'RF':
        clf = RandomForestClassifier(oob_score=True, bootstrap=True)
        
    ParamDict = RandomParamSearch(clf, X_train, y_train, N_iter_search, 
                                    NumbModel, Param, ss, 0, Model)
    
    return ParamDict, clf

def ModelStacking(Params, Models, NumbModels, X_train, y_train, X):
    N_splits = 5 # number of splits for shuffle splitting
    NewX = pd.Series().reindex_like(y_train) # create temp pandas Series 
    ResultParams = [None]*len(Models)
    for i in range(len(NumbModels)):
        print('## Searching parameter set for model: {} ... '.format(Models[i]))
        ResultParams[i], ThisClf = ModelParamSearch(Params[i], Models[i], NumbModels[i], 
                                     N_splits, X_train, y_train)
        print('## Creating classifiers and predicting results of model: {} ...'.format(Models[i]))
        for ii in range(len(ResultParams[i])):
            # create classifiers and fit it
            ThisClf.set_params(**ResultParams[i][ii]).fit(X_train, y_train)
            ThisPred = ThisClf.predict(X_train) # prediction of this model
            ThisTempPredSeries = pd.Series(ThisPred, index=y_train.index)
            NewX = pd.concat([NewX, ThisTempPredSeries], axis=1)
            
    ColumnNumbers = [x for x in range(NewX.shape[1])]
    ColumnNumbers.remove(0)
    NewX = NewX.iloc[:, ColumnNumbers]
    
    ColNames = []
    for i in range(len(NumbModels)):
        for ii in range(NumbModels[i]):
            ColNames.append(Models[i]+str(ii+1))
    NewX.columns = ColNames # rename columns
    
    # Let's remove some features that show high correlation
    NewX = RemoveCorrFeat(NewX,0.9,np.around(NewX.shape[1]*.8).astype(int))
    
    NewNewX = pd.Series()
    for i in range(NewX.columns.shape[0]):
        ThisModelName = NewX.columns[i]
        ThisModelNameStr = ThisModelName[0:len(ThisModelName)-len(filter(str.isdigit,ThisModelName))]
        ThisModelNumber = int(filter(str.isdigit,ThisModelName))
        WhichModel = Models.index(ThisModelNameStr)
        if ThisModelNameStr == 'XGB':
            ThisClf = XGBoostClassifier(eval_metric = 'auc', num_class = 2, 
                                    nthread = 4, silent = 1)
        elif ThisModelNameStr == 'NN':
            ThisClf = MLPClassifier(max_iter=500)
        elif ThisModelNameStr == 'FM':
            ThisClf = FactorizationMachineClassifier(max_iter=500)
        elif ThisModelNameStr == 'LR':
            ThisClf = LogisticRegression(max_iter=500)
        elif ThisModelNameStr == 'KNN':
            ThisClf = KNeighborsClassifier()
        elif ThisModelNameStr == 'RF':
            ThisClf = RandomForestClassifier(oob_score=True, bootstrap=True)
        
        ThisClf.set_params(**ResultParams[WhichModel][ThisModelNumber-1]).fit(X_train, y_train)
        ThisPred = ThisClf.predict(X) # prediction of this model
        ThisTempPredSeries = pd.Series(ThisPred)
        NewNewX = pd.concat([NewNewX, ThisTempPredSeries], axis=1)
    
    ColumnNumbers = [x for x in range(NewNewX.shape[1])]
    ColumnNumbers.remove(0)
    NewNewX = NewNewX.iloc[:, ColumnNumbers]
    
    NewNewX.columns = NewX.columns # rename columns
    
    return NewX, NewNewX
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)

# XGB parameters
XGBparam = {"eta": [0.01, 0.05, 0.1, 0.3, 0.5],
                  "gamma": [0, 1, 10, 50, 100],
                  "max_depth": [1, 4, 6, 9, 12, 20],
                  "min_child_weight": [1, 5, 10, 20],
                  "max_delta_step": [0, 1, 3, 5, 7, 9],
                  "subsample": [0.9, 1.0],
                  "colsample_bytree": [0.9, 1.0],
                  "lambda": [1, 10, 100, 1000],
                  "alpha": [0, 1, 10, 100, 1000],
                  "tree-method": ["auto", "exact", "approx", "hist"],
                  "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                  "p": [1, 2]}

# NN parameters
NNHiddenLayer = CreateNNHiddenLayer(1, 3, 1, 300, 10)
NNparam = {"hidden_layer_sizes": NNHiddenLayer,
                  "activation": ["relu", "identity", "logistic", "tanh"],
                  "solver": ["lbfgs", "sgd", "adam"],
                  "alpha": [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4],
                  "learning_rate": ["invscaling","adaptive","constant"],
                  "momentum": [0.3, 0.5, 0.9],
                  "beta_1": [0.1, 0.3, 0.5, 0.9],
                  "beta_2": [0.1, 0.3, 0.5, 0.9]}

# FM parameters
FMparam = {"degree": [2, 3],
                  "loss": ["logistic", "squared_hinge", "squared"],
                  "alpha": [1e-3, 1e-2, 0.1, 1, 10, 1e2],
                  "beta": [1e-3, 1e-2, 0.1, 1, 10, 1e2],         
                  "fit_lower": ["explicit", "augment", "None"],
                  "fit_linear": ["True", "False"],
                  "init_lambdas": ["ones", "random_signs"]}

# LR parameters
LRparam = {"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                  "C": [1e-3, 1e-2, 0.1, 1, 10, 1e2],
                  "fit_intercept": [True, False]}

# KNN parameters
KNNparam = {"n_neighbors": [1, 3, 5, 7, 9],
                  "weights": ["uniform", "distance"],
                  "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                  "p": [1, 2]}

# RF parameters
RFparam = {"n_estimators": [4, 6, 8, 10, 12, 14, 16], "criterion": ["gini", "entropy"]}

## 1. First stage - single models
# list of models
Models = ['XGB', 'NN', 'FM', 'LR', 'KNN', 'RF']

# number of models 
NumbModels = [10, 2, 2, 2, 2, 2] # total 35 NN: 10 score 0.58

# list of parameters
Params = [XGBparam, NNparam, FMparam, LRparam, KNNparam, RFparam]

print('# First stage - single models: {}.'.format(Models))
print('# Number of models to use: {}.'.format(NumbModels))                                                      

FirstX, FirstXX = ModelStacking(Params, Models, NumbModels, X, y, Test)

## 2. Second stage
Models = ['XGB']
NumbModels = [5] # [5 1 4] without FM is the best 0.85
Params = [XGBparam]
#Params = [XGBparam, NNparam, FMparam, LRparam]

print('# Second stage : {}.'.format(Models))
print('# Number of models to use: {}.'.format(NumbModels))    

SecondX, SecondXX = ModelStacking(Params, Models, NumbModels, FirstX, y, FirstXX)

## 3. Final stage
Models = ['XGB']
NumbModels = [1]
Params = [XGBparam]

print('# Final stage : {}.'.format(Models))
print('# Number of models to use: {}.'.format(NumbModels))    

FinalX, FinalXX = ModelStacking(Params, Models, NumbModels, SecondX, y, SecondXX)

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': FinalXX['XGB1'].values })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)
