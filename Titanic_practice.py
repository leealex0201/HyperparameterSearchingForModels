import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

cwd = os.getcwd()
train = pd.read_csv(cwd + '/Titanic/train.csv')
test = pd.read_csv(cwd + '/Titanic/test.csv')

PassengerId = test['PassengerId']

temp_train = train.copy()
temp_train = temp_train.drop(['Survived'], axis=1)

Data = pd.concat([temp_train, test])

# Let's do title first
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

Data['Title'] = Data['Name'].apply(get_title)
'''
title_hist = Data['Title'].hist
u_title = Data['Title'].unique()
u_title_count = np.zeros(u_title.shape) # initialization
o_title = Data['Title'].values
for c, i in enumerate(u_title):
    this_title = i
    this_count = 0
    for ii in o_title:
        if this_title == ii:
            this_count += 1
    u_title_count[c] = this_count

u_title_count = u_title_count.astype(int)

plt.figure(figsize=(12,5))
plt.bar(np.arange(u_title_count.size),u_title_count)
plt.xticks(np.arange(u_title_count.size),u_title)
'''
Data['Title'] = Data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
Data['Title'] = Data['Title'].replace('Mlle', 'Miss')
Data['Title'] = Data['Title'].replace('Ms', 'Miss')
Data['Title'] = Data['Title'].replace('Mme', 'Mrs')

def conv_uniq_str_2_double(str_array):
    int_vec = np.zeros(str_array.shape)
    str_vec_unique = np.unique(str_array)
    
    first_rand = np.random.uniform(0, 10) # mean
    second_rand = np.random.uniform(0.5, 3)
    rand_val = np.zeros(str_vec_unique.shape)
    
    for i in range(str_vec_unique.shape[0]):
        rand_val[i] = first_rand + (i*second_rand)
        
    for i in range(str_array.shape[0]):
        this_char = str_array[i]
        corresponding_ind = np.where(str_vec_unique==this_char)[0][0]
        int_vec[i] = rand_val[corresponding_ind]
    
    return int_vec

temp_title = Data['Title'].values
temp_title_int = conv_uniq_str_2_double(temp_title)
Data = Data.drop(['Title'], axis=1)
Data['Title'] = temp_title_int

# WANTED TO LOOK AT NULL RATE
null_rate = []
for i in list(Data):
    this_null_count = Data[i].isnull().sum()
    null_rate.append(100*(this_null_count/Data.shape[0]))

Data['Sex'] = Data['Sex'].map({'female': 0, 'male': 1}).astype(int)

Data = Data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

age_avg = Data['Age'].mean()
age_std = Data['Age'].std()
age_null_count = Data['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
Data['Age'][np.isnan(Data['Age'])] = age_null_random_list
Data['Age'] = Data['Age'].astype(int)


Data = Data.drop('Cabin', axis=1)

Data['Embarked'].fillna('P', inplace=True)
temp_title = Data['Embarked'].values
temp_title_int = conv_uniq_str_2_double(temp_title)
Data = Data.drop(['Embarked'], axis=1)
Data['Embarked'] = temp_title_int

Data['FamilySize'] = Data['SibSp'] + Data['Parch'] + 1
Data['IsAlone'] = 0
Data.loc[Data['FamilySize'] == 1, 'IsAlone'] = 1

Data['Fare'].fillna(Data['Fare'].mean(), inplace=True)

# End of feature engineering.
temp_train = Data.iloc[:temp_train.shape[0],:]
Train = temp_train.copy()
Train['Survived'] = train['Survived']
Test = Data.iloc[temp_train.shape[0]:,:]

X = Train.iloc[:,0:Data.shape[1]] # data
y = Train.iloc[:,Data.shape[1]] # label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# now let's tray LogisticRegression first
from sklearn.linear_model import LogisticRegression
LogisticRegression_clf = LogisticRegression(solver='lbfgs', max_iter=500).fit(X_train, y_train)
LogisticRegression_y_test = LogisticRegression_clf.predict(X_test)

from sklearn.metrics import roc_auc_score
print('ROC score for the logistic regression model (default set) is {}.'.format(roc_auc_score(LogisticRegression_y_test, y_test)))


# Now, let's try the randomized hyperparameter search
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit

LRparam = {"solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                  "C": [1e-3, 1e-2, 0.1, 1, 10, 1e2],
                  "fit_intercept": [True, False]}
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
LogisticRegression_RS_clf = LogisticRegression(max_iter=2000)
random_search = RandomizedSearchCV(LogisticRegression_RS_clf, param_distributions=LRparam,
                                       n_iter=10,scoring='roc_auc',
                                       cv=ss)
random_search.fit(X_train, y_train)

MeanTestScore = random_search.cv_results_['mean_test_score']
MeanTrainScore = random_search.cv_results_['mean_train_score']
MeanTestScoreSortedDesInd = (-MeanTestScore).argsort()[:MeanTestScore.shape[0]] # sorted index of cv in terms of testing score

ResultInd = []

for i in range(MeanTestScore.shape[0]):
    if MeanTrainScore[MeanTestScoreSortedDesInd[i]] < 0.95:
        ResultInd.append(MeanTestScoreSortedDesInd[i])
        
CVResult = random_search.cv_results_
ResultParamList = []
for i in range(9):
    ResultParamList.append(CVResult['params'][ResultInd[i]])
    
scores = []
for i in range(len(ResultParamList)):
    this_temp_clf = LogisticRegression(**ResultParamList[i]).fit(X_train, y_train)
    this_temp_y_test = this_temp_clf.predict(X_test)
    scores.append(roc_auc_score(this_temp_y_test, y_test))
    
LogisticRegression_best_paramset_clf = LogisticRegression(**ResultParamList[np.argmax(scores)]).fit(X_train, y_train)
LogisticRegression_best_paramset_y_test = LogisticRegression_best_paramset_clf.predict(X_test)
print('ROC score for the logistic regression model (best param set) is {}.'.format(roc_auc_score(LogisticRegression_best_paramset_y_test, y_test)))
