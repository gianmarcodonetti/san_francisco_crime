# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:10:04 2016

@author: Giammi
"""

import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

# SETUP ==========================================================================================
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import normalize
#import xgboost
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

workspace = "C:\\Users\\Giammi\\OneDrive\\Università\\Machine Learning\\project\\data"
os.chdir(workspace)

train = './local_train.csv'
test = './local_test.csv'

test = pd.read_csv(test, delimiter=',')
train = pd.read_csv(train, delimiter=',')

# Add the field of Block
train['Block'] = train.Address.str.contains('.?Block.?')
test['Block'] = test.Address.str.contains('.?Block.?')

# Printing some info on the data
print(train.info())
headTest = test[:5]
headTrain = train[:5]

def showExecTime(startPoint, initialString = ""):
    eex = time.time()
    seconds = round(eex - startPoint, 2)
    minutes = (seconds/60)
    hours = int(minutes/60)
    minutes = int(minutes % 60)
    seconds = round(seconds % 60, 2)
    print("\n- "+initialString+" Execution time: %sh %sm %ss -" % (hours, minutes, seconds))

# PRE-PROCESSING =================================================================================
"""
Convert categorical attributes to number
"""
# For the TRAIN data
# Select the features
crime = train['Crime']
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = pd.get_dummies(train.Hour)
month = pd.get_dummies(train.Month)
year =  pd.get_dummies(train.Year)
block = train.Block.astype(int)
xs = train.X
ys = train.Y
# normalize georefenrences
xs = pd.DataFrame(normalize(xs[:,np.newaxis], axis=0).ravel(), columns=['X'])
ys = pd.DataFrame(normalize(ys[:,np.newaxis], axis=0).ravel(), columns=['Y'])

train_data = pd.concat([hour, days, year, block, district, xs, ys], axis=1)
train_data['Crime'] = crime
#==============================================================================
# # Remove the single outlier!!!
# train_data = train_data[train_data['X'] != max(xs['X'])]
#==============================================================================


# For TEST data
# Select the features
crime = test['Crime']
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
hour = pd.get_dummies(test.Hour)
month = pd.get_dummies(test.Month)
year =  pd.get_dummies(test.Year)
block = test.Block.astype(int)
xs = test.X
ys = test.Y
# normalize georefenrences
xs = pd.DataFrame(normalize(xs[:,np.newaxis], axis=0).ravel(), columns=['X'])
ys = pd.DataFrame(normalize(ys[:,np.newaxis], axis=0).ravel(), columns=['Y'])

test_data = pd.concat([hour, days, year, block, district, xs, ys], axis=1)
test_data['Crime'] = crime

del days, district, hour, month, year, block, xs, ys, crime


# MODELS ======================================================================================
"""
Choose the model you want, fit it with the data,
then you can perform some bagging or
some boosting on the base model you have. 
"""
# Build up the features
features = list(train_data.columns[:-10])
train = train_data[:] # useful in order to reduce the fitting time
test = test_data[:]

# Base-Model construction
begin = time.time()
baseModel = naive_bayes.BernoulliNB()
baseModel.fit(train[features], train['Crime'])
showExecTime(begin, "BernoulliNB fitted.")

# SVC
begin = time.time()
svc = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, class_weight='auto'))
svc.fit(train[features], train['Crime'])
showExecTime(begin, "SVC fitted.")

# Logistic Regression
begin = time.time()
C = [1] # use this for cross validation
lr = [] # list for the models
for i in range(len(C)):
    lr.append(linear_model.LogisticRegression(C=C[i], solver='lbfgs'))
    lr[i].fit(train[features], train['Crime'])
showExecTime(begin, "LogisticRegression fitted.")

# Logistic Regression with CV
begin = time.time()
lrcv = linear_model.LogisticRegressionCV(Cs=10, solver='lbfgs', n_jobs=-1)
lrcv.fit(train[features], train['Crime'])
showExecTime(begin, "Cross-Validated LogisticRegression fitted.")

# Random Forest
begin = time.time()
randomForest = ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1, min_samples_leaf=50)
randomForest.fit(train[features], train['Crime'])
showExecTime(begin, "RandomForestClassifier fitted.")

# AdaBoost
begin = time.time()
adaBoost = ensemble.AdaBoostClassifier(base_estimator=baseModel, n_estimators=10)
adaBoost.fit(train[features], train['Crime'])
showExecTime(begin, "AdaBoostClassifier fitted.")

# Bagging
begin = time.time()
bagging = ensemble.BaggingClassifier(baseModel, n_jobs=-1, n_estimators=10)
bagging.fit(train[features], train['Crime'])
showExecTime(begin, "BaggingClassifier fitted.")


# Choose the model ===============================================================================
model = baseModel # or any other different model

# Prediction
predicted_train = np.array(model.predict_proba(train[features]))
predicted_test = np.array(model.predict_proba(test[features]))

len_test, len_train = len(test['Crime'].value_counts()), len(train['Crime'].value_counts())

if len_test != len_train:
    k_test = set(test['Crime'].value_counts().keys())
    k_train = set(train['Crime'].value_counts().keys())
    k_missing = list(k_train.difference(k_test))
    for i in range(len(k_missing)):
        index = k_missing[i]
        predicted_test = np.hstack((predicted_test[:,:index],predicted_test[:,index+1:]))
        for j in range(i, len(k_missing)):
            if k_missing[j] > k_missing[i]:
                k_missing[j] -= 1

print ("\n\tLog_loss on the train :", log_loss(train['Crime'], predicted_train))
print ("\tLog_loss on the test :", log_loss(test['Crime'], predicted_test))

truncation = True
if truncation:
    tot_len = len(predicted_test)
    for i in range(tot_len):
        sys.stdout.write("\r\t"+str( round(i/tot_len*100, 2) )+" %")
        predicted_test[i] = np.around(predicted_test[i], decimals=5)

    print("\tTRUNCATION")

    print ("\tLog_loss on the test :", log_loss(test['Crime'], predicted_test))


# BASELINE =======================================================================================
# Uniform
for i in range(len(predicted_test)):
    for j in range(len(predicted_test[i])):
        predicted_test[i][j] = 1/len(predicted_test[i])
        
for i in range(len(predicted_train)):
    for j in range(len(predicted_train[i])):
        predicted_train[i][j] = 1/len(predicted_train[i])

print ("\n\tLog_loss on the train :", log_loss(train['Crime'], predicted_train))
print ("\tLog_loss on the test :", log_loss(test['Crime'], predicted_test))

# Random
(r, c) = len(predicted_train), len(predicted_train[0])
predicted_train = np.random.random((r, c))
for i in range(len(predicted_train)):
    for j in range(len(predicted_train[i])):
        predicted_train[i][j] /= np.sum(predicted_train[i])

(r, c) = len(predicted_test), len(predicted_test[0])
predicted_test = np.random.random((r, c))
for i in range(len(predicted_test)):
    for j in range(len(predicted_test[i])):
        predicted_test[i][j] /= np.sum(predicted_test[i])
        
print ("\n\tLog_loss on the train :", log_loss(train['Crime'], predicted_train))
print ("\tLog_loss on the test :", log_loss(test['Crime'], predicted_test))

