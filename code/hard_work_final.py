# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:09:00 2016

@author: Giammi
"""

# SETUP ==========================================================================================
import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import linear_model
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

workspace = "C:\\Users\\Giammi\\OneDrive\\Università\\Machine Learning\\project\\data"
os.chdir(workspace)

train = './my_train.csv'
test = './my_test.csv'

test = pd.read_csv(test, delimiter=',')
train = pd.read_csv(train, delimiter=',')

# Printing some info on the data
print(train.info())
headTest = test[:5]
head = train[:5]
print(head)

# Add the field of Block
train['Block'] = train.Address.str.contains('.?Block.?')
test['Block'] = test.Address.str.contains('.?Block.?')

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
# For the train data
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

# For test data
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


del days, district, hour, month, year, block, xs, ys, crime


# MODELS =======================================================================================
"""
Choose the model you want, fit it with the data,
then you can perform some bagging or
some boosting on the base model you have. 
"""
# Build up the features
features = list(train_data.columns[:-1])

# Base-Model construction
begin = time.time()
baseModel = naive_bayes.BernoulliNB()
baseModel.fit(train_data[features], train_data['Crime'])
showExecTime(begin, "BernoulliNB fitted.")

# Logistic Regression
begin = time.time()
lr = linear_model.LogisticRegression(solver='lbfgs')
lr.fit(train_data[features], train_data['Crime'])
showExecTime(begin, "LogisticRegression fitted.")

# Random Forest
begin = time.time()
randomForest = ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1, min_samples_leaf=50)
randomForest.fit(train_data[features], train_data['Crime'])
showExecTime(begin, "RandomForestClassifier fitted.")

# AdaBoost
begin = time.time()
adaBoost = ensemble.AdaBoostClassifier(base_estimator=baseModel, n_estimators=10)
adaBoost.fit(train_data[features], train_data['Crime'])
showExecTime(begin, "AdaBoostClassifier fitted.")

# Bagging
begin = time.time()
bagging = ensemble.BaggingClassifier(baseModel, 15)
bagging.fit(train_data[features], train_data['Crime'])
showExecTime(begin, "BaggingClassifier fitted.")


# Choose the model ===============================================================================
model = lr # or baseModel, choose the model you want

# Prediction
predicted_train = np.array(model.predict_proba(train_data[features]))
predicted_test = np.array(model.predict_proba(test_data[features]))

print ("\n\tLog_loss on the train :", log_loss(train_data['Crime'], predicted_train))
del predicted_train


# SUBMISSION =====================================================================================
truncation = True
if truncation:
    tot_len = len(predicted_test)
    for i in range(tot_len):
        sys.stdout.write("\r\t"+str( round(i/tot_len*100, 2) )+" %")
        predicted_test[i] = np.around(predicted_test[i], decimals=5)

submission = pd.concat([test['Id'], pd.DataFrame(predicted_test)], axis=1)
categories = pd.get_dummies(train.Category).columns.tolist()
submission.columns = ['Id'] + categories

workspace = "C:\\Users\\Giammi\\OneDrive\\Università\\Machine Learning\\project\\submissions"
os.chdir(workspace)

del train_data, test_data, predicted_test
submission.to_csv("naive_bayes_8.csv", header=True, index=False)
