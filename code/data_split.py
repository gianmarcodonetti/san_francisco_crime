# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:51:00 2016

@author: Giammi
"""

# SETUP ==========================================================================================
import os
import sys
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

# Directory where the data are stored
workspace = 'C:\\Users\\Giammi\\OneDrive\\Università\\1 anno\\2 semestre\\'
workspace += 'Data Mining & Text Mining\\kaggle competitions\\san francisco crime'
os.chdir(workspace)

train = './train.csv'
test = './test.csv'

test = pd.read_csv(test, parse_dates=['Dates'], delimiter=',')
train = pd.read_csv(train, parse_dates=['Dates'], delimiter=',')

# Printing some info on the data
print(train.info())
print(train[:10])


# FUNCTIONS ======================================================================================
def split_dataframe(data, ratio=0.5, speed = 1):
    my_train = pd.DataFrame(columns=data.columns)
    my_test = pd.DataFrame(columns=data.columns)
    tot = len(data)
    i = 0
    while i in range(tot):
        sys.stdout.write("\r\t"+str( round( i/tot*100, 1) )+" %")
        random_n = random.random()
        if random_n >= ratio:
            # test case
            my_test = my_test.append(data.iloc[i:i+speed])
        else:
            my_train = my_train.append(data.iloc[i:i+speed])
        i += speed
    if i != tot:
        my_train = my_train.append(data.iloc[i:])
    sys.stdout.write("\r\t"+str( round( 1*100, 1) )+" %")
    return my_train, my_test

# PRE-PROCESSING =================================================================================
"""
Drop useless features
"""
train.drop('Descript', axis=1, inplace=True)
train.drop('Resolution', axis=1, inplace=True)

"""
Add new features to the dataset:
    - Weekday (Monday, Tuesday, ...)
    - Hour of day
    - Month
    - Year
    - Day of month
"""
#train['DayOfWeek'] = train.Dates.dt.dayofweek
train['Hour'] = train.Dates.dt.hour
train['Month'] = train.Dates.dt.month
train['Year'] = train.Dates.dt.year
train['DayOfMonth'] = train.Dates.dt.day
#test['DayOfWeek'] = test.Dates.dt.dayofweek
test['Hour'] = test.Dates.dt.hour
test['Month'] = test.Dates.dt.month
test['Year'] = test.Dates.dt.year
test['DayOfMonth'] = test.Dates.dt.day

train.drop('Dates', axis=1, inplace=True)
test.drop('Dates', axis=1, inplace=True)

"""
Convert Category(label) to number
"""
label_cat = preprocessing.LabelEncoder()
crime = label_cat.fit_transform(train.Category)
train['Crime'] = crime

# To csv file
train.to_csv("my_train.csv", index = False)
test.to_csv("my_test.csv", index = False)

# DIVIDE THE TRAIN ===============================================================================
ratio = 0.75
speed = 100
local_train, local_test = split_dataframe(train, ratio, speed)

# Oh wait man, there already exists a function, as always!
local_train, local_test = train_test_split(train, train_size=0.75)

print("Final ratio:", round(len(local_train)/(len(local_train)+len(local_test)), 3), "for train,",
      round(len(local_test)/(len(local_train)+len(local_test)), 3), "for test")


# SAVE THE NEW DATASETS ==========================================================================
workspace = "C:\\Users\\Giammi\\OneDrive\\Università\\Machine Learning\\project\\data"
os.chdir(workspace)

local_train.to_csv("local_train.csv", index = False)
local_test.to_csv("local_test.csv", index = False)