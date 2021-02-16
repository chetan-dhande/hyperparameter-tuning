# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:33:22 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("Fraud_check.csv")
data.info()
data.isnull().sum()
data.describe()

categorical_feature_mask = data.dtypes==object
print(categorical_feature_mask)

categorical_cols = data.columns[categorical_feature_mask].tolist()
print(categorical_cols)

le=preprocessing.LabelEncoder()
data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
data[categorical_cols].head(10)
data.info()

bins = [0, 30000, np.inf]
names = [0, 1]# 0=risky,1=good
data['Taxable.Income'] = pd.cut(data['Taxable.Income'], bins, labels=names)
data.info()
data['Taxable.Income'].head()
data.columns

x= data[['Undergrad', 'Marital.Status','City.Population','Work.Experience', 'Urban']]
y= data[['Taxable.Income']]

x_train,x_test,y_train,y_test= train_test_split(x, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(x_train,y_train)
preds = model.predict(x_train)
#type(preds)
#pd.Series(preds).value_counts()

#y_train
#a=pd.Series(preds)
#np.array('y_train')
#pd.crosstab(a,y_test)

import sklearn
sklearn.metrics.confusion_matrix(y_train, preds)

from sklearn.metrics import accuracy_score
accuracy_score(y_train, preds)
