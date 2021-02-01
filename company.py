# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:44:35 2020

@author: Chetan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv("D:\\chetan\\assignment\\11decision tree\\Company_Data.csv")
data.head()
data.describe()
data.info()
from sklearn.preprocessing import LabelEncoder

categorical_feature_mask = data.dtypes==object
print(categorical_feature_mask)
categorical_cols = data.columns[categorical_feature_mask].tolist()
print(categorical_cols)
le=LabelEncoder()
data[categorical_cols] = data[categorical_cols].apply(lambda col: le.fit_transform(col))
data[categorical_cols].head(10)
data.info()
data.describe()
data.isnull().sum()

bins = [0, 8, np.inf]
names = [0, 1]# 0=low sale,1=high sale
data['Sales_n'] = pd.cut(data['Sales'], bins, labels=names)
data.info()
data.head(5)

colnames = list(data.columns)
predictors = colnames[1:11]
target = colnames[11]
print(colnames)
data.fillna(method='ffill',inplace=True)

from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
type(preds)

pd.crosstab(test[target],preds)
np.mean(preds==test.Sales_n)

#trying to improve score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,make_scorer
from sklearn.model_selection import cross_val_score

n_components = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Create lists of parameter for Decision Tree Classifier
criterion = ['gini', 'entropy']
max_depth = [4,6,8,12]

parameters = dict(pca__n_components=n_components,
                      decisiontree__criterion=criterion,
                      decisiontree__max_depth=max_depth)
scorer = make_scorer(f1_score)

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_obj = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=10)
grid_fit = grid_obj.fit(train[predictors],train[target])

best_clf = grid_obj.best_estimator_

best_clf

scores = cross_val_score(best_clf,train[predictors],train[target])
print(scores)
print(scores.mean)

help(GridSearchCV)
