# -*- coding: gbk -*-
"""
Created on Wed May 24 00:03:51 2017

@author: AM
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df1 = pd.read_csv('merge_id1.csv',encoding = 'gbk')
Z = df1[df1.power_consumption <= 1000]
Z = Z.iloc[:,[1,2,4]].values
X,y = Z[:,:-1],Z[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
forest = RandomForestRegressor(
        n_estimators=1000,
        criterion='mse',
        random_state =1,
        n_jobs = -1)
forest.fit(X_train,y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

plt.subplot(2,2,3)
plt.show()