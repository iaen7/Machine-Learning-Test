# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:38:12 2017

@author: AM
"""

from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Load date.
df_total_consum = pd.read_csv("merge_power.csv")
df_weather = pd.read_csv("weather.csv",encoding = "gbk")
df_total_consum = df_total_consum.iloc[:,1:]
df_weather = df_weather.iloc[:,1:4]
df_total_merge = df_total_consum.merge(df_weather, on = 'record_date')
df_pred_weather = pd.read_excel('扬中.xls',7)
df_pred_weather = df_pred_weather.iloc[:,0:3]
dic = {"日期":"record_date","最高气温":"high_tem","最低气温":"low_tem"}
df_pred_weather =\
    df_pred_weather.rename(columns = dic)

#    
X,y = df_total_merge.iloc[:,2:].values, df_total_merge.iloc[:,1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
MLP = MLPRegressor(hidden_layer_sizes=(5000,))
MLP.fit(X_train,y_train)
y_train_pred = MLP.predict(X_train)
y_test_pred = MLP.predict(X_test)


print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))
'''

X_Sep = df_pred_weather.iloc[:,1:].values
y_Sep = forest.predict(X_Sep)

#Save as a DateFrame

period = pd.date_range(start = '20160901',end = '20160930')
pred_consum = pd.Series(y_Sep, index = period)
pred_consum = pred_consum.astype('int64')
pred_consum.index.rename('predict_date',inplace = True)
pred_consum.rename('predict_power_consumption', inplace = True)
pred_consum.to_csv('Tianchi_power_predict_table.csv')
'''