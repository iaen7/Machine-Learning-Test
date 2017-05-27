# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:50:25 2017

@author: AM
"""

import matplotlib.pyplot as plt
import pandas as pd
import re

df = pd.read_csv("Tianchi_power.csv")
df_id1 = df[df.user_id == 1]
df_weather = pd.read_excel('扬中.xls',27)
for i in range(26,7,-1):
    df_weather = pd.concat([df_weather,pd.read_excel('扬中.xls',i)],ignore_index= True)
df_weather = df_weather.iloc[:,0:4]
dic = {"日期":"record_date","最高气温":"high_tem","最低气温":"low_tem","天气":"weather"}
df_weather =\
    df_weather.rename(columns = dic)
df_weather.record_date = df_weather.record_date.str.replace('-','/')
datepat = re.compile(r'(\d+)/[0](\d)/(\d+)')
for i in range(len(df_weather)):
    df_weather.iloc[i,0] = datepat.sub(r'\1/\2/\3',df_weather.iloc[i,0])
datepat = re.compile(r'(\d+)/(\d+)/[0](\d)')
for i in range(len(df_weather)):
    df_weather.iloc[i,0] = datepat.sub(r'\1/\2/\3',df_weather.iloc[i,0])
df_merge1 = pd.merge(df_weather,df_id1,on = 'record_date')
df_merge1 = df_merge1.set_index('record_date')
df_merge1 = df_merge1.iloc[:,[0,1,2,4]]
df_weather.to_csv('weather.csv')
df_merge1.to_csv('merge_id1.csv')