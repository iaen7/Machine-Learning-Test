# -*- coding: utf-8 -*-
"""
Created on Thu May 18 00:39:05 2017

@author: AM
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Tianchi_power.csv")
df_id1 = df[df.user_id == 1]
#date = df_id1.iloc[:,0]
#y1 = df_id1.iloc[:,2]
df_weather = pd.read_excel('扬中.xls',27)
for i in range(26,7,-1):
    df_weather = pd.concat([df_weather,pd.read_excel('扬中.xls',i)])