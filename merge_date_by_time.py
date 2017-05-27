# -*- coding: utf-8 -*-
"""
Created on Fri May 26 00:29:20 2017

@author: AM
"""

import matplotlib.pyplot as plt
import pandas as pd
import re

df = pd.read_csv("Tianchi_power.csv")
df_id1 = df[df.user_id == 1]
df_sum_power = pd.DataFrame(columns = ['record_date','power_consumption'])
df_sum_power['record_date'] = df_id1['record_date']
for date in df_id1['record_date']:
    df_sum_power.loc[df_sum_power.record_date == date,'power_consumption'] =\
                sum(df.loc[df.record_date == date,'power_consumption'].values)
df_sum_power.to_csv('merge_power.csv')
