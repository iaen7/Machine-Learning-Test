# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:50:54 2017

@author: AM
"""

import numpy as np

A = np.array([1,2,3])
s = A.shape[0]
for i in range(s):
    A[i] = 3
print (A)