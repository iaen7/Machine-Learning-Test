# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 00:05:22 2017

@author: AM
"""

import perceptron as ppt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header = None)
y = df.iloc[:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

ppn = ppt.Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X,y)
print(ppn.predict(X))
print (ppn.errors)
plt.plot(range(1,len(ppn.errors) + 1),ppn.errors, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Numbers of misclassification')
plt.show()