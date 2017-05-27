# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:31:34 2017

@author: AM
"""

import numpy as np

class Perceptron(object):
    '''
    Parameter: 
        eta:float and n_iter:int
    Attributes:
        w_:1-d array and errors:list
    '''
    def __init__(self, eta = 0.1, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        
    '''
    adjust the parameter of w_
    '''
    
    def fit(self,X,y):
        self.w_ = np.zeros(X.shape[1] + 1)
        self.errors = []
        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                if update != 0.0:
                    error += 1
            self.errors.append(error)
                
    '''
    input X
    This function return a estimate vector Y
    '''
    def net_res(self, X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_res(X) >= 0.0, 1, -1)
    