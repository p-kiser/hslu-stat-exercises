#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:28:30 2019

@author: pkiser
"""


import numpy as np

#%% a)
x = np.loadtxt(r"./data/oldfaithful.txt")
n = np.size(x)
nboot = 1000
tmpdata = np.random.choice(x, n*nboot, replace=True)
bootstrapsample = np.reshape(tmpdata, (n, nboot))
xbarstar = np.mean(bootstrapsample, axis=0)
d = np.percentile(xbarstar, q=[2.5, 97.5])
print("Vertrauensintervall: ",d)

#%% b)
x = np.loadtxt(r"./data/oldfaithful.txt")
n = np.size(x)
nboot = 1000
tmpdata = np.random.choice(x, n*nboot, replace=True)
bootstrapsample = np.reshape(tmpdata, (n, nboot))
xbarstar = np.median(bootstrapsample, axis=0)
d = np.percentile(xbarstar, q=[2.5, 97.5])
print("Vertrauensintervall: ",d)

#%% c)
