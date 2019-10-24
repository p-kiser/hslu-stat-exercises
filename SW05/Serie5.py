#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:45:35 2019

@author: pkiser
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

#%%
n=10
x = st.norm.rvs(size=n)
st.probplot(x,plot=plt)
    #%%
n=20
x = st.norm.rvs(size=n)
st.probplot(x,plot=plt)
#%%
n=50
x = st.norm.rvs(size=n)
st.probplot(x,plot=plt)
#%%
n=100
x = st.norm.rvs(size=n)
st.probplot(x,plot=plt)
#%%
n=1000
x = st.norm.rvs(size=n)
st.probplot(x,plot=plt)

#%%
x = st.t.rvs(size=20, df=20)
st.probplot(x,plot=plt)
#%%
x = st.t.rvs(size=100, df=20)
st.probplot(x,plot=plt)

#%%
x = st.t.rvs(size=20, df=1)
st.probplot(x,plot=plt)
#%%
x = st.t.rvs(size=100, df=1)
st.probplot(x,plot=plt)