#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:11:16 2019

@author: pkiser
"""

#%% 12.1
import matplotlib.pyplot as plt
import scipy.stats as st

#%% a)

#p(x) = X~Bin(1000,0.5)
st.binom.pmf(k=1000,n=1000,p=0.5)

#%% b)

# E = n * p = 500
#sqrt(e) ~= 31.6

#%% c)


# 0 + 0*sum

#%% d)

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np

# %matplotlib.rcParams['figure.dpi'] = 150

sp2012 = pd.read_table('./data/sp2012.txt')
df = DataFrame(sp2012)
plt.plot(df)
plt.xlabel("Zeit (Tage)")
plt.ylabel("Wert")
plt.title('S&P 500 - Aktienkurs 2012')
plt.show()

#%%

from scipy.stats import norm
steps = np.array(norm.rvs(size=250, loc=0.483, scale=11))
sp_simulated = np.empty([250])
sp_simulated[0] = 1257.6
for i in range(249):
    sp_simulated[i+1] = sp_simulated[i]+ steps[i]
plt.plot(sp_simulated)
plt.xlabel("Zeit (Tage)")
plt.ylabel("Wert")
plt.title('S&P - Aktienkurs (simuliert)')
plt.show()

#%% 12.2 a)


