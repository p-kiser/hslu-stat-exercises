#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:45:05 2019

@author: pkiser

"""

#%% 13.1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.graphics.tsaplots import plot_acf

Tesla = pd.read_csv("./data/Tesla.csv", sep="\t",header=0)
Tesla["Date"] = pd.DatetimeIndex(Tesla["Date"])
Tesla.set_index("Date", inplace=True)
Tesla["log_volume"] = np.log(Tesla["Volume"])
Tesla["log_return"] = Tesla["log_volume"] - Tesla["log_volume"].shift(1)
Tesla["log_return"].plot()

plot_acf(DataFrame(Tesla["log_return"]).dropna(), lags=120, ax=ax2)

#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from statsmodels.graphics.tsaplots import plot_acf

Tesla = pd.read_csv("./data/Tesla.csv", sep="\t",header=0)
Tesla["Date"] = pd.DatetimeIndex(Tesla["Date"])
Tesla.set_index("Date", inplace=True)
Tesla["log_volume"] = np.log(Tesla["Volume"])
Tesla["log_return"] = Tesla["log_volume"] - Tesla["log_volume"].shift(1)
fig, (ax1, ax2) = plt.subplots(ncols=2)
Tesla["log_return"].plot(ax=ax1)
plot_acf(DataFrame(Tesla["log_return"]).dropna(), lags=120, ax=ax2)

#%% 13.2

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 1, 1000, endpoint=False)
noise = np.random.normal(size=1000)
signal = 0.5*t
plt.plot(t, signal + noise)
plt.ylim(-2, 2)
plot_acf(DataFrame(signal + noise).dropna(), lags=120, ax=ax2)
#%% random walk

import matplotlib.pyplot as plt
import numpy as np
d = np.random.choice(a=[-1,1], size=10000, replace=True)

# no drift
x = np.cumsum(d) 

plt.plot(x)
plt.xlabel("Random Walk")
plt.ylabel("y-Abweichung in [m]")
plt.show()


#%% random walk with drift

d = np.random.choice(a=[-1,1], size=10000, replace=True)
delta = 5*10**(-2)
x = np.cumsum(d)
y = np.zeros(10000)
for i in range(1,10000):
    y[i] = delta+y[i-1]+d[i]
plt.plot(y)
plt.plot(x)
plt.xlabel("Random Walk mit Drift")
plt.ylabel("y-Abweichung in [m]")
plt.show()

#%% gaussian white noise

w = np.random.normal(size=1000)
plt.plot(w)
plt.show()

#%% rolling window
from pandas import DataFrame

n = 3

w = DataFrame(np.random.normal(size=1000))
w.rolling(window=n).mean().plot()
plt.show()

#%% autoregressive zeitreihen

d = np.random.choice(a=[-1,1], size=10000, replace=True)
a = np.zeros(10000)

u = 1.1
v = -0.9

for i in range(2,10000):
    a[i] = (u*a[i-1]) + (v*a[i-2]) + d[i]
plt.plot(a)
plt.show()

#%% auto correlation function

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

w = DataFrame(np.random.normal(size=1000))
MA = DataFrame(w.rolling(window=3).mean()).dropna()

plot_acf(MA, lags=12, c="C1")
plt.vlines(x=2.1, ymin=0, ymax=1/3, color="red", linestyle='--', label="Theoretisch")
plt.vlines(x=1.1, ymin=0, ymax=2/3, color="red", linestyle='--')
plt.vlines(x=0.1, ymin=0, ymax=1, color="red", linestyle='--')
plt.legend()


#%%

temps = np.array([21, 23, 22, 23, 23, 25, 24, 25, 26, 28, 27])
n_days = temps.size
days = np.arange(n_days)
slope, intercept = np.polyfit(days, temps, 1)
estimates = intercept + slope * days

plt.plot(temps)
plt.plot(estimates)
plt.show()

def find_temp_error(days, temps, intercept, slope):
    estimates = intercept + slope * days
    temp_error = temps - estimates
    return temp_error

temp_error = find_temp_error(days, temps, intercept, slope)
squared_error = temp_error ** 2

np.corrcoef(days, temps)[0,1]

#%% autocorrelation
 days_i_temps = temps[1:]
 days_i_minus_1 = temps[:-1]
 np.corrcoef(days_i_minus_1, days_i_temps)
 
