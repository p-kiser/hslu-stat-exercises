#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:53:29 2019

@author: pkiser
"""

# serie 11

#%% c)
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
pw_electric = pd.read_csv('data/PW_electric.csv', sep=',', skiprows=2,
                          header=0, encoding = "utf-8", index_col=0)
pw_electric.head()
#%%
pw_electric_luzern = DataFrame(pw_electric.ix["Luzern",1:])
pw_electric_luzern
pw_electric_luzern["Year"] = pd.DatetimeIndex(pw_electric_luzern.index)
pw_electric_luzern.set_index("Year", inplace=True)
pw_electric_luzern.plot()
plt.xlabel("Jahr")
plt.ylabel("Anzahl Elektro-Autos Luzern")
plt.show()

#%% d)

pw_electric_zurich = DataFrame(pw_electric.loc["Zürich",'1990':'2017'])
pw_electric_zurich["Year"] = pd.DatetimeIndex(pw_electric_zurich.index)

pw_electric_zurich.set_index("Year", inplace=True)
pw_electric_zurich.plot()
plt.xlabel("Jahr")
plt.ylabel("Anzahl Elektro-Autos Zürich")
plt.show()

#%% e)
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np

# Relativer Zuwachs in Luzern
pw_electric_luzern["rel"] = np.log(pw_electric_luzern.astype('float'))
- np.log(pw_electric_luzern.shift(1).astype('float'))

# Relativer Zuwachs in Zürich
pw_electric_zurich["rel"] = np.log(pw_electric_zurich.astype('float'))
- np.log(pw_electric_zurich.shift(1).astype('float'))

pw_rel = pd.DataFrame({"Luzern" : pd.Series(pw_electric_luzern["rel"]),
"Zürich" : pd.Series(pw_electric_zurich["rel"])})
pw_rel.plot()
plt.show()

#%% 11.2

AusBeer = pd.read_csv("data/AustralianBeer.csv",sep=";",header=0)
AusBeer.head()
AusBeer["Quarter"] = pd.DatetimeIndex(AusBeer["Quarter"])
AusBeer.set_index("Quarter", inplace=True)
AusBeer.columns=["Megalitres"]
AusBeer.head()
AusBeer.describe()
AusBeer.plot()
plt.ylabel("Megalitres Beer")

#%% b)

AusBeer.resample("A").mean().plot()
AusBeer['quarter'] = AusBeer.index.quarter
AusBeer.boxplot(by="quarter")

#%% c)

from statsmodels.tsa.seasonal import seasonal_decompose

AusBeer1 = AusBeer.copy()
seasonal_decompose(AusBeer1, model="additive", freq=4).plot()

#%%
from stldecompose import decompose
AusBeer_stl = decompose(AusBeer["Megalitres"], period=12)
AusBeer_stl.plot();

#%% 11.3

Electricity = pd.read_csv("data/AustralianElectricity.csv", sep=";", header=0)
Electricity.head()
Electricity["Quarter"] = pd.DatetimeIndex(Electricity["Quarter"])
Electricity.set_index("Quarter", inplace=True)
Electricity.columns=["Electricity production Australia"]
Electricity.head()
Electricity.plot()
plt.ylabel("Million Kilowatthours")

#%% b)

def boxcox(x,lambd):
    return np.log(x) if (lambd==0) else (x**lambd-1)/lambd
# replace "yourSeries" by the name of your series
Electricity_tr = boxcox(Electricity, 0.3)
Electricity_tr.plot()
plt.show()

#%% c)
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(Electricity_tr, model="additive", freq=4).plot()
plt.show()

#%% d)
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_decompose(Electricity_tr, model="additive", freq=4).plot()
plt.show()
