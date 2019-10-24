#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:43:16 2019

@author: pkiser
"""
#%% Aufgabe 1.1
from pandas import Series,DataFrame
import pandas as pd

#%% a) Datei einlesen
data = pd.read_csv(r"./data/child.csv", sep=",", index_col=0)
data.head()

#%% b)
data.shape

#%% c)
data.describe()
data.mean()
data.median()

#%% d)
for c in ['Netherlands','China']:
    if (c in data.index):
        print(f"{c} is in data")
    else:
        print(f"{c} is NOT in data")

data.index
data.columns

#%% e)
drunk = data.sort_values(by='Drunkenness', ascending=False)
drunk["Drunkenness"].iloc[:5]

#%% f)
data['Infant.mortality'].nsmallest(1)

#%% g)
data.loc[data['Physical.activity'].mean() > data['Physical.activity'], :]['Physical.activity']
data['Physical.activity'].mean()

#%% Aufgabe 1.2
import pandas as pd
from pandas import DataFrame, Series

#%% a)
fuel = pd.read_table(r"./data/d.fuel.dat", sep=",", index_col=0)

#%% b)
fuel.loc[5, :]

#%% c)
fuel.loc[1:5]

#%% d)
fuel["mpg"].mean()

#%% e)
fuel.loc[7:22, "mpg"].mean()

#%% f)
t_kml = fuel["mpg"]*1.6093/3.789
t_kg = fuel["weight"]*0.45359

#%% g)

print(f"Mean Km/l: {t_kml.mean()}")
print(f"Mean Kg: {t_kg.mean()}")

#%% Aufgabe 1.3
import math
import pandas as pd
from pandas import DataFrame, Series

x = Series([2.1,2.4,2.8,3.1,4.2,4.9,5.1,6.0,6.4,7.3,10.8,12.5,13.0,13.7,14.8,17.6,19.6,23.0,25.0,35.2,39.6])

#%% a)
x.sum()
(x**2).sum()

#%% b)
mean = x.sum()/x.size
std = math.sqrt(1/(x.size-1) * ((x-x.mean())**2).sum())

print(f"Mean (calc): {mean}; Mean (pandas){x.mean()}")
print(f"Std (calc): {std}; Std (pandas){x.std()}")

#%% c)
x_sorted = x.sort_values()

if (x.size % 2 == 0):
    median = x_sorted.loc[x.size/2]
else:
    lower = math.floor(x.size/2)
    upper = math.ceil(x.size/2)
    median = (x[lower]+x[upper])/2
    
print(f"Mean (calc): {mean}; Mean (pandas){x.mean()}")

#%% d)
x.quantile(q=.75)

#%%

z = (x - x.mean()) / x.std()

print(f"std = {round(z.std(),2)}")
print(f"mean = {round(z.mean(),2)}")

#%% Aufgabe 1.5
from pandas import Series, DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% a)
geysir = pd.read_table(r"./data/geysir.dat", sep=" ", index_col=0);
geysir.head()

#%% b)
plt.subplot(221)
geysir["Zeitspanne"].plot(kind="hist", edgecolor="black")
plt.xlabel("10 Klassen")
plt.subplot(222)
geysir["Zeitspanne"].plot(kind="hist",
bins=20,
edgecolor="black")
plt.xlabel("20 Klassen")
plt.subplot(223)
geysir["Zeitspanne"].plot(kind="hist",
bins=np.arange(41,107,11),
edgecolor="black")
plt.xlabel("Klassengrenzen 41, 52 , 63, 74 , 85, 96")
plt.show()

#%% c)
geysir["Eruptionsdauer"].plot(kind="hist",
normed=True,
cumulative=True)

plt.axhline(y=0.6,  color='r',linestyle='--')
plt.axvline(x=2, color='b',linestyle='--')