#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 12:11:55 2019

@author: pkiser
"""

#%% Aufgabe 2.1
from pandas import Series
import matplotlib.pyplot as plt

grades = Series([4.2, 2.3, 5.6, 4.5, 4.8, 3.9, 5.9, 2.4, 5.9, 6, 4, 3.7, 5, 5.2, 4.5, 3.6, 5, 6, 2.8, 3.3, 5.5, 4.2, 4.9, 5.1])
#%% a) 
sorted = grades.sort_values(ascending=True)
sorted.index = np.arange(1, sorted.size+1)
sorted.describe()
#%%
middle = sorted.size/2 -1
sorted[middle-1] = sorted[middle-2] = sorted[middle-3] = 1
sorted.describe()

#%% b)
plt.subplot(221)
grades.plot(kind="hist", edgecolor="black")

plt.subplot(222)
sorted.plot(kind="hist", edgecolor="black")

plt.subplot(223)
grades.plot(kind="box")

plt.subplot(224)
sorted.plot(kind="box")

plt.show()

#%% Aufgabe 2.2
import pandas as pd
schlamm = pd.read_table(r"./data/klaerschlamm.dat", sep=" ", index_col=0)
schlamm = schlamm.drop("Labor",1)
schlamm.head()

#%% a)

schlamm.plot(kind="box")

#%%
schlamm.describe()

#%% b)

messfehler = schlamm - schlamm.median() 
messfehler.T.plot(kind="box")

#%% 2.4
import pandas as pd
import numpy as np

#%% a)
hubble = pd.read_table("./data/hubble.txt", sep=" ")
hubble.head()
hubble.plot(kind="scatter", x="recession.velocity", y="distance")

#%% b)
hubble.plot(kind="scatter", x="recession.velocity", y="distance")
b,a = np.polyfit(x=hubble["recession.velocity"],y=hubble["distance"], deg=1)
x = np.linspace(hubble["recession.velocity"].min(),hubble["recession.velocity"].max())

plt.plot(x, a + b*x, color="red")
plt.show()

#%% c)
hubble.corr().iloc[0,1]

#%% 2.5
import pandas as pd
import numpy as np

#%% a)
income = pd.read_table(r"./data/income.dat", sep=" ")
income.columns
income.plot(kind="scatter", y="Income2005", x="Educ")
income.plot(kind="scatter", y="Income2005", x="AFQT")

#%% b)
b,a = np.polyfit(x=income["Educ"], y=income["Income2005"], deg=1)
x = np.linspace(income["Educ"].min(), income["Educ"].max())

income.plot(kind="scatter", y="Income2005", x="Educ")
plt.plot(x, a+b*x, c="red")

print(f"a = {a}; b = {b}")

#%% c)
income.corr()

#%% 2.6

import matplotlib.pyplot as plt
import numpy as np
x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8])
y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])

#%% a)
plt.subplot(221)
plt.scatter(x=x, y=y1)
cx = np.linspace(x.min(), x.max())
b,a = np.polyfit(x=x, y=y1, deg=1)
plt.plot(cx, a+b*cx, c="red")

plt.subplot(222)
plt.scatter(x=x, y=y2)
b,a = np.polyfit(x=x, y=y2, deg=1)
plt.plot(cx, a+b*cx, c="red")

plt.subplot(223)
plt.scatter(x=x, y=y3)
b,a = np.polyfit(x=x, y=y3, deg=1)
plt.plot(cx, a+b*cx, c="red")

plt.subplot(224)
plt.scatter(x=x4, y=y4)
cx = np.linspace(x4.min(), x4.max())
b,a = np.polyfit(x=x4, y=y4, deg=1)
plt.plot(cx, a+b*cx, c="red")

plt.show()

#%% b)
b1,a1 = np.polyfit(x=x, y=y1, deg=1)
b2,a2 = np.polyfit(x=x, y=y2, deg=1)
b3,a3 = np.polyfit(x=x, y=y3, deg=1)
b4,a4 = np.polyfit(x=x4, y=y4, deg=1)

print(f"a1 = {a1}; b1 = {b1}")
print(f"a2 = {a2}; b2 = {b2}")
print(f"a3 = {a3}; b3 = {b3}")
print(f"a4 = {a4}; b4 = {b4}")

#%% c)
np.corrcoef(x=x, y=y1)
#%%
np.corrcoef(x=x, y=y2)
#%%
np.corrcoef(x=x, y=y3)
#%%
np.corrcoef(x=x4, y=y4)

#%% 2.7
from pandas import Series,DataFrame
import pandas as pd
from fancyimpute import KNN

#%% a)
data = pd.read_csv("./data/child.csv", sep=",", index_col=0)
data.describe()

#%% c)
data.describe()
data_nonan = data.dropna()
data_nonan.describe()
#%% b)
np.NaN in data # ?
#%% d)
data_twonan = data.dropna(axis=1, thresh=28)
data_twonan.describe()
#%% e)
values = data.values
data_imputed = DataFrame(KNN(k=3).fit_transform(values))
data_imputed
print(data_imputed)

