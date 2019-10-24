#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:05:06 2019

@author: pkiser
"""

from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.formula.api import ols

#%% a)
df=DataFrame({
        "Typ": np.repeat(["T1", "T2", "T3", "T4"], [6, 6, 6, 6]),
        "Druckfestigkeit" : [655.5, 788.3, 734.3, 721.4, 679.1, 699.4,
                             789.2, 772.5, 786.9, 686.1, 732.1, 774.8,
                             737.1, 639.0, 696.3, 671.7, 717.2, 727.1,
                             535.1, 628.7, 542.4, 559.0, 586.9, 520.0]
})

print(df)

#%% 
sns.stripplot(x="Typ", y="Druckfestigkeit", data=df)
plt.xlabel("Typ")
plt.ylabel("Druckfestigkeit")
plt.show()

#%%
sns.boxplot(x="Typ", y="Druckfestigkeit", data=df)
plt.xlabel("Typ")
plt.ylabel("Druckfestigkeit")
plt.show()

#%% b)

fit = ols("Druckfestigkeit~Typ", data=df).fit()
fit.params

#%% c)

# H_0 = mu_1 = mu_2 = mu_3 = mu_4
from statsmodels.stats.anova import anova_lm
anova_lm(fit)

#%% 10.2

from pandas import DataFrame
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
df=DataFrame({
"Behandlung": np.repeat(["A", "B", "C", "D"], [4, 6, 6, 8]),
"Koagulationszeit" : [62, 60, 63, 59, 63, 67,
71, 64, 65, 66, 68, 66,
71, 67, 68, 68, 56, 62,
60, 61, 63, 64, 63, 59]
})

#%% a)
sns.stripplot(x="Behandlung", y="Koagulationszeit", data=df)
plt.xlabel("Behandlung") # 
plt.ylabel("Koagulationszeit")      
plt.show()
sns.boxplot(x="Behandlung", y="Koagulationszeit", data=df)
plt.xlabel("Behandlung")
plt.ylabel("Koagulationszeit")
plt.show()

#%% b)
gm = df.mean()
#%% c)

for behandlung in ["A","B","C","D"]:
    print(behandlung,"MEAN: ", df[df['Behandlung'] == behandlung].mean())
    print(behandlung,"VAR: ", df[df['Behandlung'] == behandlung].var())

#%% d)
# Pooled Var: MS_E =  SS_E / DF_E
# DF_E = n - g

n=df["Koagulationszeit"].size
g=4
dfe = 1/(n-g)
dfe
    
#%%  f)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

fit = ols("Koagulationszeit~Behandlung", data=df).fit()
fit.params

#%%
anova_lm(fit)

    
    