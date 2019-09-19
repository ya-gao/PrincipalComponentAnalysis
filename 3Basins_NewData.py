# -*- coding: utf-8 -*-
# @Author: Ya
# @Date:   2018-11-06
# @Last Modified by:   Ya
# @Last Modified time: 2019-03-06

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

import utils.dictutils as du
import utils.betaplotutils as pu

def new_parsefile(filename):
    import datetime as dt

    newds = du.datset(None, None)
    tmpdf = pd.read_csv(filename)
    tmpdf['date'] = [dt.datetime.strptime(i, "%Y-%m-%d") for i in tmpdf['date']]

    for dkey in list(tmpdf):
        if dkey == "date":
            continue
        newdf = pd.DataFrame({'date': tmpdf['date'], 'value': tmpdf[dkey]})
        newds[dkey] = newdf
        newds[dkey].units = ""
    return newds 

ds = new_parsefile("LakeErieCentral_daily_averaged_data.csv")
newds = new_parsefile("LakeErieEastern_daily_averaged_data.csv")
ds.append(None, None, newdat=newds)
newds = new_parsefile("LakeErieWestern_daily_averaged_data.csv")
ds.append(None, None, newdat=newds)

## vectorization 
y = np.array(ds["chla"]["value"])
nan_indices = np.argwhere(np.isnan(y))    ## returns indices of np.nan
## remove rows containing np.nan in y
keeper = np.isfinite(y)
y = y[keeper]

## vectorization
X_list = []
keepkeys = [
    'lwst',
    'cloud',
    'inflow',
    'precip',
    'evap',
    'surf_runoff',
    'wind_s',
    'wind_d'
]

dep = ds.dkeys().copy()
for key in ds.dkeys():
    if not (key in keepkeys):
        dep.remove(key)
print(dep)

for day in range(len(ds['chla']['date'])):
    Xi = []
    for x in dep:
        Xi.append(ds[x]["value"][day])
    X_list.append(Xi)
X = np.array(X_list)
print(X.shape)
## remove rows in X where associating y was np.nan
X = np.delete(X, (nan_indices), axis = 0)
print(X.shape)

## removes rows containing np.nan in X
nan_indices = np.argwhere(np.isnan(X))  ## returns indices of np.nan
nan_indices = np.delete(nan_indices, ([1]), axis = 1) ## returns row indices
X = np.delete(X, (nan_indices), axis = 0)   ## delete rows in X
print(X.shape)
y = np.delete(y, (nan_indices), axis = 0)   ## delete associating value in y
print(y.shape)

## standardization(center&scale the data: 0 means and unit variance for each y)
X_std = StandardScaler().fit_transform(X)
## X_std = preprocessing.scale(X) also works

## create a PCA object
pca = PCA() 
# pca = PCA(n_components=X.shape[1]) also works

pca.fit(X_std)
print(pca)
print('explained variance')
print(pca.explained_variance_)
print('explained variance ratio')
print(pca.explained_variance_ratio_)
## calculate percentage of variation that each component account for
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
print('percentage of variance')
print(per_var)

## Scree Plot
screefig, screeax = plt.subplots()
labels = ["PC" + str(x) for x in range(1, len(per_var)+1)]  ## create labels
screeax.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
pu.easy_axformat(
    screefig,
    screeax,
    title='Scree Plot',
    ylabel="Percentage of Explained Variance",
    xlabel="Principal Component",
    grid=True
)
plt.show()

## which feature contribute more to each PC
weighted_avg = [0,0,0,0,0,0,0,0]
for i in range(len(dep)):
    loading_scores = pd.Series(pca.components_[i], index=dep)
    for j in range(len(loading_scores)):
        weighted_avg[j] += loading_scores[j] * per_var[i]
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    sequence = sorted_loading_scores[:].index.values
    print('PC', i+1)
    print(loading_scores[sequence])
    labels = [x for x in dep]
    
    plt.bar(x=range(1, len(dep)+1), height=loading_scores, tick_label=labels, zorder=4)
    tmpax = plt.gca()
    pu.easy_axformat(
        1,
        tmpax,
        xlabel='Principal Component' + str(i+1),
        ylabel='Loading(Eigenvector)',
        title='Component Loadings of Principal Component' + str(i+1),
        grid=True
    )
    plt.show()
print(weighted_avg)

## weighted avg Plot
labels = [x for x in dep]

fig, ax = plt.subplots()
ax.bar(x=range(1, len(dep)+1), height=weighted_avg, tick_label=labels)
pu.easy_axformat(
    fig,
    ax,
    title='Loadings of Indicators',
    xlabel='Indicators',
    ylabel='Loading(Eigenvector)',
    grid=True
)
plt.show()











        
