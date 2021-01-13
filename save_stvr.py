# -*- coding: utf-8 -*-
"""
Read Matlab data from ED-CS study and resave in Pandas dataframe.
Select a unit closest to the mean response curve to show.

Created on Mon Sep 28 17:25:05 2020
@author: yooji
"""

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import pickle
from scipy.io import loadmat
from statsmodels.formula.api import ols
from scipy.stats import kendalltau
from matplotlib import pyplot as plt

# %% Load data and plot STVR
# Matlab and excel files.
FN_stvr = 'data/all_stvr.mat'
FN_sgn = 'data/sgn.xlsx'

all_data = loadmat(FN_stvr)
DF_sgn = pd.read_excel(FN_sgn, sheet_name='python')

# Clean & organize Matlab data
groups = ['stvr_ms', 'stvr_cs', 'stvr_ad']
new_grp = ['ED-US', 'ED-CS', 'AD']
fields = ['stvr', 'pval', 'animal', 'group', 'unit', 'pps', 'deaf_dur']

DF = pd.DataFrame()
for n, key in enumerate(groups):
    data = all_data[key]
    group_data = []
    for k in range(data.shape[1]):
        row = {}
        for j, field in enumerate(fields):
            elem = data[0, k][j][0]
            if len(elem) <= 1:
                elem = data[0, k][j][0][0]
            row[field] = elem
        # Add SGN to DF
        row['SGN'] = np.mean(DF_sgn[DF_sgn.animal.str.contains(row['animal'])].SGN)
        row['group'] = new_grp[n]
        group_data.append(row)
    DF = DF.append(pd.DataFrame(group_data), ignore_index=True)

DF['animal'].loc[:][DF.animal == 'J1'] = 'J10' # This was a mistake in matlab db

# Unravel DF to include one pps, stvr, p-value set per row
DF_raw = DF
DF_pps = DF.explode('pps')
DF_pval = DF.explode('pval')
DF = DF.explode('stvr')
DF['pps'] = DF_pps.pps
DF['pval'] = DF_pval.pval

# Plot pps vs STVR
# plt.figure()
# ax1 = sns.boxplot(x="pps", y="stvr", hue="group", data=DF, palette="Set2")

# %% Find examples closest to mean
DF.pps = DF['pps'].astype('float')
DF.stvr = DF['stvr'].astype('float')

plt.figure()
ax2 = sns.lineplot(x='pps', y='stvr', hue='group', err_style='band',
                   ci=None, estimator='mean', data=DF)
ax2.set_xscale('log')

groups = DF.group.unique()
pps_set = DF.pps.unique()

stvr_mean = DF['stvr'].groupby([DF['group'], DF['pps']]).mean()
xx = np.array(DF_raw[DF_raw.group == 'ED-CS'].stvr)

diff = []
for x in xx:
    # diff.append(np.sum((xx[k]-stvr_mean[:][1])**2))
    diff.append(np.sum((x-stvr_mean['ED-CS'])**2))

diff = np.array(diff)
ind = np.argmin(diff)
uni_list = DF_raw[DF_raw.group == 'ED-CS'].unit
print(uni_list.iloc[ind])


# %%
# Save DF
# DF.to_csv('stvr.csv', index=False)
# F = open('stvr.pkl', 'wb')
# pickle.dump(DF, F)
# F.close()
