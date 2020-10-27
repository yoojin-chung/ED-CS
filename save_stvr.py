# -*- coding: utf-8 -*-
"""
Read in Matlab process data from ED-CS study and plot STVRs, SGN and analyze.

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

# %% Find examples closest to mean or median
DF.pps = DF['pps'].astype('float')
DF.stvr = DF['stvr'].astype('float')

plt.figure()
ax2 = sns.lineplot(x='pps', y='stvr', hue='group', err_style='band',
                   ci=None, estimator='mean', data=DF)
ax2.set_xscale('log')

groups = DF.group.unique()
pps_set = DF.pps.unique()

# stvr_mean = []
# for j, group in enumerate(groups):
#     stvr_pps = []
#     for k, pps in enumerate(pps_set):
#         stvr_pps.append(np.mean(DF[(DF.group==group) & (DF.pps==pps)].stvr))
#     stvr_mean.append(stvr_pps)
stvr_mean = DF['stvr'].groupby([DF['group'], DF['pps']]).mean()
xx = np.array(DF_raw[DF_raw.group == 'ED-CS'].stvr)

diff = []
for k, x in enumerate(xx):
    # diff.append(np.sum((xx[k]-stvr_mean[:][1])**2))
    diff.append(np.sum((xx[k]-stvr_mean['ED-CS'])**2))

diff = np.array(diff)
ind = np.argmin(diff)
uni_list = DF_raw[DF_raw.group == 'ED-CS'].unit
print(uni_list.iloc[ind])

# %% Two-way anova analysis of STVR
# Set types for anova
DF.group = DF['group'].astype('category')
DF.pps = DF['pps'].astype('category')
DF.stvr = DF['stvr'].astype('float')

# Arcsine transform to make the data more Gaussian-like
DF['stvr_tr'] = np.arcsin(np.sqrt(DF.stvr))

# Sum is the jiggery pokery needed for 2-way anova with interaction
model = ols('stvr_tr ~ C(group, Sum) * C(pps, Sum)', data=DF).fit()
tbl = sm.stats.anova_lm(model, typ=3)
print('Two-way ANOVA with interaction')
print(tbl)

# Interaciton is not significant so we remove it for EF calculation
model = ols('stvr_tr ~ C(group) + C(pps)', data=DF).fit()
tbl = sm.stats.anova_lm(model, typ=2)
print('\nTwo-way ANOVA no interaction')
print(tbl)

# Add effect size here
part_omega = tbl.df['C(group)']*(tbl.sum_sq['C(group)']/tbl.df['C(group)']-
                                 tbl.sum_sq['Residual']/tbl.df['Residual'])/\
             (tbl.sum_sq['C(group)'] + (sum(tbl.df) + 1 - tbl.df['C(group)'])*
             tbl.sum_sq['Residual']/tbl.df['Residual'])
print('\nPartial omega squared for group: %0.5f' % part_omega)

part_omega = tbl.df['C(pps)']*(tbl.sum_sq['C(pps)']/tbl.df['C(pps)']-
                                 tbl.sum_sq['Residual']/tbl.df['Residual'])/\
             (tbl.sum_sq['C(pps)'] + (sum(tbl.df) + 1 - tbl.df['C(pps)'])*
             tbl.sum_sq['Residual']/tbl.df['Residual'])
print('\nPartial omega squared for pps: %0.5f' % part_omega)


# %% SGN
# Plot SGN count as function of duration of deafness
plt.figure()
ax3 = sns.scatterplot(data=DF_sgn, x="dur_deafness", y="SGN", hue="group")

# Test if there is a correlation between SGN and STVR
# Select best STVR per unit

# Use groupby!
# units = DF[~DF.SGN.isnull()].unit.unique()
# stvr_unit = []
# for k, unit in enumerate(units):
#     if (unit[0] != 'I'):
#         a = max(DF[DF.unit == units[k]].stvr)
#         b = unit
#         c = DF[DF.unit == units[k]].SGN.unique()
#         d = DF[DF.unit == units[k]].animal.unique()
#         e = DF[DF.unit == units[k]].group.unique()
#         stvr_unit.append([a, b, c[0], d[0], e[0]])
# DF_unit = pd.DataFrame(stvr_unit, columns=['stvr', 'unit', 'SGN', 'animal', 'group'])

DF_unit = DF[DF.group == 'ED-CS'].groupby(['animal', 'unit']).max()
tau, p_value = kendalltau(DF_unit.SGN, DF_unit.stvr)
plt.figure()
ax4 = sns.scatterplot(data=DF_unit, x="SGN", y="stvr", hue="animal")

print('Correlation between SGN and ITD stvr within ED-CS group')
print('Kendall\'s tau: %0.4f' % tau)
print('P value: %0.4f' % p_value)


# %%
# Save DF
# DF.to_csv('stvr.csv', index=False)
# F = open('stvr.pkl', 'wb')
# pickle.dump(DF, F)
# F.close()
