# -*- coding: utf-8 -*-
"""
Read and save localization task performance for each animal.

Plotting moved to Jupyter notebook.
Created on Thu Sep 17 14:30:58 2020
@author: yooji
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import binom_test
from functools import partial
from localization_task import load_local

# %%
data_folder = os.path.join("C:\\", "Users", "yooji", "Dropbox",
                           "work", "research", "Rabbit IC ED CS", "behavior")

animals = ['J6', 'J7', 'J8', 'J10']
ses_num = [30, 30, 44, 46]
res_all = dict()

for n in range(len(animals)):
    res_all[animals[n]] = \
        load_local(data_folder, animals[n], range(1, ses_num[n]+1))

F = open('behav.pkl', 'wb')
pickle.dump(res_all, F)
F.close()

# %%
animals = ['J6', 'J7', 'J8', 'J10']
F = open('behav.pkl', 'rb')
res_all = pickle.load(F)
F.close()

DFbeh = pd.DataFrame()
for n, animal in enumerate(animals):
    DF = res_all[animal]
    DF['animal'] = animal
    DFbeh = DFbeh.append(DF, ignore_index=True)

DFbeh.to_csv('behav.csv')

# # %% Moved to Jupyter notebook. Using seaborn.
# my_binom = partial(binom_test, alternative='greater')
# fig = plt.figure()
# # fig.subplots_adjust(wspace=0, hspace=0)
# for n, animal in enumerate(animals):
#     # result = res_all[animals[n]]
#     result = DFbeh[DFbeh.animal==animal]
#     vis_res = np.array(list(map(
#         my_binom, result['sucVis'], result['nVis'], 1/(result['nTar']-1))))
#     aud_res = np.array(list(map(
#         my_binom, result['sucAud'], result['nAud'], 1/(result['nTar']-1))))

#     ax = fig.add_subplot(2, 2, n+1)
#     ax.scatter(result['ses'][result['nVis'] != 0],
#                vis_res[result['nVis'] != 0], marker='x', label='Vis+Aud')
#     ax.scatter(result['ses'][result['nAud'] != 0],
#                aud_res[result['nAud'] != 0], marker='d', label='Aud')
#     ax.plot([1, 46], [0.05, 0.05], '--k')
#     if n > 1:
#         ax.set_xlabel('Session')
#     if n % 2 == 0:
#         ax.set_ylabel('P-value')
#     if n == 0:
#         ax.legend(loc='best')
#     ax.set_xlim([1, 50])
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#     # ax.tick_params(direction='in')
#     ax.set_title(animals[n])

# # plt.savefig('behav_result.eps', format='eps')
