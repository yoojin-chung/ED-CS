# -*- coding: utf-8 -*-
"""
Read and save localization task performance for each animal.

Plotting moved to Jupyter notebook.
Created on Thu Sep 17 14:30:58 2020
@author: yooji
"""

import pandas as pd
import os
import pickle
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
