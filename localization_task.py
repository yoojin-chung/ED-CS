"""Read and plot localization task performance."""


import pandas as pd
import numpy as np
import os
from parse_file import parse_ini_config


def load_local(data_folder, animal_ID, ses_number):
    """Load all trials in ses_number and return results in DataFrame."""
    nses = len(ses_number)
    ses_data = []

    for q in range(nses):
        ses = ses_number[q]
        if animal_ID == 'J10' and ses == 11:  # J10-session11 was mislabled
            ses_path = os.path.join(data_folder, animal_ID, 'Session ' +
                                    str(ses), 'J6-Session10')
        else:
            ses_path = os.path.join(data_folder, animal_ID, 'Session ' +
                                    str(ses), animal_ID+'-Session'+str(ses))
        sum_fn = ses_path+'-Summary.ini'
        print(sum_fn)

        meta, _ = parse_ini_config(sum_fn, 0)
        ntrials = int(meta['Summary']['ntrials'])

        # Initialze variables for each trial
        n_tar = np.zeros(ntrials)
        data = []

        for r in range(ntrials):
            # Extract number of targets, vis cue, and outcome for each trial
            tr_fn = ses_path+'-Trial'+str(r+1)+'.txt'
            meta, loc = parse_ini_config(tr_fn, 1)

            n_tar[r] = int(meta['Arena Rects']['Targets'][1])
            success = \
                True if meta['Binaural Localization Data File']['Success'] \
                == 'TRUE' else False
            vis = 'Feeder LED intensity' not in \
                meta['Binaural Localization Data File']
            not_timeout = True if float(loc.iloc[-1]['Time(s)']) < 60 \
                else False
            data.append((success, vis, not_timeout))
        data = np.array(data)

        for n in set(n_tar):
            # Group trials into subsessions by number of targets
            tar_data = data[n_tar == n]
            tmp = list(calc_success(tar_data))
            tmp.extend([n, ses])
            ses_data.append(tmp)

    result = pd.DataFrame(ses_data, columns=['nVis', 'nAud', 'sucVis',
                                             'sucAud', 'nTar', 'ses'])
    return result


def calc_success(tar_data):
    """Calcuate success rate by type of trials."""
    success = tar_data[:, 0]
    vis = tar_data[:, 1]

    # not_timeout = tar_data[:, 2]
    # success = success[not_timeout]
    # vis = vis[not_timeout]

    nVis = np.sum(vis)
    nAud = np.sum(np.invert(vis))
    sucVis = sum(success[vis])
    sucAud = sum(success[np.invert(vis)])
    return nVis, nAud, sucVis, sucAud
