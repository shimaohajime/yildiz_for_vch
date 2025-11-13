'''
Minimally replicating the pyro implementation of Yildiz model for the publcation.
'''

# %load_ext autoreload
# %autoreload 2

from pyro.poutine import trace
from pyro.infer import SVI, Trace_ELBO
import torch
import pandas as pd
import numpy as np
import datetime
import argparse
import json
import os
import time
from copy import copy

import time

import sys

# Get the absolute path of pyro-npsde/src
pyro_npsde_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyro-npsde/src"))
sys.path.append(pyro_npsde_path)

from npsde_pyro import format_input_from_timedata, pyro_npsde_run

from preprocessing import *


def read_labeled_timeseries(df, reset_time = False, time_unit = 1, data_dim=None):
    """
    Load a CSV file that contains timeseries data delineated by labels.
    Modifying it from preprocessing.py to take dataframe instead of state dict.
    """

    labels = df.iloc[:,0].to_numpy()
    indices =  np.where(np.logical_not(np.equal(labels[1:], labels[:-1])))[0] + 1
    if not data_dim:
        data_dim = df.shape[1]-2

    time_column = np.split(df.iloc[:,1].to_numpy(dtype=np.float64) / time_unit, indices)
    data_columns = np.split(df.iloc[:,2:data_dim+2].to_numpy(dtype=np.float64), indices, axis=0) 

    if reset_time:
        time_column = [segment - segment[0] for segment in time_column]

    return (time_column, data_columns)



if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    data = 'MR-replication' #'Artist' # 'GMD-VDEM-rGDP' # 
    # Load the data
    if data == 'Artist':
        var = 'style'
        df_path = os.path.join(data_dir, "Artist_npsde_top100.csv")
        df = pd.read_csv(df_path)
        # When there's duplicate in (Label,Time), take the mean
        df = df.groupby(['Label', 'Time']).mean().reset_index()
        save_file_name = 'Artist_top100'
    elif data == 'GMD-VDEM-rGDP':
        df_path = os.path.join(data_dir, "GMD_VDEM_npsde_rGDP_pc.csv")
        df = pd.read_csv(df_path)
        save_file_name = 'GMD_VDEM_rGDP_pc'
    elif data == 'GMD-VDEM-exports':
        df_path = os.path.join(data_dir, "GMD_VDEM_npsde_exports_GDP.csv")
        df = pd.read_csv(df_path)
        save_file_name = 'GMD_VDEM_exports'
    elif data == 'MR-replication':
        df_path = os.path.join(data_dir, "mr_repliciation_for_npsde_pyro.csv")
        df = pd.read_csv(df_path)
        # When there's duplicate in (Label,Time), take the mean
        df = df.groupby(['Label', 'Time']).mean().reset_index()
        save_file_name = 'MR_replication'
        
    
    time_series, data_series = read_labeled_timeseries(df, reset_time=True)

    X = format_input_from_timedata(time_series, data_series)

    start_time = time.time()

    print('Training the model...')
    #pyro_npsde_run(X, n_vars, steps, lr, Nw, sf_f,sf_g, ell_f, ell_g, noise, W, fix_sf, fix_ell, fix_Z, delta_t, \
    # save_model=None, Z=None, Zg=None, U_map=None, Ug_map=None)

    npsde = pyro_npsde_run(X, 2, 50, 0.02, 50, 1, 0.2, [1.0, 1.0], 0.5, [1.0, 1.0], 7, 0, 0, 0, 0.1, \
        save_model=f'{save_file_name}_pyro_model', Z=None, Zg=None, U_map=None, Ug_map=None)
    
    npsde.plot_model(X, f"{save_file_name}_pyro_model", Nw=50)

    print('Training time: ', time.time() - start_time)
