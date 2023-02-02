# %%
import copy
import json
import logging  # import first to make sure logging works
import pathlib
import pickle
import pprint
import warnings
from collections import defaultdict

import box
import matplotlib.pyplot as plt
import numpy as np
import oasis
import pandas as pd
import scipy
import scipy.io
from numpy.core.fromnumeric import trace
from scipy import optimize, signal, stats
from scipy.optimize import dual_annealing, minimize
from scipy.signal import (convolve, decimate, medfilt, order_filter,
                          resample_poly, savgol_filter)
#from statsmodels.tsa.ar_model import AutoReg


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("filenum", type=int,
                    help="process_filenum")
args = parser.parse_args()
print(f"processing file number {args.filenum}")
f_num = args.filenum
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# f_num=0


# %%
basedir = pathlib.Path(".")
data_dir = basedir/ "DATA"

results_dir=basedir /"Results022_detect_only"
results_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=results_dir/f'detect_{f_num:04d}.log',
    level=logging.INFO
)

names=np.genfromtxt('names.txt',dtype='str')
print(names)
file_name = names[f_num]

from my_funcs_detect_and_save import *

# %%
# import at the end to make sure logging works


# %%
# params for preprocessing file
load_params                          = box.Box()
load_params.riseMarginTimeBracket    = 12  # time in ms taken before the peak (or first peak in the cluster) for fitting
load_params.decayMarginTimeBracket   = 24  # time in ms taken after the peak (or last peak in the cluster) for fitting
# params_global.decayMarginTimeBracket = 30  # time in ms taken after the peak (or last peak in the cluster) for fitting
load_params.eventGap                 = 2.0  # time in ms between two events so that the latter event is considered valid
load_params.cutoff                   = 1000
load_params.best_order               = 8
load_params.cutoff_high              = 2
load_params.order_high               = 4


detrend_params                   = box.Box()
detrend_params.f_new             = 5_000
detrend_params.savgol_len_ms     = 5
detrend_params.ord_len_ms        = 50 #was 201
detrend_params.f_ord_decimate    = 200  # was 101 
detrend_params.post_savgol       = False
# detrend_params.pre_savgol        = False #keep False?
detrend_params.subs_quantile     = 0.5   #orignally 0.25

# params_global.

split_params                  = box.Box()
split_params.thresh           = 0.2 # increase? #doesn't matter for findpeaks
split_params.split_gap_ms     = 2.0 # reduce? 
split_params.conv_filt_ms     = 2.5
split_params.factor           = 0.5    #final detect

segment_params         = box.Box()
segment_params.thresh  = 1.8 #*MAD lower numbers => more events per cluster
segment_params.pre_c_ms   = 10
segment_params.post_c_ms  = 20
segment_params.pre_j_ms   = 10
segment_params.post_j_ms  = 10



fit_params                  = box.Box()
fit_params.jitter           = 10e-3 #increase more esp with Dual Annealing
fit_params.height_low_fact  = 15 #increase more? 10/20?
fit_params.height_high_fact = 3
fit_params.tau1_min         = 0.5e-3
fit_params.tau1_max         = 50e-3
fit_params.tau1_init        = 5e-3
fit_params.tau2_min         = 0.1e-3
fit_params.tau2_max         = 10e-3
fit_params.tau2_init        = 0.7e-3
fit_params.noise_factor     = 1
fit_params.regularization   = 0.20
fit_params.order            = 2
fit_params.method           = "Powell"  
fit_params.maxiter          = 300
fit_params.maxfev           = 3000
fit_params.xtol             = 1e-8
fit_params.ftol             = 1e-8
fit_params.offset_fact      = 2
fit_params.maxiter_DA       = 3000
fit_params.maxfev_DA        = 30000

decon_params         = box.Box()
decon_params.sn_fact = 0.8  #initial_detect try both reduce and increase
decon_params.td1     = 3.5e-3
decon_params.tr1     = 0.7e-3



decon_params.max_iter= 10

# special for file 14
#decon_params.max_iter= 2

decon_params.penalty = 1  # try penalty1 with findpeaks


params                = box.Box()
params.load_params    = load_params
params.detrend_params = detrend_params
params.segment_params = segment_params
params.fit_params     = fit_params
params.decon_params   = decon_params
params.split_params   = split_params

params.makeplots      = True

# %%

if __name__ == '__main__':
    dat_file = data_dir / file_name
    f_name=dat_file.stem
    print(f_name)
    #break

    logging.info(f"""
    =============================================
    starting file {f_name} 
    =============================================
    """)
    

    print(f_name)
    param_file=results_dir / ("params_"+ dat_file.with_suffix('.json').name)
    decon_file=results_dir / ("decon_"+ dat_file.with_suffix('.csv').name)
    fit_file=results_dir / ("fit_"+ dat_file.with_suffix('.csv').name)
    
    

    out_dir = results_dir / ("segments_"+ f_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    params.to_json(indent=4,filename=param_file)
    
    # try:
    sig_dat = readMatlabFile(dat_file, params=params.load_params)
    decon_df,return_data=runfile(sig_dat,params,out_dir,f_name)
    # except:
    #     print(f"FAILED {f_name} ")
    #     logging.error(f"""
    #     =============================================
    #     FAILED FILE {f_name} 
    #     =============================================
    #     """)

    #saving
    # np.save(results_dir / (f_name +"_trace_sub_zeroed" ),trace_zeroed)
    # np.save(results_dir / (f_name +"_time_sub" ),times)

    
    # fit_df_h.to_csv(fit_file)
    decon_df.to_csv(decon_file)
    np.savez(results_dir / (f_name +"_return_data"),**return_data)
    logging.info(f"""
    =============================================
    finished file {f_name} 
    =============================================
    """)
    
        # break


    # %%


