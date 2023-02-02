import logging


# %%
import copy
import json
import pathlib
import pickle
import pprint
import warnings
from collections import defaultdict

import box
from box import Box

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy import optimize, signal, stats
from scipy.optimize import dual_annealing, minimize
# from scipy.signal import decimate, savgol_filter
from scipy.signal import (convolve, decimate, medfilt, order_filter,
                          resample_poly, savgol_filter)
import oasis
from collections import namedtuple


def rms(x, order=1.5):
    return np.linalg.norm(x, order) * len(x)**(-1/order)

def pritish_double_exp_unlog(ti, t_0, height, itau_1, itau_2):
    trel = ti-t_0
    trel = trel*(trel > 0.0)
    inv_subs = 1/(itau_1-itau_2)
    # from dayan abboth 5.34
    norm = -itau_1 * np.power(itau_1/itau_2, itau_2*inv_subs) * inv_subs
    return (height*norm)*(np.exp(-1*trel*itau_1) - np.exp(-1*trel*itau_2))*(trel > 0.0)


# weird ones
def func_double_exp(params, t1, locs, y_temp):
    """inplace function params-> fit
    Args:
        params ([type]): [description]
        t1 ([type]): [description]
        locs ([type]): [description]
        y_temp ([type]): [description]
    Returns:
        [type]: [description]
    """
    y_temp.fill(params[0])
    logtau_1 = np.log(np.exp(params[3::4]) + np.exp(params[4::4]))
    logtau_2 = params[4::4]
    o_times = locs + params[1::4]
    heights = np.exp(params[2::4])
    itaus_1 = np.exp(-1*logtau_1)
    itaus_2 = np.exp(-1*logtau_2)

    for o_time, height, itau_1, itau_2 in zip(o_times, heights, itaus_1, itaus_2):
        y_temp += pritish_double_exp_unlog(t1, o_time, height, itau_1, itau_2)
    return y_temp


def residual_double_exp(params, t1, y, locs, y_temp, noise=7, regularization=0.01, order=2):
    func_double_exp(params, t1, locs, y_temp)
    jitter = params[1::4]  # jitter
    # heights penalty rms(heights,order=0.5)/arr.mean()
    return 1*rms(y-y_temp, order)/noise + regularization * rms(jitter, order)/1e-3


def fit_double_exp_new(curr, params):

    cn=curr.cn
    num_peaks = len(curr.peaks)
    # make bound
    bounds = [(-1*params.offset_fact*curr.MAD, params.offset_fact *curr.MAD)]  # offset
    x_init = [0.0]
    for loc, height in zip(curr.ev_times, curr.heights):
        bounds.append((-1 * params.jitter, params.jitter))   # ot   # jitter =3.0e-3
        x_init.append(0.0e-3)

        # logheight  params.height_low_fact/high = 3
        bounds.append((np.log(height/params.height_low_fact), np.log(height*params.height_high_fact)))
        x_init.append(np.log(height))

        # roughly logtau1, actually (log(tau1 -tau2))
        bounds.append((np.log(params.tau1_min), np.log(params.tau1_max)))
        x_init.append(np.log(params.tau1_init))

        bounds.append((np.log(params.tau2_min), np.log(params.tau2_max)))             # other option  logtau2
        x_init.append(np.log(params.tau2_init))
    y_temp_sub = np.full_like(curr.trace, 0.0)

    # final fit on all data using the aprox fit as init conditions
    logging.info(f"fitting double_exp full trace: cluster {cn}")
    # y_temp = np.full_like(curr.trace_raw, 0.0)

    da_fit = dual_annealing(
        residual_double_exp,
        args=(curr.times, curr.trace, curr.ev_times,
            y_temp_sub, curr.MAD * params.noise_factor,
            params.regularization,params.order ),  # regularization =0.1 , noise_factor=1
        x0=x_init,
        bounds=bounds,
        seed=1234,
        maxiter=int(params.maxiter_DA*num_peaks),
        maxfun=int(params.maxfev_DA*np.sqrt(num_peaks)),
        no_local_search=True,
        # local_search_options={
        #     'method': 'Powell',
        #     'options' :  {
        #     'maxiter':2
            # }
        # },
    )
    logging.info(f'cluster {cn} : Dual Annealing fitting = {da_fit}')

    final_fit = minimize(
        residual_double_exp,
        x0=da_fit.x,
        bounds=bounds,
        method=params.method,  # methoid= "Powell"
        # method='L-BFGS-B',
        args=(curr.times, curr.trace, curr.ev_times,
            y_temp_sub, curr.MAD * params.noise_factor,
            params.regularization,params.order ),  # regularization =0.1 , noise_factor=1
        # jac='3-point',
        options={
            'maxiter': int(params.maxiter*num_peaks),  # params.maxiter=1000
            'maxfev': int(params.maxfev*num_peaks),  # params.maxfev=10000
            'xtol': params.xtol,  # params.xtol=1e-7
            'ftol': params.ftol  # params.ftol = 1e-7
            # 'eps':1e-9,
        }
    )

    if not final_fit.success:
        logging.warn(f"cluster {cn} : final fit did not converge")
        logging.info(f"cluster {cn} : " + final_fit.message)
    logging.info(f'cluster {cn} :res = {final_fit.fun}')
    final_fit_dict = dict(final_fit)
    final_fit_dict = {k: final_fit_dict[k] for k in ['fun',  'nit', 'nfev', 'status', 'success', 'message', 'x']}
    logging.info(f'cluster {cn} : \n{final_fit_dict}')
    logging.info(f"cluster {cn} : finished fitting double_exp full trace")

    return final_fit


def fit_and_parse(curr, fit_params,f_name):
    # curr, fit_params = packed
    fit_res = fit_double_exp_new(curr, fit_params)
    results_arr=[]
    for i, peak in enumerate(curr.peaks):
        res_peak = resNT(
            filename = f_name,
            ev_number = peak,
            baseline = fit_res.x[0],
            time = fit_res.x[4*i+1] + curr.ev_times[i],
            height = np.exp(fit_res.x[4*i+2]),
            tau_1 = np.exp(fit_res.x[4*i+3])+np.exp(fit_res.x[4*i+4]),
            tau_2 = np.exp(fit_res.x[4*i+4]),
            success = fit_res.success,
            cluster_number = curr.cn,
            tlim_min = curr.tl[0], 
            tlim_max = curr.tl[1]
        )
        results_arr.append(res_peak)
    return results_arr


resNT = namedtuple("resNT", ['filename','ev_number', 'baseline', 'time', 'height', 'tau_1',
                             'tau_2', 'success', 'cluster_number', 'tlim_min', 'tlim_max'])

fix_scalar = lambda x : x.item() if x.ndim==0 else  x

def read_npz(file_name):
    data= np.load(file_name,allow_pickle=True)
    data_dic={ k: fix_scalar(data[k]) for k in data}
    return Box(data_dic)
