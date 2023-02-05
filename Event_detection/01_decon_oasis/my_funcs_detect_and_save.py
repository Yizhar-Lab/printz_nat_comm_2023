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
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy import  signal, stats
# from scipy.optimize import dual_annealing, minimize
# from scipy.signal import decimate, savgol_filter
from scipy.signal import (convolve, decimate, medfilt, order_filter,
                          resample_poly, savgol_filter)
import oasis
from collections import namedtuple




def readMatlabFile(matfile, params):
    '''
    Read the data of the full recording trace, not parsed:
    '''
    logging.info(f"reading {matfile}")
    data_signal = box.Box()
    data = box.Box(scipy.io.loadmat(matfile, squeeze_me=True))
    data_signal.trace_raw = -data.trace_data

    data_signal.locations = data.trueEventIndsOrigin

    data_signal.f_sample = 1000*data.rate

    n=len(data_signal.trace_raw)
    step=1/data_signal.f_sample
    data_signal.t = np.arange(0,n*step,step)  # in sec
    data_signal.file_name = matfile.stem
    logging.info(f"finished reading {matfile}")
    return data_signal



# %%

def detrend_and_decimate_new(trace,f_sample, params):
    """
    detrend_and_decimate AI is creating summary for detrend_and_decimate

    Args:
        trace ([type]): array
        factor ([type]): (this much downsampling should give 2000Hz)
    """

    logging.info("detrending")
    
    f_new = int(params.f_new)
    print(f_sample,f_new)
    f_sample2= (int(f_sample)//1000)*1000
    print(f_sample2,f_new)
    leng =len(trace)

    up = int(f_new/np.gcd(f_sample2,f_new))
    down = int(f_sample2*up/f_new)
    print(up,down)
    factor=down/up
    logging.info(f"up = {up}, down = {down}")

    # up = int(100_000//f_sample)
    # down = int(100_000//f_new)


    trace_sub = resample_poly(trace,up,down,padtype='edge')
    dt=1/f_new
    times_sub = np.linspace(0.0,leng/f_sample,len(trace_sub))

    ord_filt_len = 2*(int(params.ord_len_ms*f_new/1000)//2)+1
    trace_sub2_ord = order_filter(trace_sub, np.ones(ord_filt_len), ord_filt_len//10)  # 10 percentile filter

    down_temp = int(f_new//params.f_ord_decimate) 
    print(f"down_temp = {down_temp}")
    trace_sub2_ord = decimate(trace_sub2_ord, down_temp, ftype='fir')
    trace_sub2_ord = medfilt(trace_sub2_ord)  #median filter after decimation
    trace_sub2_ord = resample_poly(trace_sub2_ord, down_temp, 1,padtype='edge')

    savgol_len1 = 2*(int(25*f_new/1000)//2)+1

    # trace_sub2_ord = savgol_filter(trace_sub2_ord, savgol_len1, 3, mode='interp')

    #added to fix length errors, URGH
    last_ind=min(len(trace_sub),len(trace_sub2_ord))
    
    trace_zerod = trace_sub[:last_ind]-trace_sub2_ord[:last_ind]
    
    times_sub = times_sub[:last_ind]


    MAD = stats.median_absolute_deviation(trace_zerod)



    if params.post_savgol:  # False
        savgol_len2 = 2*(int(params.savgol_len_ms*f_new/1000)//2)+1
        trace_zerod = savgol_filter(trace_zerod, savgol_len2, 3, mode='interp')  # params.savgol_len=7
    
    trace_zerod = trace_zerod - np.quantile(trace_zerod, params.subs_quantile)  # params.subs_quantile=0.25
    logging.info("finished detrending")
    
    # times[]

    return trace_zerod, times_sub, MAD , factor


def tau2g(td, tr, r=2000):
    """
    Converts the taus into the parameters of AR2 process

    Args:
        td ([type]): tau_d
        tr ([type]): tau_f
        r (int, optional): [description]. Defaults to 2000.
    """
    itau_1 = 1/td
    itau_2 = 1/tr

    inv_subs = 1/(itau_1-itau_2)
    # from dayan abott 5.34
    norm = -itau_1 * np.power(itau_1/itau_2, itau_2*inv_subs) * inv_subs
    (g1, g2) = (np.exp(-1/td/r)+np.exp(-1/tr/r), -np.exp(-1/td/r) * np.exp(-1/tr/r))
    return (g1, g2), norm


def cmax2ints(cmax, j1, MAD,f_new ,  params):
    """[summary]

    Args:
        cmax ([type]): [description]
        j1 ([type]): [description]
        MAD ([type]): [description]
        f_new ([type]): [description]
        params ([type]): [description]

    Returns:
        [type]: [description]
    """
    logging.info("clustering events")
    boo = cmax > params.thresh*MAD  # 1*MAD
    indices = np.nonzero(boo[1:] != boo[:-1])[0] + 1
    myind = np.indices(cmax.shape)[0]
    b = np.split(myind, indices)
    b = b[0::2] if boo[0] else b[1::2]
    tlims = [(x[0], x[-1]) for x in b]
    len(tlims)
    tlims = np.array(tlims)

    # fix edges:
    pre_c = int(f_new*params.pre_c_ms/1000)
    post_c = int(f_new*params.post_c_ms/1000)

    pre_j = int(f_new*params.pre_j_ms/1000)
    post_j = int(f_new*params.post_j_ms/1000)


    boo = cmax > params.thresh*MAD  # 1*MAD
    for tl in tlims:
        boo[tl[0]-pre_c: tl[1]+post_c] = True

    for j in j1:
        boo[j-pre_j: j+post_j] = True

    indices = np.nonzero(boo[1:] != boo[:-1])[0] + 1
    myind = np.indices(cmax.shape)[0]
    b = np.split(myind, indices)
    b = b[0::2] if boo[0] else b[1::2]
    tlims = [(x[0], x[-1]) for x in b]
    tlims = np.array(tlims)
    logging.info("finished clustering events")

    return tlims




def impulse_response(g, length):
    g1, g2 = g
    ts = np.arange(length)
    ys = [1.0]
    ys.append(g1*ys[0])
    for i, t in enumerate(ts):
        ynew = g1*ys[i+1] + g2*ys[i]
        ys.append(ynew)
    return np.array(ys)




def runfile(sig_dat, params,out_dir,f_name):

    decon_params = params.decon_params

    return_data=box.Box()

    trace_zeroed,times, MAD , factor = detrend_and_decimate_new(sig_dat.trace_raw, sig_dat.f_sample, params.detrend_params) #use this MAD for detection

    f_new = params.detrend_params.f_new
    logging.info(f"f_new = {f_new} :  ")
    
    #remove endpoints filterig artefacr
    ms25 = int(25*f_new/1000)
    trace_zeroed=trace_zeroed[ms25:-ms25]
    times=times[ms25:-ms25]
  
    return_data.trace_zeroed=trace_zeroed
    return_data.MAD=MAD
    return_data.times=times
    return_data.factor = factor

    logging.info(f"decimate factor = {factor} :  this should give approx {f_new} Hz")

    logging.info(f"MAD = {MAD}")
    MAD2 = stats.median_absolute_deviation(trace_zeroed)
    logging.info(f"MAD2 = {MAD2}")


    #sos_hpf = signal.butter(10, 100.0, 'hp', fs=f_new, output='sos') # 100Hz Hpf
    #filtered = signal.sosfiltfilt(sos, trace_zeroed)
    #MAD_hpf = stats.median_absolute_deviation(trace_zeroed)
    #logging.info(f"MAD2 = {MAD_hpf}")

    # make the AR2 parameters from taus
    g_first, norm_first = tau2g(decon_params.td1, decon_params.tr1,r=f_new)


    # peak of the ISR of AR2
    mult1 = impulse_response(g_first, int(100/1000*f_new) ).max()
    # mult2 = impulse_response(g_second, 100).max()
    return_data.mult1=mult1
    # run oasis
    logging.info("starting decon1")

    c1, s1, b, g, lam = oasis.functions.deconvolve(
        trace_zeroed,
        g=np.array(g_first),
        sn=MAD*decon_params.sn_fact,
        b=0.0,
        optimize_g=0,
        penalty=decon_params.penalty,
        max_iter = decon_params.max_iter,
        decimate=0
    )
    return_data.c1=c1
    return_data.s1=s1

    j1, h1 = s2evs_findpeaks(s1, mult1, MAD,f_new ,params.split_params)
    
    print(len(sig_dat.locations))
    logging.info(f"originally {len(sig_dat.locations)} events; decon found{len(j1)} ")
    # getting segments

    # cmax = np.where(c1 > c2, c1, c2)
    cmax = c1.copy()

    tlims = cmax2ints(cmax, j1, MAD,f_new , params.segment_params)

    cluster_num = np.searchsorted(tlims[:, 0], j1, "right")-1

    decon_df = pd.DataFrame()
    decon_df["event_loc_ind"] = j1
    decon_df['heights'] = h1
    decon_df['cluster_num'] = cluster_num
    decon_df['event_loc_time'] = times[j1]

    clusters_with_peaks = sorted(list(set(cluster_num)))
    # results_arr = []

    
    def curr_gen():
        for cn in clusters_with_peaks:
            logging.info(f"starting cluster {cn}")
            curr = box.Box()
            curr.cn=cn
            curr.peaks = np.where(cluster_num == cn)[0]
            curr.locs = j1[curr.peaks]
            curr.heights = h1[curr.peaks]
            curr.ev_times = times[curr.locs]
            curr.tl = tlims[cn]

            curr.seg = slice(curr.tl[0], curr.tl[1])
            curr.trace = trace_zeroed[curr.seg]
            curr.times = times[curr.seg]
            curr.MAD = MAD2 #use the final MAD for fitting
            logging.info(f"segment {cn} : number of events {len(curr.peaks)}")
            yield curr#,params.fit_params)
            # fit_res = fit_double_exp_new(curr, params.fit_params)
            # logging.info(f"finished cluster {cn}\n")
    curr_arr=[*curr_gen()]
    curr_arr.sort(key = lambda x:len(x.peaks),reverse = True)

    for num,curr in enumerate(curr_arr):
        np.savez(out_dir / f"{f_name}_segment_{curr.cn:04d}_{len(curr.peaks)}",**curr)


    return decon_df, return_data


resNT = namedtuple("resNT", ['ev_number', 'baseline', 'time', 'height', 'tau_1',
                             'tau_2', 'success', 'cluster_number', 'tlim_min', 'tlim_max'])



def s2evs_findpeaks(s, mult, MAD,f_new ,  params):
    logging.info("s2evs_findpeaks : finding individual events")
    window = signal.windows.triang(2*(int(f_new*params.conv_filt_ms/1000)//2)+1 )
    s2=signal.convolve(s,window,mode="same")
    split_gap = int(f_new*params.split_gap_ms/1000)
    jumps,p_dict=scipy.signal.find_peaks(s2,height=MAD*params.factor, prominence=MAD*params.factor,distance=split_gap)
    heights=[]
    for pt  in jumps:
        heights.append(mult*np.sum(s[pt-2:pt+4]))
    heights = np.array(heights)
    logging.info("s2evs_findpeaks : finished finding individual events")
    return jumps, heights
    
# %%
