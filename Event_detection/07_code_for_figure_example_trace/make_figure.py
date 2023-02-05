# %%
import numpy as np
import pandas as pd
import logging
from box import Box
import matplotlib.pyplot as plt
import pathlib

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
# %%
def read_npz(file_name):
    data= np.load(file_name,allow_pickle=True)
    data_dic={ k: fix_scalar(data[k]) for k in data}
    return Box(data_dic)
fix_scalar = lambda x : x.item() if x.ndim==0 else  x

# %%
filename="File_2018_11_01_0025_Cell3_return_data.npz"

data = read_npz(filename)
# %%
fit_events= pd.read_csv("all_combined_with_80_20_taus.csv")
# %%
meta_df = pd.read_csv("META_DF.csv")
# %%
evoked_df = pd.read_csv("EVOKED_DF.csv")


#%%
decon_data =  pd.read_csv("decon_File_2018_11_01_0025_Cell3.csv",index_col=0)

# %%
curr_evoked_selected = evoked_df.query("date == '2018-11-01' and cell == 3 and rep_num == 0 and pre_num == 100" )

#%%
ll,ul = curr_evoked_selected.time.min()-0.03,curr_evoked_selected.time.max()+0.03
i_ll,i_ul=np.where((ll<data.times) & (data.times<ul))[0][[0,-1]]

#%%
curr_evoked_unselected = fit_events.query(f"date == '2018-11-01' and cell == 3 and  record ==25 and {ll}<time<{ul}" )
#%%
curr_decon=decon_data.query(f"{ll}<event_loc_time<{ul}").copy()




#%%
curr_decon['cn_mod5']=curr_decon.cluster_num.apply(lambda x : x%5)



#%%
def pritish_double_exp_unlog(ti, t_0, height, itau_1, itau_2):
    trel = ti-t_0
    trel = trel*(trel > 0.0)
    inv_subs = 1/(itau_1-itau_2)
    # from dayan abboth 5.34
    norm = -itau_1 * np.power(itau_1/itau_2, itau_2*inv_subs) * inv_subs
    return (height*norm)*(np.exp(-1*trel*itau_1) - np.exp(-1*trel*itau_2))*(trel > 0.0)


# weird ones
def func_double_exp(params, t1, y_temp):
    y_temp.fill(params[0])
    logtau_1 = np.log(params[3::4]) 
    logtau_2 = np.log(params[4::4])
    o_times =  params[1::4]
    heights = params[2::4]
    itaus_1 = np.exp(-1*logtau_1)
    itaus_2 = np.exp(-1*logtau_2)

    for o_time, height, itau_1, itau_2 in zip(o_times, heights, itaus_1, itaus_2):
        y_temp += pritish_double_exp_unlog(t1, o_time, height, itau_1, itau_2)
    return y_temp
# %%

with plt.style.context(['science','no-latex']):

    fig,axs = plt.subplots(3,1,figsize=[20,10],sharex=True,sharey=True)
    ax=axs[0]
    ax.plot(data.times,data.trace_zeroed,label="zero baseline trace")
    ax.plot(data.times,data.s1 * data.mult1,label="deconcolved")
    ax.plot(data.times,data.c1,label="denoised trace")
    curr_decon.plot.scatter('event_loc_time','heights',c='cn_mod5',ax=ax,colorbar=False,s=50,cmap="Set1",label='detected events')
    ax.axhline(data.MAD,label='1 Std',color='k',ls='dashed')
    # ax.axhline(0.0,label='1 Std',color='k',ls='dashed')
    ax.set_xlim(ll,ul)
    ax.set_ylim(-3*data.MAD,data.trace_zeroed[i_ll:i_ul].max() + 1*data.MAD)
    ax.legend()
    ax.set_ylabel('current (pA)')
    ax.set_xlabel('time (s)')




    ax=axs[1]
    ax.plot(data.times,data.trace_zeroed,label="zero baseline trace")
    ax.set_xlim(ll,ul)
    ax.set_ylim(-3*data.MAD,data.trace_zeroed[i_ll:i_ul].max() + 1*data.MAD)
    ax.legend()
    ax.set_ylabel('current (pA)')
    ax.set_xlabel('time (s)')

    for num,clus in curr_evoked_unselected.groupby('cluster_number'):
        x_init=[clus.baseline.iloc[0]]
        for t,h,t1,t2 in zip(clus.time,clus.height,clus.tau_1,clus.tau_2):
            x_init.extend([t,h,t1,t2])
        my_slice= slice(clus.tlim_min.iloc[0],clus.tlim_max.iloc[0])
        times = data.times[my_slice]
        temp_output = np.zeros_like(times)
        func_double_exp(x_init,times,temp_output)
        ax.plot(times,temp_output)

    curr_evoked_unselected.plot.scatter('time','height',s=50,label='fit_events',ax=ax,color='k')

    ax=axs[2]
    ax.plot(data.times,data.trace_zeroed,label="zero baseline trace")
    ax.set_xlim(ll,ul)
    ax.set_ylim(-3*data.MAD,data.trace_zeroed[i_ll:i_ul].max() + 1*data.MAD)

    ax.set_ylabel('current (pA)')
    ax.set_xlabel('time (s)')


    selected=set(curr_evoked_selected.ev_number)
    for num,clus in curr_evoked_unselected.groupby('cluster_number'):
        for t,h,t1,t2,ev in zip(clus.time,clus.height,clus.tau_1,clus.tau_2,clus.ev_number):
            x_init=[clus.baseline.iloc[0]]
            x_init.extend([t,h,t1,t2])

            i_ll_ev=np.searchsorted(data.times,t-2e-3)
            i_ul_ev=np.searchsorted(data.times,t+3*t1+3*t2+2e-3)

            my_slice= slice(i_ll_ev,i_ul_ev)
            times = data.times[my_slice]
            temp_output = np.zeros_like(times)
            func_double_exp(x_init,times,temp_output)
            ax.plot(times,temp_output,color= 'blue' if ev in selected else 'red' )
        # ax.hlines(clus.baseline.iloc[0],clus.tlim_min.iloc[0],clus.tlim_max.iloc[0])
        ax.hlines(clus.baseline.iloc[0],data.times[clus.tlim_min.iloc[0]],data.times[clus.tlim_max.iloc[0]],color='gray')

    ax.axhline(2.5*data.MAD,label='2.5 Std',color='k',ls='dashed')
    curr_evoked_unselected.plot.scatter('time','height',s=50,label='fit_events',ax=ax,color='k')
    ax.legend()


    plt.savefig('plots/trace_figure_v3_pretty.png',dpi=500)
    plt.savefig('plots/trace_figure_v3_pretty.svg')
    plt.savefig('plots/trace_figure_v3_pretty.eps', format='eps')


# %%
# plt.hlines(clus.baseline.iloc[0],data.times[clus.tlim_min],data.times[clus.tlim_max])


# %%
# %%

# %%
# for num,clus in curr_evoked_selected.groupby('cluster_number'):
#     x_init=[clus.baseline.iloc[0]]
#     for t,h,t1,t2 in zip(clus.time,clus.height,clus.tau_1,clus.tau_2):
#         x_init.extend([t,h,t1,t2])
#     my_slice= slice(clus.tlim_min.iloc[0],clus.tlim_max.iloc[0])
#     times = data.times[my_slice]
#     temp_output = np.zeros_like(times)
#     func_double_exp(x_init,times,temp_output)
#     # times=clus.time
#     # heights=clus.height
#     # tau_1 = clus.tau_1
#     # tau_2 = clus.tau_2
#     # baseline = 
#     # print(clus)
#     break
# %%

# %%
