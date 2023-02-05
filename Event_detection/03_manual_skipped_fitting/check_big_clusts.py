#%%
# %%
import numpy as np
import pandas as pd
import logging
from box import Box
import matplotlib.pyplot as plt
import pathlib


#%%

fix_scalar = lambda x : x.item() if x.ndim==0 else  x



def read_npz(file_name):
    data= np.load(file_name,allow_pickle=True)
    data_dic={ k: fix_scalar(data[k]) for k in data}
    return Box(data_dic)


out_dir = pathlib.Path("./skipped_split/")

#%%

file=pathlib.Path("Results030_fit_detected/job_0/skipped/File_2017_11_23_0011_Cell3_cnct_segment_0004_2.npz")

data = read_npz(file)

# %%
t2ind = lambda t : np.searchsorted(data.times, t)+data.tl[0]

split_inds=list(data.tl)

split_inds.append(t2ind(1.85))

split_inds=sorted(split_inds)
# %%
plt.plot(data.times,data.trace)
plt.scatter(data.ev_times,data.heights,c='r')
plt.vlines(splits,0,data.trace.max(),color='k')# %%



#%%
for i,(s,e) in enumerate(zip(split_inds[:-1],split_inds[1:])):
    f_name="_".join(file.stem.split("_")[:-1])+f"new{i:03}"
    print(fname,s,e)
    data_new=Box()
    
    t0 = data.tl[0]

    data_new.tl=[s,e]
    # mask=np.where()
    data_new.trace = data.trace[s-t0:e-t0]
    data_new.times = data.times[s-t0:e-t0]
    
    data_new.MAD=data.MAD
    data_new.seg =slice(s,e,None)
    
    seg_peaks=np.where(
        (data.ev_times<data.times[e-t0-1])&
        (data.ev_times>data.times[s-t0])
        )

    data_new.peaks = data.peaks[seg_peaks]
    data_new.locs = data.locs[seg_peaks]
    data_new.heights = data.heights[seg_peaks]
    data_new.ev_times = data.ev_times[seg_peaks]
    data_new.cn = data.cn
    data_new.subcluster = i

    np.savez(out_dir / f"{f_name}_{data_new.subcluster:03d}",**data_new)    
    
    # break
# newfile

