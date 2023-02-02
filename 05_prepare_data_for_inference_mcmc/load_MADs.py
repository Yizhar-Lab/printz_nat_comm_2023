# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from box import Box
# %%
all_events = pd.read_csv("all_combined_with_80_20_taus.csv",index_col=0)


#%%
fix_scalar = lambda x : x.item() if x.ndim==0 else  x

def read_npz_MAD(file_name):
    data= np.load(file_name,allow_pickle=True, mmap_mode='r')
    # data_dic={ k: fix_scalar(data[k]) for k in data}
    # return Box(data_dic)
    return fix_scalar(data['MAD'])


def get_date(name):
    arr= name.split("_")
    return f"{arr[1]}-{arr[2]}-{arr[3]}"

def get_cell(name):
    arr= name.split("_")
    return int(arr[5][4:])

def get_rec(name):
    arr= name.split("_")
    return int(arr[4])

# %%
MAD_dict={}
for file in Path("res").glob("*.npz"):
    f_name="_".join( file.stem.split("_")[:-2])
    print(get_date(file.stem))
    curr_MAD = read_npz_MAD(file)
    MAD_dict[f_name]=curr_MAD
    # break
MAD_dict
# %%
def get_mad_from_name( df) :
    for key in MAD_dict:
        if df.iloc[0].startswith(key):
            return MAD_dict[key]

all_events['MAD']= all_events.groupby(['date','cell','record']).filename.transform(get_mad_from_name)
# %%
all_events.to_csv("all_events_with_MAD.csv")
# %%
all_events.query("height > 2.5*MAD").to_csv("all_events_with_h_geq_2.5_MAD.csv")
all_events.query("height > 3.0*MAD").to_csv("all_events_with_h_geq_3.0_MAD.csv")
# %%
