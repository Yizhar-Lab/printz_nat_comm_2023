#%%
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# %%
csv_dir1 =Path("fit_csvs_short/")
csv_dir2 =Path("fit_csvs_skipped_split/")

#%%

df_list=[]
for csv_file in sorted(csv_dir1.glob("*/*.csv")):
    df_list.append(pd.read_csv(csv_file,index_col=0))
full_df1=pd.concat(df_list,copy=False,ignore_index=True)

#%%
df_list=[]
for csv_file in sorted(csv_dir2.glob("*/*.csv")):
    df_list.append(pd.read_csv(csv_file,index_col=0))
full_df2=pd.concat(df_list,copy=False,ignore_index=True)
# %%
# %%

# %%

# %%
def get_date(name):
    arr= name['filename'].split("_")
    return f"{arr[1]}-{arr[2]}-{arr[3]}"

# %%
def get_cell(name):
    arr= name['filename'].split("_")
    return int(arr[5][4:])

def get_rec(name):
    arr= name['filename'].split("_")
    return int(arr[4])

#%%
full_df1['date'] = pd.to_datetime(full_df1.apply(get_date,axis=1))
full_df1['cell'] = full_df1.apply(get_cell,axis=1)
full_df1['record'] = full_df1.apply(get_rec,axis=1)
# %%
full_df2['date'] = pd.to_datetime(full_df2.apply(get_date,axis=1))
full_df2['cell'] = full_df2.apply(get_cell,axis=1)
full_df2['record'] = full_df2.apply(get_rec,axis=1)

# %%
full_df = pd.concat([full_df1,full_df2],copy=True,ignore_index=True)
# %%
full_df_sorted = full_df.sort_values("ev_number",kind="mergesort",).sort_values("record",kind="mergesort").sort_values("date",kind="mergesort")
full_df_sorted.drop_duplicates(['filename','ev_number'],inplace=True)
full_df_sorted.reset_index(drop=True,inplace=True)

# %%
row_temp=full_df_sorted.sample()
# %%
def max_norm(tau_1,tau_2):
    tau_r = tau_1*tau_2/(tau_1-tau_2)
    N = ((tau_2/tau_1)**(tau_r/tau_1) -   (tau_2/tau_1)**(tau_r/tau_2)   )**-1
    peak_t = tau_r * np.log(tau_1/tau_2)
    return N,peak_t
# %%
resolution=51
tau_1_min,tau_1_max =6e-4,6e-2
tau_2_min,tau_2_max =1e-4,1e-2
tau_1_samples=np.logspace(np.log10(tau_1_min),np.log10(tau_1_max),resolution)
tau_2_samples=np.logspace(np.log10(tau_2_min),np.log10(tau_2_max),resolution)

x1,y1=np.meshgrid(tau_1_samples,tau_2_samples)
x,y=x1.flatten(),y1.flatten()
# %%
def v_of_t(tau_1,tau_2):
    t=np.linspace(0,0.1,10001)
    N,peak_t= max_norm(tau_1,tau_2)
    return t,N*(np.exp(-t/tau_1) - np.exp(-t/tau_2)),peak_t

# %%
# t,v=v_of_t(0.001344,0.000636)
# %%
def taus_to_percent_crossing(tau_1,tau_2,percent=10):
    p1=percent/100.0
    p2=1-percent/100.0
    # print(p1,p2)
    t,v,peak_t=v_of_t(tau_1,tau_2)


    t_ups=t[np.where(np.diff(np.sign(v-p2)))]
    t_downs = t[np.where(np.diff(np.sign(v-p1)))]
    return [*(t_ups-t_downs)*np.array([1,-1]),peak_t]
# %%
from scipy.interpolate import RectBivariateSpline,LSQBivariateSpline,NearestNDInterpolator,LinearNDInterpolator

#%%
resolution=101
tau_1_min,tau_1_max =6e-4,6e-2
tau_2_min,tau_2_max =1e-4,1e-2
tau_1_samples=np.logspace(np.log10(tau_1_min),np.log10(tau_1_max),resolution)
tau_2_samples=np.logspace(np.log10(tau_2_min),np.log10(tau_2_max),resolution)

x1,y1=np.meshgrid(tau_1_samples,tau_2_samples)
x,y=x1.flatten(),y1.flatten()


# %%

points=np.array([*zip(x,y)])
z =np.array([*map(lambda x: taus_to_percent_crossing(*x,10),points)])
interp = LinearNDInterpolator(list(zip(x, y)), np.array(z))
# interp = NearestNDInterpolator(list(zip(x, y)), np.array(z))


full_df_sorted_res=interp(np.array(full_df_sorted[['tau_1','tau_2']]))
full_df_sorted['t_up_10_90'] = full_df_sorted_res[:,0]
full_df_sorted['t_down_10_90'] = full_df_sorted_res[:,1]
full_df_sorted['t_peak_offset'] = full_df_sorted_res[:,2]


points=np.array([*zip(x,y)])
z =np.array([*map(lambda x: taus_to_percent_crossing(*x,20),points)])
interp = LinearNDInterpolator(list(zip(x, y)), np.array(z))
# interp = NearestNDInterpolator(list(zip(x, y)), np.array(z))

full_df_sorted_res=interp(np.array(full_df_sorted[['tau_1','tau_2']]))
full_df_sorted['t_up_20_80'] = full_df_sorted_res[:,0]
full_df_sorted['t_down_20_80'] = full_df_sorted_res[:,1]
full_df_sorted['t_peak_offset'] = full_df_sorted_res[:,2]


#  piece_res=piece[['tau_1','tau_2']].apply(interp,axis=1)
# %%
full_df_sorted.to_csv("all_combined_with_80_20_taus.csv")
# %%
