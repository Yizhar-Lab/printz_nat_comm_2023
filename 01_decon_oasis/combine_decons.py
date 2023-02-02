#%%
import numpy as np
import pandas as pd


from pathlib import Path
# %%
decon_results = Path("./res/")
# %%
dfs={}
for file in decon_results.glob("decon*.csv"):
    dfs[file.stem] = pd.read_csv(file,index_col=0)
    # break
    # dfs.append(pd.
# %%
all_decon_df = pd.concat(dfs)
# %%
all_decon_df.index.names = ["file_name","event_no_in_file"]
with_filnames=all_decon_df.reset_index()
# %%

groups_sizes = with_filnames.groupby(['file_name','cluster_num']).size().reset_index(name='counts')
# %%
long_groups = groups_sizes.query("counts>20")
short_groups = groups_sizes.query("counts<=20")

#%%
long_set= set(zip(*map(long_groups.get, ['file_name', 'cluster_num'])))

# %%
# jus_long = [  df for (tup,df) in with_filnames.groupby(['file_name','cluster_num'])
#     if tup in long_set]

# %%
long_dfs=[]
short_dfs=[]
for (tup,df) in with_filnames.groupby(['file_name','cluster_num']):
    if tup in long_set:
        long_dfs.append(df)
    else:
        short_dfs.append(df)
    # break
# %%
just_long = pd.concat(long_dfs,ignore_index=True,copy=False)
just_short = pd.concat(short_dfs,ignore_index=True,copy=False)

# %%
just_long.to_csv("long_segments.csv")
just_short.to_csv("short_segments.csv")
