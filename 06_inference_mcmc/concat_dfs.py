#%%
from os import path
from numpy import save
import pandas as pd
# %%
from pathlib import Path
# %%
csvs=Path("NEWResults_inference_57/csvs/")
# %%


df_list=[]
for csv  in sorted(csvs.glob("sum*")):
    print(csv)
    curr_csv=pd.read_csv(csv,header=[0,1,2],index_col=[0,1,2])
    df_list.append(curr_csv)
summary_concat_df=pd.concat(df_list,copy=True)
summary_concat_df.sort_index(inplace=True)

# %%

df_list=[]
for csv  in sorted(csvs.glob("loo*")):
    print(csv)
    curr_csv=pd.read_csv(csv,header=[0,1,2,3],index_col=[0,1,2])
    # break
    df_list.append(curr_csv)
loo_waic_concat_df=pd.concat(df_list,copy=True)
loo_waic_concat_df.sort_index(inplace=True)

#%%
df_list=[]
for csv  in sorted(csvs.glob("compare*")):
    print(csv)
    curr_csv=pd.read_csv(csv,header=[0,1,2],index_col=[0,1,2])
    # break
    df_list.append(curr_csv)
compare_concat_df=pd.concat(df_list,copy=True)
compare_concat_df.sort_index(inplace=True)
# %%
save_path=Path("concat_dataframes")
save_path.mkdir(exist_ok=True)
#%%
compare_concat_df.to_csv(save_path / "compare.csv")
summary_concat_df.to_csv(save_path / "summary.csv")
loo_waic_concat_df.to_csv(save_path / "loo_waic.csv")
# %%
