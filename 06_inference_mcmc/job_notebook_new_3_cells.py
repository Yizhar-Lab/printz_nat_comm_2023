import logging  # import first to make sure logging works
from pathlib import Path
import os

# basedir = Path("MCMC_inference_connected")
basedir=Path(".")



results_dir=basedir /"NEWResults_inference_58"
results_dir.mkdir(parents=True, exist_ok=True)

logs=results_dir/  "logs"
logs.mkdir(parents=True, exist_ok=True)


jobs=results_dir/  "jobs"
jobs.mkdir(parents=True, exist_ok=True)


steps=1


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("offset", type=int, help=f"Offset from 1:{steps} number of jobs. Total number of jobs can be changed in the py file.")
args = parser.parse_args()
print(f"processing offset : {args.offset}")
offset = args.offset-1


compile_dir =  Path(f"/var/tmp/pritish_theano_compiles_new/")
compile_dir.mkdir(parents=True, exist_ok=True)


os.environ['THEANO_FLAGS']=f"base_compiledir={compile_dir.absolute()}"


csvs=results_dir/  "csvs"
csvs.mkdir(parents=True, exist_ok=True)

curr_job  = jobs / f"job_{offset:04d}_started"
curr_job.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=logs/f'offset_{offset:04d}.log',
    level=logging.INFO
)



# %%
import pymc3 as pm
import pandas as pd
import arviz as az
import xarray as xr
import time

# %%
from inference_class import MixtureModelInference
# %%
data_folder = Path("new_exports")
# %%
logging.info(f"\n=========\n loading data \n=========\n")
evoked = pd.read_csv(data_folder / "EVOKED_DF.csv", index_col=0)
intervals = pd.read_csv(data_folder / "META_DF.csv", index_col=0)

# %%
pairs_to_infer = pd.read_csv(data_folder / "pairs_to_infer_3_cells.csv", index_col=0)
# %%
pair_name_columns = ['date', 'cell', 'pre_num']
intervals_groups = intervals.groupby(pair_name_columns)
evoked_groups = evoked.groupby(pair_name_columns)



# %%

priors = {}
priors["evoked_window"] = 90e-3
priors["rate_mu"] = 4.83
priors["rate_sigma"] = 3.79
priors["evoked_per_trial_mu"] = 1.0
priors["evoked_per_trial_sigma"] = 0.35
priors["bump_center_mu"] = 12.7  # in ms
priors["bump_width_mu"] = 6.33  # in ms
priors["bump_center_sigma"] = 6.33  # in ms
# priors["bump_width_sigma"] = 0.3  # in ms
priors["bump_width_sigma"] = 1.5  # in ms
priors["beta_alpha"] = 2
priors["beta_beta"] = 2

# %%
sample_params = {}
sample_params["draws"] = 2000
sample_params["tune"] = 5000
sample_params["init"] = 'jitter+adapt_full'
sample_params["return_inferencedata"] = True
sample_params["target_accept"] = 0.95
sample_params["cores"] = 1
sample_params["progressbar"] = False
sample_params["chains"] = 4



# %%

this_job = pairs_to_infer.iloc[offset::steps]

summary_dict={}
compare_dict={}
loo_waic_dict={}
for index,curr_name in this_job.iterrows():
    # print(curr_name)
    curr_name_tuple = (*curr_name,)
    logging.info(f"\n=========\n Processing{curr_name_tuple} \n=========\n")
    #extract data
    
    data = {}
    data["inter_rate"] = intervals_groups.get_group(curr_name_tuple)
    data["event_time"] = evoked_groups.get_group(curr_name_tuple)

    #run inference
    curr_inference = MixtureModelInference(data=data, priors=priors, sample_params=sample_params)
    curr_inference.fit()
    curr_inference.make_all_dfs()

    #save_data_in_dicts
    summary_dict[curr_name_tuple]=curr_inference.summary_df
    compare_dict[curr_name_tuple]=curr_inference.compare_df
    loo_waic_dict[curr_name_tuple]=curr_inference.loo_waic_df

    logging.info(f"\n=========\n Finished{curr_name_tuple} \n=========\n")


logging.info(f"\n=========\n Started saving csvs \n=========\n")
summary_df_job=pd.DataFrame.from_dict(summary_dict,orient="index")
compare_df_job=pd.DataFrame.from_dict(compare_dict,orient="index")
loo_waic_df_job=pd.DataFrame.from_dict(loo_waic_dict,orient="index")

summary_df_job.index.rename(pair_name_columns, inplace=True)
compare_df_job.index.rename(pair_name_columns, inplace=True)
loo_waic_df_job.index.rename(pair_name_columns, inplace=True)

summary_df_job.to_csv(csvs / f"summary_{offset:04d}.csv")
compare_df_job.to_csv(csvs / f"compare_{offset:04d}.csv")
loo_waic_df_job.to_csv(csvs / f"loo_waic_{offset:04d}.csv")
logging.info(f"\n=========\n Finished saving csvs \n=========\n")

curr_job.rmdir()
