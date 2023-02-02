# %%
import numpy as np
import pandas as pd
import logging
from box import Box
import matplotlib.pyplot as plt
import pathlib









import argparse
parser = argparse.ArgumentParser()

parser.add_argument("offset", type=int,
                    help="processing_offset_filenum")
args = parser.parse_args()
print(f"processing offset :  {args.offset}")

offset = args.offset





# use step=1000 to split data into 1000 batches on a cluster
#       in that case offset is a number between 0 to step-1
# use step=1 to run the fitting serially on a PC 
#       in that case offset is 0 

# step=1000 
step=1










# %%




# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument("filename", type=str,
#                     help="process_filenum")
# args = parser.parse_args()
# print(f"plotting file {args.filename}")
# f_name = args.filename

# %%

basedir = pathlib.Path(".")
results_dir=basedir / f"Results031_fit_detected/job_{offset}/"
results_dir.mkdir(parents=True, exist_ok=True)

skipped_dir = results_dir / "skipped/"
skipped_dir.mkdir(parents=True, exist_ok=True)



logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=results_dir/f'fit.log',
    # filename=results_dir/f'fit_{f_num:04d}.log',
    level=logging.INFO
)

from just_fit_funcs import *
# %%




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
# fit_params.method           = "TNC"  
fit_params.maxiter          = 300
fit_params.maxfev           = 3000
fit_params.xtol             = 1e-9
fit_params.ftol             = 1e-9
fit_params.offset_fact      = 2
fit_params.maxiter_DA       = 5000
fit_params.maxfev_DA        = 50000


# %%

#fit_and_parse([curr,fit_params])
# %%

names=np.genfromtxt(basedir / 'names_for_fit.txt',dtype='str')
arr=[]
for file in names[offset::step]:
    file_p =pathlib.Path(file)
    curr = read_npz(file_p)
    f_name = file_p.stem
    if len(curr.peaks) > 20:
        logging.info(f'skipped {f_name}')
        print(f'skipped {f_name}')
        np.savez(skipped_dir / f"{file_p.name}",**curr)
        continue
    # f_name = "_".join(file_p.stem.split("_")[:-2])
    # f_name = "_".join(file_p.stem.split("_")[:6])
    logging.info(f'starting {f_name}')
    print(f_name)
    arr.append(fit_and_parse(curr,fit_params,f_name))
    logging.info(f'finished {f_name}')
results_arr = [item for sublist in arr for item in sublist]

fit_df = pd.DataFrame(results_arr)

fit_file=results_dir / "fit.csv"

fit_df.to_csv(fit_file)
