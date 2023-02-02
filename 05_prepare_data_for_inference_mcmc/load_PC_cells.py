# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from pathlib import Path
from box import Box
# %%
all_events = pd.read_csv("all_combined_with_80_20_taus.csv",index_col=0)


#%%
fix_scalar = lambda x : x.item() if x.ndim==0 else  x

def read_npz_MAD(file_name):
    data= np.load(file_name,allow_pickle=True, mmap_mode='r')
    data_dic={ k: fix_scalar(data[k]) for k in data}
    return Box(data_dic)
    # return data
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

#%%


def readMatlabFile(matfile, params):
    '''
    Read the data of the full recording trace, not parsed:
    '''
    # logging.info(f"reading {matfile}")
    # data_signal = box.Box()
    data = Box(scipy.io.loadmat(matfile, squeeze_me=True))
    # data_signal.trace_raw = -data.trace_data

    # data_signal.locations = data.trueEventIndsOrigin

    # data_signal.f_sample = 1000*data.rate

    # n=len(data_signal.trace_raw)
    # step=1/data_signal.f_sample
    # data_signal.t = np.arange(0,n*step,step)  # in sec
    # data_signal.file_name = matfile.stem
    # logging.info(f"finished reading {matfile}")
    return data


# %%
PC_dict={}
# num=0
for file in sorted(Path("../DATA/").glob("*.mat")):
    print(file)
    f_name="_".join( file.stem.split("_")[1:])
    print(get_date(file.stem))
    print(get_rec(file.stem))
    print(get_cell(file.stem))
    # curr_MAD = read_npz_MAD(file)
    data = readMatlabFile(file,())
    if "cellsWithPC" in data.keys():
        if type(data['cellsWithPC'])==int:
            data['cellsWithPC'] = [data['cellsWithPC']]
        PC_dict[f_name]=sorted(data['cellsWithPC'])
    else:
        PC_dict[f_name]=[]
    print(PC_dict[f_name])
    # print(data.keys())
    # if num>10:
    #     break
    # num+=1
    # break
#%%
new_PC_dict=[]
for i,file in enumerate(PC_dict):
    print(file)
    (date,cell,rec) = get_date("FILE_"+file),get_cell("FILE_"+file),get_rec("FILE_"+file)
    for prenum in PC_dict[file]:
        new_PC_dict.append( {
            "date" : date,
            "cell" : cell,
            "rec" : rec,
            "pre_num" : prenum
        } )


PC_DF=pd.DataFrame(new_PC_dict)
    # print(date,cell,rec)
    # break
# MAD_dict
#%%
#PC_DF["NUM_PCS"]=PC_DF.PC_cells.apply(len)
#%%
PC_DF.to_csv("CELLS_WITH_PC.csv")
PC_DF.to_json("CELLS_WITH_PC.json",orient='records',lines=True)

#%%
print("number of pairs with PC is")
print(len(PC_DF.drop_duplicates(["cell","date","pre_num"])))
# %%
# def get_mad_from_name( df) :
#     for key in MAD_dict:
#         if df.iloc[0].startswith(key):
#             return MAD_dict[key]

# all_events['MAD']= all_events.groupby(['date','cell','record']).filename.transform(get_mad_from_name)
# # %%
# all_events.to_csv("all_events_with_MAD.csv")
# # %%
# all_events.query("height > 2.5*MAD").to_csv("all_events_with_h_geq_2.5_MAD.csv")
# all_events.query("height > 3.0*MAD").to_csv("all_events_with_h_geq_3.0_MAD.csv")
# %%
