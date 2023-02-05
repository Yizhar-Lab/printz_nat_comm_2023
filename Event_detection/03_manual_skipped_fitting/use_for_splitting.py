# %%
import numpy as np
import pandas as pd
from box import Box
import csv
from pathlib import Path,PureWindowsPath
import matplotlib.pyplot as plt
# %%
df=pd.read_excel("./Split times per file.xlsx",index_col=0)

# %%
def applyf(s):
    if s[0]=='?':
        return []
    else:
        return list(np.fromstring(s,sep=','))


#%%
df_filtered=df.applymap(str).applymap(applyf)
df_filtered.reset_index(inplace=True)

df_filtered.columns=['file_no','times']
df_filtered.set_index('file_no')
# %%


times=df_filtered['times']
# %%

for f_no, times in df_filtered.iterrows():
    if not times[1]:
        print(f_no, times[1])

    # break
# %%
# %%

# %%
skipped = Path("./skipped")
# %%
for file in skipped.glob("*.npz"):
    print(file)
    break
# %%
with np.load('file_list.npz') as data:
    file_list = data['arr_0']



#%%
fix_scalar = lambda x : x.item() if x.ndim==0 else  x
def read_npz(file_name):
    with np.load(file_name,allow_pickle=True) as data:
    # data= np.load(file_name,allow_pickle=True)
        data_dic={ k: fix_scalar(data[k]) for k in data}
    return Box(data_dic)
# %%

out_dir = Path("./skipped_split_new/")
if not out_dir.exists():
    out_dir.mkdir()
# out_dir.
#%%

did_not_split=[]

for f_no,file_wndws in enumerate(file_list):
    filepath = Path(PureWindowsPath(file_wndws).as_posix())
    print(filepath.stem)
    data = read_npz(filepath)
    # break
    t2ind = lambda t : np.searchsorted(data.times, t)+data.tl[0]
    
    split_inds=list(data.tl)
    for t_split in df_filtered.loc[f_no][1]:
        split_inds.append(t2ind(t_split))
    split_inds=sorted(split_inds)
    
    if len(split_inds)==2:
        #didn't split it
        did_not_split.append((f_no,filepath))


    for i,(s,e) in enumerate(zip(split_inds[:-1],split_inds[1:])):
        f_name="_".join(filepath.stem.split("_")[:-1])+f"_new{i:03d}"
        # print(f_name,s,e)
        data_new=Box()
        
        t0 = data.tl[0]
        t1 = data.tl[1]
        data_new.tl=[s,e]
        # mask=np.where()
        data_new.trace = data.trace[s-t0:e-t0]
        data_new.times = data.times[s-t0:e-t0]
        
        data_new.MAD=data.MAD
        data_new.seg =slice(s,e,None)
        
        s0 = s-t0 if s>t0 else 0
        e0 = e-t0-1 if e>t0-1 else t1-t0-1
        print(s0,e0)
        if s0>=e0:
            print("skipping")
            continue
        
        seg_peaks=np.where(
            (data.ev_times<data.times[e0])&
            (data.ev_times>data.times[s0])   # can't have minus 1 here, but get error check
            )
        print(f"did not skip : {len(seg_peaks[0])}")
        data_new.peaks = data.peaks[seg_peaks]
        data_new.locs = data.locs[seg_peaks]
        data_new.heights = data.heights[seg_peaks]
        data_new.ev_times = data.ev_times[seg_peaks]
        data_new.cn = data.cn
        data_new.subcluster = i

        np.savez(out_dir / f"{f_name}",**data_new)    
    # break

print(did_not_split)

    
# %%
# t0 = data.tl[0]
# plt.plot(data.times,data.trace)
# plt.scatter(data.ev_times,data.heights,c='r')
# plt.vlines(data.times[split_inds[1:-1]-t0],0,data.trace.max(),color='k')# %%
# %%
print(did_not_split)



# %%
for ind, filepath in did_not_split:
    if ind !=380:
        continue
    filepath_stem=filepath.stem
    data = read_npz(filepath)
    t2ind = lambda t : np.searchsorted(data.times, t)+data.tl[0]
    
    split_inds=list(data.tl)
    for t_split in df_filtered.loc[f_no][1]:
        split_inds.append(t2ind(t_split))
    split_inds=sorted(split_inds)
    


    # for i,(s,e) in enumerate(zip(split_inds[:-1],split_inds[1:])):
    #     f_name="_".join(filepath.stem.split("_")[:-1])+f"_new{i:03d}"
    #     # print(f_name,s,e)
    #     data_new=Box()
        
    #     t0 = data.tl[0]

    #     data_new.tl=[s,e]
    #     # mask=np.where()
    #     data_new.trace = data.trace[s-t0:e-t0]
    #     data_new.times = data.times[s-t0:e-t0]
        
    #     data_new.MAD=data.MAD
    #     data_new.seg =slice(s,e,None)
        
    #     seg_peaks=np.where(
    #         (data.ev_times<data.times[e-t0-1])&
    #         (data.ev_times>data.times[s-t0-1])
    #         )

    #     data_new.peaks = data.peaks[seg_peaks]
    #     data_new.locs = data.locs[seg_peaks]
    #     data_new.heights = data.heights[seg_peaks]
    #     data_new.ev_times = data.ev_times[seg_peaks]
    #     data_new.cn = data.cn
    #     data_new.subcluster = i

    # break
# %%
t0 = data.tl[0]
plt.plot(data.times,data.trace)
plt.scatter(data.ev_times,data.heights,c='r')
# plt.vlines(data.times[split_inds[1:-1]-t0],0,data.trace.max(),color='k')# %%

tmin,tmax = data.times[[0,-1] ]
a,b=0.,0.32
plt.xlim(tmin+a*(tmax-tmin),  tmin+b*(tmax-tmin) )
# %%

from scipy.signal import sosfiltfilt, butter,hilbert,find_peaks

#%%
sos = butter(4, 200, output='sos',fs=5000,btype='lowpass')
y = sosfiltfilt(sos, data.trace)
z= hilbert(y)

sos2 = butter(4, 50, output='sos',fs=5000,btype='lowpass')
zz = sosfiltfilt(sos2, np.abs(z))


# %%
sos2 = butter(4, [1,4], output='sos',fs=5000,btype='bandpass')
asdf = sosfiltfilt(sos2, z)

#%%
troughs=find_peaks(-zz,height=-2.0*data.MAD)[0]
print(len(troughs))
# troughs = troughs[zz[troughs]< np.median(zz[troughs])]
# print(len(troughs))
trough_times=data.times[troughs]
# %%
plt.plot(data.times,data.trace)
# plt.plot(data.times,y)
plt.scatter(data.ev_times,data.heights,c='r',zorder=101)
# plt.plot(data.times,np.abs(z),label='z')
# plt.plot(data.times,np.abs(zz),label='zz')
# plt.scatter(data.times[troughs],zz[troughs],c='k',zorder=100)
plt.scatter(data.ev_times,data.heights,c='r')

plt.legend()
# plt.plot(data.times,asdf)
tmin,tmax = data.times[[0,-1] ]
# a,b=0.2,0.25
a,b=0.,1.0
plt.xlim(tmin+a*(tmax-tmin),  tmin+b*(tmax-tmin) )

# %%

# %%
import heapq
# %%
heap = []
id=0
id+=1
heapq.heappush( heap,(-1*len(data.peaks),id,data ))
# %%
while heap[0][0]< -21:
    num,tid,item=heapq.heappop(heap)
    print(num,tid)
    t0 = item.tl[0]
    t1= item.tl[1]
    m1 = data.times[(t0+t1)//2 -t0]
    m = troughs[np.argmin(np.abs(trough_times -m1))]
    print(t0,t1,m1,m)
    #first peice 

    
    # s=t0
    # e=t0+m
    for s,e in zip([t0,t0+m],[t0+m,t1]):
        data_new=Box()    
        data_new.tl=[s,e]
        data_new.trace = item.trace[s-t0:e-t0]
        data_new.times = item.times[s-t0:e-t0]
        
        data_new.MAD=item.MAD
        data_new.seg =slice(s,e,None)

        s0 = s-t0 if s>t0 else 0
        e0 = e-t0-1 if e>t0-1 else t1-t0-1
        print(s0,e0)
        if s0>=e0:
            print("skipping")
            continue
        
        seg_peaks=np.where(
            (item.ev_times<item.times[e0])&
            (item.ev_times>item.times[s0])   # can't have minus 1 here, but get error check
            )



        seg_peaks=np.where(
            (item.ev_times<item.times[e-t0-1])&
            (item.ev_times>item.times[s-t0])
            )

        data_new.peaks = item.peaks[seg_peaks]
        data_new.locs = item.locs[seg_peaks]
        data_new.heights = item.heights[seg_peaks]
        data_new.ev_times = item.ev_times[seg_peaks]
        data_new.cn = item.cn

        id+=1
        print(f"s = {s} and e = {e}" )
        print(f"pushing {-1*len(data_new.peaks)} , {id}" ,)
        heapq.heappush(heap,(-1*len(data_new.peaks),id,data_new))
        # data_new.subcluster = i

    print("done")

    # break
# %%
list_from_heap = [v[2] for v in heap]
for i,item in enumerate(list_from_heap):
    item.subcluster = i
# %%
for item in list_from_heap:
    i=item.subcluster
    f_name="_".join(filepath_stem.split("_")[:-1])+f"_new{i:03d}"
    np.savez(out_dir / f"{f_name}",**item)    


# %%
