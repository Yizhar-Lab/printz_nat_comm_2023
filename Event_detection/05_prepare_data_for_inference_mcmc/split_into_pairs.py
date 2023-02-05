# %%
from collections import defaultdict
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from box import Box
import scipy
import scipy.io
import json


from interval import interval, inf, imath
from interval import fpu
import itertools


# %%
class MyInterval(interval):
    def complement(self) -> 'MyInterval':
        chain = itertools.chain(self, [[inf, None]])
        out = []
        prev = [None, -1*inf]
        for this in chain:
            if prev[1] != this[0]:
                out.append([prev[1], this[0]])
            prev = this

        return self.__class__(*out)

    def measure(self):
        return fpu.up(lambda: sum((c.sup - c.inf for c in self), 0))


# %%
def get_evoked_interval(tjs, margin):
    trig_ind = np.searchsorted(tjs['lightOnsetTimes'], tjs['triggerTimes'])

    evoked_interval = MyInterval()

    for pre_num, ind_beg in enumerate(trig_ind):
        if pre_num == len(trig_ind)-1:
            ind_end = len(tjs['lightOnsetTimes'])
        else:
            ind_end = trig_ind[pre_num+1]
    #     print(pre_num,(ind_beg,ind_end))
        light_times = np.array(tjs['lightOnsetTimes'][ind_beg:ind_end])
        before_light_times = light_times.min()-margin
        after_light_times = light_times.max()+margin
        evoked_interval = evoked_interval | MyInterval[before_light_times, after_light_times]
    return evoked_interval


# %%

def get_date(name):
    arr = name.split("_")
    return f"{arr[1]}-{arr[2]}-{arr[3]}"


def get_cell(name):
    arr = name.split("_")
    return int(arr[5][4:])


def get_rec(name):
    arr = name.split("_")
    return int(arr[4])


def get_uniq(name):
    return (get_date(name), get_cell(name), get_rec(name))


def get_df_events_in_time(tcs_good_sorted, tmin, tmax):
    inds = tcs_good_sorted.time.searchsorted([tmin, tmax])
    return tcs_good_sorted[inds[0]:inds[1]]


def get_events(pre_num, light_times, tcs_good_sorted, rep_num, evoked_len=0.09, clip_margin=0.2):

    rows = []
    intervals = []
    lt_min, lt_max = light_times.min(), light_times.max()
    # tcs_clipped=tcs_good_sorted.query("time > @lt_min- @clip_margin  and time < @lt_max + @clip_margin")
    tcs_clipped = get_df_events_in_time(tcs_good_sorted, lt_min-clip_margin, lt_max + clip_margin)
    for trial, light_time in enumerate(light_times):
        rep_df = tcs_clipped.query("time > @light_time  and time < @light_time + @evoked_len")
        rep_dict = rep_df.T.to_dict()
        for event in rep_dict:
            ev_dict = rep_dict[event]
            ev_dict['time_from_light'] = ev_dict["time"]-light_time
            ev_dict["pre_num"] = pre_num+1
            ev_dict["trial"] = trial
            ev_dict["rep_num"] = rep_num
            rows.append(ev_dict)
        intervaldict = {}
        intervaldict["date"] = date1
        intervaldict["date"] = date1
        intervaldict["cell"] = cell1
        intervaldict["rec"] = rec1
        intervaldict["pre_num"] = pre_num+1
        intervaldict["rep_num"] = rep_num
        intervaldict["tlen"] = evoked_len
        spont_events = [a for a, b in rep_df.time.iteritems()]
        intervaldict["events"] = spont_events
        intervaldict["is_evoked"] = True
        intervaldict["num_events"] = len(spont_events)
        intervals.append(intervaldict)

    return rows, intervals


def get_intervals(date1, cell1, rec1, pre_num, light_times, tcs_good_sorted, spont_interval, rep_num, evoked_len=0.09, margin=5.0):
    interval_min_len = 0.01
    lt_min, lt_max = light_times.min(), light_times.max()
    around_interval = MyInterval((lt_min - margin, lt_max + margin))
    around_interval = around_interval & spont_interval
    tcs_clipped = get_df_events_in_time(tcs_good_sorted, lt_min - margin, lt_max + margin)

    small_intervals = [MyInterval((ct, ct+evoked_len)) for ct in np.arange(lt_min-margin, lt_max+margin, evoked_len)]
    intervals = []
    for si in small_intervals:
        curr_intersect = around_interval & si
        if curr_intersect.measure() > interval_min_len:
            # print(si)
            intervaldict = {}
            intervaldict["date"] = date1
            intervaldict["date"] = date1
            intervaldict["cell"] = cell1
            intervaldict["rec"] = rec1
            intervaldict["pre_num"] = pre_num + 1
            intervaldict["rep_num"] = rep_num
            intervaldict["tlen"] = curr_intersect.measure()
            spont_events = [a for a, b in tcs_clipped.time.iteritems() if b in curr_intersect]
            intervaldict["events"] = spont_events
            intervaldict["is_evoked"] = False
            intervaldict["num_events"] = len(spont_events)
            intervals.append(intervaldict)
    return intervals


# %%
# load data
all_events = pd.read_csv("all_events_with_h_geq_2.5_MAD.csv", index_col=0)
lights_and_trigs = np.load("lights_and_trigs.npz", allow_pickle=True)

# %%
# remove the files with blockers.
uniq_lights_and_trigs = {get_uniq(k): v for k, v in lights_and_trigs.items()
                         if "Blocker" not in k}
files_uniq = set(uniq_lights_and_trigs.keys())

files_uniq_dict = defaultdict(list)
for date, cell, rec in sorted(files_uniq):
    files_uniq_dict[(date, cell)].append((date, cell, rec))


# %%
# loop over all the files and aggregate events and intervals
metalist = []
evokedlist = []
binslist = []
margin = 5.0  # in seconds how much to look around for spontaenous events (the total interval is double of this)
clip_margin = 0.2
evoked_len = 0.09
grouped = all_events.groupby(["date",	"cell",	"record"])
for date, cell in files_uniq_dict:
    print(date, cell)
    for (rep_num, (date1, cell1, rec1)) in enumerate(files_uniq_dict[(date, cell)]):
        rows = []
        ints = []
        print((date1, cell1, rec1))
        curr_dict = uniq_lights_and_trigs[(date1, cell1, rec1)].item()
        tcs_good_sorted = grouped.get_group((date1, cell1, rec1)).sort_values("time")
        rec_tmax = np.max([tcs_good_sorted.time.max(), curr_dict['lightOffsetTimes'].max()])

        recording_interval = MyInterval((0, rec_tmax+0.2))
        print(recording_interval)
        evoked_interval = get_evoked_interval(curr_dict, 0.1)
        spont_interval = recording_interval & evoked_interval.complement()

        begin_ind = np.searchsorted(curr_dict['lightOnsetTimes'], curr_dict['triggerTimes'])
        end_ind = np.array(list(begin_ind[1:])+[len(curr_dict['lightOnsetTimes'])])
        for pre_num, (ind_beg, ind_end) in enumerate(zip(begin_ind, end_ind)):
            light_times = np.array(curr_dict['lightOnsetTimes'][ind_beg:ind_end])
            events, evoked_intervals = get_events(pre_num, light_times, tcs_good_sorted, rep_num,
                                                  evoked_len=evoked_len, clip_margin=clip_margin)
            spont_intervals = get_intervals(
                date1, cell1, rec1, pre_num, light_times, tcs_good_sorted, spont_interval, rep_num,
                evoked_len=evoked_len, margin=margin)
            rows.extend(events)
            ints.extend(evoked_intervals)
            ints.extend(spont_intervals)

        evokedlist.append(pd.DataFrame(rows))
        binslist.append(pd.DataFrame(ints))

# %%

EVOKED_DF = pd.concat(evokedlist, ignore_index=True)
META_DF = pd.concat(binslist, ignore_index=True)

EVOKED_DF.sort_values(["date", "cell", "pre_num"], inplace=True)
META_DF.sort_values(["date", "cell", "pre_num"], inplace=True)

# %%
export_folder = "new_exports"

EVOKED_DF.to_csv(f"{export_folder}/EVOKED_DF.csv", float_format="%g")
META_DF.to_csv(f"{export_folder}/META_DF.csv", float_format="%g")
EVOKED_DF.to_json(f"{export_folder}/EVOKED_DF.json", orient='records', lines=True)
META_DF.to_json(f"{export_folder}/META_DF.json", orient='records', lines=True)
# %%

# %%
# load cells with PC
PC_DF = pd.read_json("CELLS_WITH_PC.json", orient='records', lines=True)
PC_DF.sort_values(["date", "cell", "pre_num"], inplace=True)
PC_DF.date = PC_DF.date.dt.strftime('%Y-%m-%d')
# %%
# get only the cells which have atleast one event in the evoked periods.
gp = EVOKED_DF.groupby(["date",	"cell",	"pre_num"])
pairs_with_events = set(sorted(gp.indices.keys()))
pairs_with_pc = set(sorted(PC_DF.groupby(["date",	"cell",	"pre_num"]).indices.keys()))
pairs_without_pc_with_events = pairs_with_events - pairs_with_pc
len(pairs_with_events), len(pairs_with_pc), len(pairs_without_pc_with_events)
# %%
# export parits
pairs_to_infer = pd.DataFrame(sorted(pairs_without_pc_with_events))
pairs_to_infer.columns = ["date",	"cell",	"pre_num"]
pairs_to_infer.to_csv(f"{export_folder}/pairs_to_infer.csv")

# %%
