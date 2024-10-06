# isort: off
import sys

sys.path.append("common")
# isort: on
import argparse
import json
import os
import sys
from tqdm import tqdm
import pickle
from binary_prog import BinaryProgram, Function
from fuzzywuzzy import fuzz
import name_utils


def get_fuzzy_ratio(name1, name2):
    return fuzz.ratio(name1, name2)




parser = argparse.ArgumentParser(description='Deduplicate dataset')
parser.add_argument('--fin', type=str, default='', help='ds file list')
parser.add_argument('--train-out-path', type=str, default='', help='train dataset path')
parser.add_argument('--test-out-path', type=str, default='', help='test dataset path')
parser.add_argument('--test-overlap-out-path', type=str, default='', help='test overlap output path')
parser.add_argument('--dedup-ratio', type=float, default=0.7, help='deduplication ratio')
args = parser.parse_args()

data_files = open(args.fin).readlines()

bins = []
for f in tqdm(data_files):
    f = f.strip()
    if not os.path.exists(f):
        continue
    with open(f, 'rb') as fin:
        bins.extend(pickle.load(fin))


# use function name to pre-filter
        
pre_filtered_bins = []
skipped_bins = []
seen_func_names = set()
for bin_prog in tqdm(bins):
    current_func_names = []
    not_seen_func_names = []
    current_seen_names = []
    for stripped_name, func in bin_prog.stripped_name2func.items():
        if stripped_name not in func.func_id_maps:
            func_name = stripped_name
        else:
            func_name = func.func_id_maps[stripped_name]
        if func_name == 'main':
            continue
        current_func_names.append(func_name)
        if func_name not in seen_func_names:
            not_seen_func_names.append(func_name)
            seen_func_names.add(func_name)
        else:
            current_seen_names.append(func_name)
    if len(not_seen_func_names) > len(current_func_names) * args.dedup_ratio:
        pre_filtered_bins.append(bin_prog)
    else:
        skipped_bins.append((bin_prog, not_seen_func_names, current_seen_names, current_func_names))
    

print("Original bins: %d, after filtering: %d, kept ratio: %.2f" % (len(bins), len(pre_filtered_bins), len(pre_filtered_bins) / len(bins)))


bins_sorted = sorted(pre_filtered_bins, key=lambda x: x.prog_name, reverse=True)

# shuffle with seed 42
import numpy as np
np.random.seed(42)
bins_shuffled = np.random.permutation(bins_sorted)

train_bins = list(bins_shuffled[:int(len(bins_shuffled) * 0.9)])
test_bins = list(bins_shuffled[int(len(bins_shuffled) * 0.9):])

def norm_func_body(func):
    new_body = func.body
    for k, v in func.func_id_maps.items():
        new_body = name_utils.replace_variable_names(new_body, k, v)
    for k, v in func.var_id_maps.items():
        new_body = name_utils.replace_variable_names(new_body, k, v)
    return new_body

for bin_prog in tqdm(train_bins, desc='Normalizing train dataset'):
    for stripped_name, func in bin_prog.stripped_name2func.items():
        func.norm_body = norm_func_body(func)

for bin_prog in tqdm(test_bins, desc='Normalizing test dataset'):
    for stripped_name, func in bin_prog.stripped_name2func.items():
        func.norm_body = norm_func_body(func)

print("Train bins: %d, Test bins: %d" % (len(train_bins), len(test_bins)))

def func2name_list(func):
    my_name = func.func_name
    names = list(func.var_id_maps.keys())
    if my_name in func.func_id_maps:
        names.append(func.func_id_maps[my_name])
    sorted_names = sorted(names)
    return '#'.join(sorted_names)

train_prog_func2entry = {}
train_prog_func2name_list = {}
train_name_list2funcs = {}
for train_bin in tqdm(train_bins):
    for stripped_name, func in train_bin.stripped_name2func.items():                        
        train_prog_func2entry[(train_bin.prog_name, stripped_name)] = func        
        name_list = func2name_list(func)
        train_prog_func2name_list[(train_bin.prog_name, stripped_name)] = name_list
        if name_list not in train_name_list2funcs:
            train_name_list2funcs[name_list] = []
        train_name_list2funcs[name_list].append((train_bin.prog_name, stripped_name))
        

def get_test_overlaps(start_idx, end_idx):
    test_overlaps = []
    for test_bin in tqdm(test_bins[start_idx:end_idx]):
        for stripped_name, func in test_bin.stripped_name2func.items():                        
            name_list = func2name_list(func)
            might_overlapped = []
            if name_list in train_name_list2funcs:
                for train_func in train_name_list2funcs[name_list]:
                    might_overlapped.append((train_func, 100, get_fuzzy_ratio(func.norm_body, train_prog_func2entry[train_func].norm_body)))
            for train_name, train_funcs in train_name_list2funcs.items():
                name_ratio = get_fuzzy_ratio(name_list, train_name)
                if name_ratio > 90:
                    for train_func in train_funcs:
                        might_overlapped.append((train_func, name_ratio, get_fuzzy_ratio(func.norm_body, train_prog_func2entry[train_func].norm_body)))

            test_overlaps.append(((test_bin.prog_name, stripped_name), might_overlapped))
    return test_overlaps

all_lens = len(test_bins)
NUM_WORKERS = 32
step = all_lens // NUM_WORKERS
intervals = [(i * step, (i + 1) * step) for i in range(NUM_WORKERS)]
intervals[-1] = (intervals[-1][0], all_lens)

import multiprocessing
from multiprocessing import Pool

with Pool(NUM_WORKERS) as p:
    res = p.starmap(get_test_overlaps, intervals)

test_overlaps = []
for r in res:
    test_overlaps.extend(r)

new_ds_overlap = open(args.test_overlap_out_path, 'w')
json.dump(test_overlaps, new_ds_overlap)
new_ds_overlap.close()

# output train and test dataset
pickle.dump(train_bins, open(args.train_out_path, 'wb'))
pickle.dump(test_bins, open(args.test_out_path, 'wb'))
print("Output train dataset to %s, test dataset to %s" % (args.train_out_path, args.test_out_path))

