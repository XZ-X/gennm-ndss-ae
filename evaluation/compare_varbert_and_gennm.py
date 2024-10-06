# isort: off
import sys
sys.path.append("common")
# isort: on

import sys
import json
import pickle
from tqdm import tqdm
from binary_prog import BinaryProgram, Function

# force reimport name_utils
# import importlib
# importlib.reload(sys.modules["name_utils"])
from name_utils import try_demangle
import numpy as np
import os
import argparse
import datasets
import eval_utils
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--test_bins", type=str, default="", help="path to gennm test bins")
parser.add_argument("--ds_origin", type=str, default="", help="path to dataset origin map")
parser.add_argument("--ds_overlap", type=str, default="", help="path to dataset no overlap")
parser.add_argument("--ds_overlap_varbert_format", type=str, default="", help="path to dataset no overlap in varbert format")
parser.add_argument("--training", type=str, default="", help="path to training dataset")
parser.add_argument("--ret_evaluated", type=str, default="", help="path to gennm ret evaluated")
parser.add_argument("--varbert_evaluated", type=str, default="", help="path to varbert ret evaluated")
parser.add_argument("--optional_another", type=str, default="", help="path to another ret evaluated for name set")
parser.add_argument("--is_dirty", action="store_true", help="whether the dataset is in DIRTY format")
parser.add_argument("--func_prefix", type=str, default="sub_", help="prefix of function name")
args = parser.parse_args()


IS_DIRTY = args.is_dirty
FUNC_PREFIX = args.func_prefix

train_ds = datasets.load_dataset(args.training)
train_entries = {}
for entry in tqdm(train_ds["train"], desc="build dict from train ds"):
    train_entries[(entry["prog_name"], entry["strip_func_name"])] = entry


##########
# about projects
##########

dataset_origin_map = {}
for line in open(args.ds_origin, "r"):
    entry = json.loads(line)
    dataset_origin_map[entry["binary_name"]] = entry

non_alpha = re.compile(r"[^a-zA-Z]")
normalized_proj_name = {}
for entry in dataset_origin_map.values():
    proj_name = entry["project_name"]
    if proj_name in normalized_proj_name:
        continue
    name_fields = re.split(r"[^a-zA-Z]", proj_name)
    normed_names = []
    for nf in name_fields:
        if len(nf) > 1:
            normed_names.append(nf.lower())
    normalized_proj_name[proj_name] = normed_names

train_ds_projects = set()
train_projects_normed = {}
for entry in train_entries.values():
    prog_name = entry["prog_name"].split("_")[0]
    origin = dataset_origin_map[prog_name]
    train_ds_projects.add(origin["project_name"])
    category = origin.get("category", "unknown")
    normed_fields = normalized_proj_name[origin["project_name"]]
    for nf in normed_fields:
        if nf not in train_projects_normed:
            train_projects_normed[nf] = set()
        train_projects_normed[nf].add(category)


def proj_has_overlap(proj_entry):
    my_proj_name = proj_entry["project_name"]
    my_category = proj_entry.get("category", "unknown")
    if IS_DIRTY:
        return my_proj_name in train_ds_projects
    if my_proj_name in train_ds_projects:
        return True
    for nf in normalized_proj_name[my_proj_name]:
        if nf in train_projects_normed:
            return my_category in train_projects_normed[nf]
    return False



#########
# about overlap
#########

no_overlap_set = set()
no_overlap = json.load(open(args.ds_overlap, "r"))
for prog, func in no_overlap:
    no_overlap_set.add((prog, func))

no_overlap_varbert_format = json.load(open(args.ds_overlap_varbert_format, "r"))
no_overlap_varbert_format_set = set()
for prog, func in no_overlap_varbert_format:
    # XXX: hack: handle dirty dataset
    if "jsonl" in prog:
        prog = prog.split("_")[0]
    if FUNC_PREFIX in func:
        func_addr = int(func.split("_")[1], 16)
        func_addr_str = f"{func_addr:016X}"
        func = func_addr_str
    no_overlap_varbert_format_set.add((prog, func))

##########################
# load data
##########################


gennm_ret_evaluated = [json.loads(l) for l in open(args.ret_evaluated, "r")]
varbert_ret_evaluated = [json.loads(l) for l in open(args.varbert_evaluated, "r")]
if args.optional_another != "":
    optional_another = [json.loads(l) for l in open(args.optional_another, "r")]
else:
    optional_another = gennm_ret_evaluated


##########################
# load results
##########################

optional_another_name_set = set()
for entry in tqdm(optional_another):
    prog = entry["prog_name"].split("_")[0]
    func = entry["func_name"]
    if not func.startswith(FUNC_PREFIX):
        continue
    func_addr = int(func.split("_")[1], 16)
    if 'sub' in FUNC_PREFIX:
        func_addr_str = f"{func_addr:016X}"
    else:
        func_addr_str = f"{func_addr:08x}"    
    var_name = entry["varname"]
    optional_another_name_set.add((prog, func_addr_str, var_name))

gennm_evaluated_by_prog_func_var_varbert_format = {}
for entry in tqdm(gennm_ret_evaluated):
    prog = entry["prog_name"].split("_")[0]
    func = entry["func_name"]
    if not func.startswith(FUNC_PREFIX):
        continue
    func_addr = int(func.split("_")[1], 16)
    if 'sub' in FUNC_PREFIX:
        func_addr_str = f"{func_addr:016X}"
    else:
        func_addr_str = f"{func_addr:08x}"    
    var_name = entry["varname"]
    gennm_evaluated_by_prog_func_var_varbert_format[(prog, func_addr_str, var_name)] = (
        entry
    )

varbert_evaluated_by_prog_func_var = {}
for entry in tqdm(varbert_ret_evaluated, desc="organize varbert evaluated ret"):
    prog = entry["prog_name"].split("_")[0]
    func = entry["func_name"]
    if func.startswith(FUNC_PREFIX):
        func_addr = int(func.split("_")[1], 16)
        if 'sub' in FUNC_PREFIX:
            func_addr_str = f"{func_addr:016X}"
        else:
            func_addr_str = f"{func_addr:016x}"
    else:
        func_addr_str = func
    var_name = entry["varname"]
    varbert_evaluated_by_prog_func_var[(prog, func_addr_str, var_name)] = entry

overlapped_prog_funcs_var = set(
    gennm_evaluated_by_prog_func_var_varbert_format.keys()
) & set(varbert_evaluated_by_prog_func_var.keys()) & optional_another_name_set

gennm_overlapped_entries = []
for prog_func_var in overlapped_prog_funcs_var:
    gennm_entry = gennm_evaluated_by_prog_func_var_varbert_format[prog_func_var]
    prog, func, var = prog_func_var
    if (prog, func) not in no_overlap_varbert_format_set:
        overlapped = True
    else:
        overlapped = False
    
    proj_entry = dataset_origin_map[prog]
    proj_in_train = proj_has_overlap(proj_entry)
    proj_category = proj_entry.get("category", "unknown")

    gennm_overlapped_entries.append(
        {
            "prog_name": prog,
            "func_name": func,
            "varname": var,
            "gt_varname": gennm_entry["gt_varname"],
            "pred_name": gennm_entry["pred_name"],
            "precision": gennm_entry["precision"],
            "recall": gennm_entry["recall"],
            "overlapped": overlapped,
            "proj_in_train": proj_in_train,
            "proj_category": proj_category,
            "proj_name": proj_entry["project_name"],
        }
    )

varbert_overlapped_entries = []
for prog_func_var in overlapped_prog_funcs_var:
    varbert_entry = varbert_evaluated_by_prog_func_var[prog_func_var]
    prog, func, var = prog_func_var
    if (prog, func) not in no_overlap_varbert_format_set:
        overlapped = True
    else:
        overlapped = False

    proj_entry = dataset_origin_map[prog]
    proj_in_train = proj_has_overlap(proj_entry)
    proj_category = proj_entry.get("category", "unknown")

    varbert_overlapped_entries.append(
        {
            "prog_name": prog,
            "func_name": func,
            "varname": var,
            "pred_name": varbert_entry["pred_name"],
            "precision": varbert_entry["precision"],
            "recall": varbert_entry["recall"],
        }
    )
print("There are {} overlapped entries".format(len(overlapped_prog_funcs_var)))

df_gennm = pd.DataFrame(gennm_overlapped_entries)
df_varbert = pd.DataFrame(varbert_overlapped_entries)

df_gennm = df_gennm.rename(
    columns={
        "pred_name": "gennm_pred_name",
        "precision": "gennm_precision",
        "recall": "gennm_recall",
    }
)

# rename pred_name to varbert_pred_name
df_varbert = df_varbert.rename(
    columns={
        "pred_name": "varbert_pred_name",
        "precision": "varbert_precision",
        "recall": "varbert_recall",
    }
)

df_all = df_gennm.merge(
    df_varbert, on=["prog_name", "func_name", "varname"], how="inner"
)

print()


non_overlap_all = df_all[df_all["overlapped"] == False]

df_to_analyze = non_overlap_all
# df_to_analyze = df_all
df_group_by_binary = df_to_analyze.groupby('prog_name').agg(
    {
        'gennm_precision': 'mean',
        'gennm_recall': 'mean',
        'varbert_precision': 'mean',
        'varbert_recall': 'mean',
        'proj_in_train': 'first',
        'proj_category': 'first',
        'proj_name': 'first',
        # also count the number of each binary
        'func_name': 'count',
    }
)
ret = df_group_by_binary.groupby('proj_in_train')[
    ['gennm_precision', 'gennm_recall', 'varbert_precision', 'varbert_recall']
].mean()

print(ret)
