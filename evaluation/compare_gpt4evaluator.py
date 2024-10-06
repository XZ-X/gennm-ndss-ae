# isort: off
import sys
sys.path.append("common")
# isort: on

import json
from tqdm import tqdm
import argparse
import datasets
import re
import pandas as pd

args = argparse.Namespace()

# #################### for DIRTY dataset


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
    no_overlap_set.add((prog.split('_')[0], func))

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

def load_evaluated_entries(evaluated):
    prog_func_var2ret_entry = {}
    prog_func2gpt4ret = {}
    for ret in evaluated:
        prog = ret["prog_name"]
        func = ret["func_name"]
        prog_func2gpt4ret[(prog, func)] = ret
        for scored_name in ret["scored_varname"]:
            if "var" not in scored_name and "v" in scored_name:
                scored_name["var"] = scored_name["v"]
            elif "var" in scored_name:
                varname = scored_name["var"]
            else:
                continue
            if "score" not in scored_name:
                continue
            score_entry = scored_name["score"]
            if "Q-A" not in score_entry or "Q-B" not in score_entry:
                continue
            if (prog, func, varname) in prog_func_var2ret_entry:
                print("dup", prog, func, varname)
            prog_func_var2ret_entry[(prog, func, varname)] = scored_name
    return prog_func_var2ret_entry, prog_func2gpt4ret

gennm_prog_func_var2ret_entry, gennm_prog_func2gpt4ret = load_evaluated_entries(gennm_ret_evaluated)
varbert_prog_func_var2ret_entry, varbert_prog_func2gpt4ret = load_evaluated_entries(varbert_ret_evaluated)


overlapped_prog_funcs_var = set(gennm_prog_func_var2ret_entry.keys()) & set(varbert_prog_func_var2ret_entry.keys())
print("In total, there are {} overlapped entries".format(len(overlapped_prog_funcs_var)))


def conver_to_entry_list(entry_map, overlapped_keys):
    ret_entries = []
    for prog_func_var in overlapped_keys:
        current_entry = entry_map[prog_func_var]
        prog, func, var = prog_func_var
        if (prog, func) not in no_overlap_set:
            overlapped = True
        else:
            overlapped = False
        proj_entry = dataset_origin_map[prog]
        proj_in_train = proj_has_overlap(proj_entry)
        proj_category = proj_entry.get("category", "unknown")
        if "score" not in current_entry:
            continue
        score = current_entry["score"]
        if "Q-A" not in score or "Q-B" not in score:
            continue
        qa_score = score["Q-A"]
        qb_score = score["Q-B"]
        ret_entries.append(
            {
                "prog_name": prog,
                "func_name": func,
                "varname": var,
                "gt_varname": current_entry["gt_varname"],
                "overlapped": overlapped,
                "proj_in_train": proj_in_train,
                "proj_category": proj_category,
                "proj_name": proj_entry["project_name"],
                "qa_score": qa_score,
                "qb_score": qb_score,
            }
        )
    return ret_entries

gennm_overlapped_entries = conver_to_entry_list(gennm_prog_func_var2ret_entry, overlapped_prog_funcs_var)
varbert_overlapped_entries = conver_to_entry_list(varbert_prog_func_var2ret_entry, overlapped_prog_funcs_var)


df_gennm = pd.DataFrame(gennm_overlapped_entries)
df_varbert = pd.DataFrame(varbert_overlapped_entries)

df_gennm = df_gennm.rename(
    columns={
        "qa_score": "gennm_qa_score",
        "qb_score": "gennm_qb_score",
    }
)

df_varbert = df_varbert.rename(
    columns={
        "qa_score": "varbert_qa_score",
        "qb_score": "varbert_qb_score",
    }
)

common_cols = [
    'prog_name', 'func_name', 'varname', 'gt_varname', 'overlapped', 'proj_in_train', 'proj_category', 'proj_name'
]
df_all = df_gennm.merge(
    df_varbert, on=common_cols, how="inner"
)

df_non_overlap = df_all[df_all["overlapped"] == False]

# count the distribution of gennm_qa_score
print(df_non_overlap["gennm_qa_score"].value_counts().sort_index())
print(df_non_overlap["gennm_qb_score"].value_counts().sort_index())
print(df_non_overlap["varbert_qa_score"].value_counts().sort_index())
print(df_non_overlap["varbert_qb_score"].value_counts().sort_index())

