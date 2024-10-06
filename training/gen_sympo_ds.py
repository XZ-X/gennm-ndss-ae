# isort: off
import sys

sys.path.append("evaluation")
# isort: on
import json
from tqdm import tqdm

from name_utils import try_demangle
import numpy as np
import argparse
import datasets
from eval_utils import score_name
from transformers import AutoTokenizer
import re
from tree_sitter import Language, Parser
import tree_sitter_utils as ts_utils

CPP_LANGUAGE = Language("tree-sitter-repos/build/my-languages.so", "cpp")
C_LANGUAGE = Language("tree-sitter-repos/build/my-languages.so", "c")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train-ds-in",
    type=str,
    default="",
)
parser.add_argument(
    "--use-gt",
    action="store_true",
)
parser.add_argument("--align-order", type=bool, default=True)

parser.add_argument(
    "--filter-data",
    action="store_true",
)


parser.add_argument("--tokenizer", type=str, default="google/codegemma-2b")
parser.add_argument("--token-level-masking", action="store_true")
parser.add_argument("--keep-format", action="store_true")
parser.add_argument("--ghidra-mode", action="store_true")
parser.add_argument("--sympo-ds-out", type=str, required=True)
MAP_NUM_WORKER = 24

args = parser.parse_args()

FUNC_PREFIX = "sub_"
if args.ghidra_mode:
    FUNC_PREFIX = "FUN_"
    print("Using GHIDRA mode, prefix is FUN_")

# Load the dataset
train_ds_in = []
for l in tqdm(open(args.train_ds_in, "r")):
    try:
        entry = json.loads(l) 
        train_ds_in.append(entry)
    except:
        continue

if args.token_level_masking:
    # validate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer_keys = [k for k in tokenizer.vocab.keys()]
    white_space_char = tokenizer.tokenize(' ')[0]
    # special pattern: a quote followed by a number
    interesting_pattern1 = re.compile(r"\'[0-9a-zA-Z]")
    interesting_pattern2 = re.compile(r"[0-9a-zA-Z]\'")
    interesting_pattern3 = re.compile(r":"+white_space_char)
    # find tokens that might affect our mask
    interesting_tokens = []
    for k in tokenizer_keys:
        if re.findall(interesting_pattern1, k):
            interesting_tokens.append(k)
        if re.findall(interesting_pattern2, k):
            interesting_tokens.append(k)
        if re.findall(interesting_pattern3, k):
            interesting_tokens.append(k)

    if len(interesting_tokens) > 0:
        print(
            "Note that I assume I can safely split a string by a quote followed by var name"
        )
        print(
            "However, the following tokens are in the vocab of the tokenizer: ",
            interesting_tokens,
        )
        exit(0)

#############################
# find sympo candidates by two criteria:
# 1. each sample has at least one variable to rename
# 2. each sample has at least two different predictions
#############################

non_empty_train_samples = [s for s in train_ds_in if len(s["var_id_maps"]) > 0]

sympo_candidates = []
# make sure each sample has at least two different candidates
for entry in tqdm(non_empty_train_samples):
    var_ids = sorted(entry["var_id_maps"].keys())
    name_list_set = set()
    interesting_name_list = []
    for preds, prob in entry["answer_and_probs"]:
        pred_name_list = [preds[v] for v in var_ids if v in preds]
        if len(pred_name_list) != len(var_ids):
            continue
        pred_names_str = "#".join(pred_name_list)
        if pred_names_str not in name_list_set:
            name_list_set.add(pred_names_str)
            interesting_name_list.append(preds)
    if len(interesting_name_list) > 1:
        sympo_candidates.append(entry)


def get_sympo_entry(entry):
    sympo_entries = []
    # first, collect the best name for each id
    best_name_for_each_id = {}
    best_score_for_each_id = {}
    gt_name_for_each_id = {}
    name_list_score = []
    name_map = list(entry["var_id_maps"].items()) + list(entry["func_id_maps"].items())
    if args.use_gt:
        for id, gt_name in name_map:
            best_name_for_each_id[id] = gt_name
            if id in entry["func_id_maps"]:
                best_name_for_each_id[id] = try_demangle(gt_name, silent=True)
            best_score_for_each_id[id] = 1
            gt_name_for_each_id[id] = best_name_for_each_id[id]
    else:
        for id, gt_name in name_map:
            gt_name_for_each_id[id] = gt_name
            if id in entry["func_id_maps"]:
                gt_name_for_each_id[id] = try_demangle(gt_name, silent=True)

    for preds, prob in entry["answer_and_probs"]:
        current_var_score_list = []
        current_func_score_list = []
        for id, gt_name in name_map:
            if id not in preds:
                continue
            pred_name = preds[id]
            pr, rc = score_name(gt_name, pred_name)
            if pr + rc < 0.001:
                f1 = 0
            else:
                f1 = 2 * pr * rc / (pr + rc)
            if id not in best_score_for_each_id:
                best_score_for_each_id[id] = 0
                best_name_for_each_id[id] = ""
            if f1 >= best_score_for_each_id[id]:
                best_score_for_each_id[id] = f1
                best_name_for_each_id[id] = pred_name
            if not id.startswith(FUNC_PREFIX):
                current_var_score_list.append(f1)
            else:
                current_func_score_list.append(f1)
        if len(current_var_score_list) == 0:
            avg_var_score = 0
        else:
            avg_var_score = np.mean(current_var_score_list)
        if len(current_func_score_list) == 0:
            avg_func_score = 0
        else:
            avg_func_score = np.mean(current_func_score_list)
        name_list_score.append((preds, avg_var_score, avg_func_score))
    var_best_scores = [
        s for k, s in best_score_for_each_id.items() if not k.startswith(FUNC_PREFIX)
    ]
    if len(var_best_scores) == 0:
        return sympo_entries
    best_score = np.mean(var_best_scores)
    # if even the best score is low, that means the model has no hope on this function, let's skip
    if best_score < 0.2:
        return sympo_entries    
    # sort first by var score, then by func score, from low to high
    sorted_name_list_score = sorted(name_list_score, key=lambda x: (x[1], x[2]))
    # everyone should align with the namemap order
    if args.align_order:
        new_ordered_best_name = {}
        for k,v in name_map:
            if k in best_name_for_each_id:
                new_ordered_best_name[k] = best_name_for_each_id[k]
        best_name_for_each_id = new_ordered_best_name
    # pick the worst one
    worst_name = sorted_name_list_score[0][0]
    rejected_score = sorted_name_list_score[0][1]
    new_ordered_worst_name = {}
    for k, v in best_name_for_each_id.items():
        if k in worst_name:
            new_ordered_worst_name[k] = worst_name[k]
    should_skip = False
    if args.align_order:
        worst_name = new_ordered_worst_name
        if len(worst_name) != len(best_name_for_each_id):
            should_skip = True
    if not should_skip:
        skip_vars = set()
        for k, s in best_score_for_each_id.items():
            if s < 0.4:
                skip_vars.add(k)
        interesting_vars = set()
        for k, v in worst_name.items():
            if k in skip_vars:
                continue
            if k not in gt_name_for_each_id:
                continue
            if k not in best_score_for_each_id:
                continue
            pr, rc = score_name(gt_name_for_each_id[k], v)
            if pr + rc < 0.001:
                f1 = 0
            else:
                f1 = 2 * pr * rc / (pr + rc)
            current_id_best = best_score_for_each_id[k]
            if f1 + 0.1 < current_id_best:
                interesting_vars.add(k)

        sympo_entries.append(
            {
                "prog_name": entry["prog_name"],
                "func_name": entry["func_name"],
                "instruction": "",
                "input": entry["ask_str"].lstrip("<bos>"),
                "chosen": ":" + str(best_name_for_each_id),
                "rejected": ":" + str(worst_name),
                "chosen_score": best_score,
                "rejected_score": rejected_score,
                "gt": ":" + str(gt_name_for_each_id),
                "interesting_vars": str(interesting_vars),
            }
        )
    for i in range(1, len(sorted_name_list_score)):
        best_name_combine = sorted_name_list_score[-i][0]
        best_name_combine_score = sorted_name_list_score[-i][1]
        new_ordered_best_name_combine = {}
        for k, v in best_name_for_each_id.items():
            if k in best_name_combine:
                new_ordered_best_name_combine[k] = best_name_combine[k]
        if args.align_order:
            if len(new_ordered_best_name_combine) != len(best_name_for_each_id):
                continue
        if args.filter_data:
            if best_name_combine_score >= best_score:
                continue
        if (
            new_ordered_best_name_combine != new_ordered_worst_name
            and new_ordered_best_name_combine != best_name_for_each_id
        ):
            if args.align_order:
                best_name_combine = new_ordered_best_name_combine
            
            skip_vars = set()
            for k, s in best_score_for_each_id.items():
                if s < 0.4:
                    skip_vars.add(k)
            interesting_vars = set()
            for k, v in best_name_combine.items():
                if k in skip_vars:
                    continue
                if k not in gt_name_for_each_id:
                    continue
                if k not in best_score_for_each_id:
                    continue
                pr, rc = score_name(gt_name_for_each_id[k], v)
                if pr + rc < 0.001:
                    f1 = 0
                else:
                    f1 = 2 * pr * rc / (pr + rc)
                current_id_best = best_score_for_each_id[k]
                if f1 + 0.1 < current_id_best:
                    interesting_vars.add(k)
            sympo_entries.append(
                {
                    "prog_name": entry["prog_name"],
                    "func_name": entry["func_name"],
                    "instruction": "",
                    "input": entry["ask_str"].lstrip("<bos>"),
                    "chosen": ":" + str(best_name_for_each_id),
                    "rejected": ":" + str(best_name_combine),
                    "chosen_score": best_score,
                    "rejected_score": best_name_combine_score,
                    "gt": ":" + str(gt_name_for_each_id),
                    "interesting_vars": str(interesting_vars),
                }
            )
            break
    return sympo_entries


sympo_entries_global = []
import multiprocessing

pool = multiprocessing.Pool(24)

ret = pool.imap_unordered(get_sympo_entry, tqdm(sympo_candidates))
for l in ret:
    sympo_entries_global.extend(l)


import pickle


if args.filter_data:

    not_reasonable_entries = []
    resonable_entries = []
    for entry in sympo_entries_global:
        if entry["chosen_score"] <= entry["rejected_score"]:
            not_reasonable_entries.append(entry)
        else:
            resonable_entries.append(entry)

    def get_callee_names(func_def, return_set=False):
        func_calls = ts_utils.find_all_recursively(func_def, "call_expression")
        callee_names = []
        for call in func_calls:
            callee_name = ts_utils.get_first_opt(call, "identifier")
            if callee_name is not None:
                callee_names.append(callee_name.text.decode("utf-8"))
        if return_set:
            return sorted(set(callee_names))
        return sorted(callee_names)

    def heuristic_is_overfit(body):
        cpp_parser = Parser()
        cpp_parser.set_language(CPP_LANGUAGE)
        root = cpp_parser.parse(bytes(body, "utf8"))
        my_def = ts_utils.find_first_recursively_opt(
            root.root_node, "function_definition"
        )
        if my_def is None:
            return False
        string_literals = ts_utils.find_all_recursively(my_def, "string_literal")
        string_literals_value = [s.text.decode("utf-8") for s in string_literals]
        interesting_strings = [s for s in string_literals_value if len(s) > 20]
        if len(interesting_strings) > 1:
            return False
        callees = get_callee_names(my_def)
        # how many 'sub_' functions are called
        sub_count = len([c for c in callees if c.startswith(FUNC_PREFIX)])
        if sub_count > len(callees) * 0.3:
            return True
        return False

    overfits = []
    not_overfits = []
    for entry in tqdm(resonable_entries, desc="using heuristic to check overfits"):
        body = entry["input"].strip().split("\n\n\nQ:")[0].strip()
        if heuristic_is_overfit(body):
            overfits.append(entry)
        else:
            not_overfits.append(entry)

    print("Beginning entry number: ", len(sympo_entries_global))
    print(
        "Reasonable entry number: %d (%.2f)"
        % (len(resonable_entries), len(resonable_entries) / len(sympo_entries_global))
    )
    print(
        "Non-Overfit entry number: %d (%.2f)"
        % (len(not_overfits), len(overfits) / len(sympo_entries_global))
    )
    final_data = not_overfits
else:
    final_data = sympo_entries_global

dataset = datasets.Dataset.from_list(final_data)

ds_shuffled = dataset.shuffle(seed=42)
ds_shuffled.push_to_hub(args.sympo_ds_out, private=True)
