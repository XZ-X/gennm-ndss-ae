# isort: off
import sys
sys.path.append('common')
# isort: on
import json
import argparse
import pickle
from tqdm import tqdm
import multiprocessing
import name_utils
import tree_sitter_cpp
import tree_sitter_utils as ts_utils
from transformers import (
    AutoTokenizer,
    AutoModel
)
import datasets
import numpy as np
from tree_sitter import Language, Parser

np.random.seed(43)

# FUNC_PREFIX = "func_"
FUNC_PREFIX = "sub_"

CPP_LANGUAGE = Language(tree_sitter_cpp.language())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-ret", type=str, default=""
    )
    parser.add_argument(
        "--train-ds", type=str, default=""
    )
    parser.add_argument(
        "--for-training", action="store_true"
    )
    parser.add_argument(
        "--gt-only", action="store_true"
    )
    parser.add_argument("--binary", type=str, default="")
    parser.add_argument("--fout", type=str, default="")
    parser.add_argument("--func-prefix", type=str, default="sub_")
    args = parser.parse_args()
    global FUNC_PREFIX
    FUNC_PREFIX = args.func_prefix
    return args

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

def get_tree_sitter_root(body):
    cpp_parser = Parser(CPP_LANGUAGE)
    root = cpp_parser.parse(bytes(body, "utf8"))
    return root


def heuristic_local_context_sufficient(entry, body, root):
    my_func_name = entry.func_name
    if my_func_name is None:
        return False
    if not my_func_name.startswith(FUNC_PREFIX):
        return True

    my_def = ts_utils.find_first_recursively_opt(
        root.root_node, "function_definition"
    )
    if my_def is None:
        return False
    string_literals = ts_utils.find_all_recursively(my_def, "string_literal")
    string_literals_value = [s.text.decode("utf-8") for s in string_literals]
    interesting_strings = [s for s in string_literals_value if len(s) > 20]
    if len(interesting_strings) > 1:
        return True
    callees = get_callee_names(my_def)
    # how many 'sub_' functions are called
    sub_count = len([c for c in callees if c.startswith(FUNC_PREFIX) and c != my_func_name])
    if sub_count > len(callees) * 0.3:
        return False
    return True

def get_call_expressions(entry, body, root):
    my_def = ts_utils.find_first_recursively_opt(
        root.root_node, "function_definition"
    )
    if my_def is None:
        return []
    call_expressions = ts_utils.find_all_recursively(my_def, "call_expression")
    ret = []
    for call in call_expressions:
        callee_name = ts_utils.get_first_opt(call, "identifier")
        if callee_name is None:
            continue
        callee_name = callee_name.text.decode("utf-8")
        call_expr = call.text.decode("utf-8")
        ids_in_call_expr = ts_utils.find_all_recursively(call, "identifier")
        ids_in_call_expr = [i.text.decode("utf-8") for i in ids_in_call_expr]
        ret.append((callee_name, call_expr, ids_in_call_expr))
    return ret

def pick_name_for_training(entry):
    return entry["answer_and_probs"][0][0]

def pick_name_for_test(entry, encoder, tokenizer):
    return entry





def rename_all(string, name_map):
    new_string = string
    for k, v in name_map.items():
        new_string = name_utils.replace_variable_names(new_string, k, v)
    return new_string


def process_one_binary(program):
    statistics = {
        'gt': 0,
        'infer': 0,
    }
    if program.prog_name not in prog_func2entry and not args.for_training:
        return {}
    # collect all possible hints
    name_hints_from_callee = {}
    name_hints_from_callsite = {}
    for stripped_name, func_entry in program.stripped_name2func.items():
        if not args.for_training:
            if stripped_name not in prog_func2entry[program.prog_name]:
                continue            
        body = func_entry.body
        root = get_tree_sitter_root(body)                
        if heuristic_local_context_sufficient(func_entry, body, root):
            if args.for_training:                
                if not args.gt_only:
                    if program.prog_name in prog_func2entry and stripped_name in prog_func2entry[program.prog_name]:                        
                        entry = prog_func2entry[program.prog_name][stripped_name]
                        my_names = pick_name_for_training(entry)
                        statistics['infer'] += 1
                    else:
                        # get gt first
                        my_names = {}
                        for var_id, var_name in func_entry.var_id_maps.items():
                            my_names[var_id] = var_name
                        for func_id, func_name in func_entry.func_id_maps.items():
                            my_names[func_id] = name_utils.try_demangle(func_name, silent=True)                        
                        statistics['gt'] += 1
                else:
                    statistics['gt'] += 1
                    my_names = {}
                    for var_id, var_name in func_entry.var_id_maps.items():
                        my_names[var_id] = var_name
                    for func_id, func_name in func_entry.func_id_maps.items():
                        my_names[func_id] = name_utils.try_demangle(func_name, silent=True)
            else:                
                entry = prog_func2entry[program.prog_name][stripped_name]
                my_names = pick_name_for_test(entry, codebert, codebert_tokenizer)
            if stripped_name in my_names:
                func_signature = ts_utils.find_first_recursively_opt(root.root_node, "function_declarator")
                if func_signature is not None:
                    my_names_for_sig = my_names
                    func_signature = func_signature.text.decode("utf-8")
                    func_signature_renamed = rename_all(func_signature, my_names_for_sig)
                    name_hints_from_callee[stripped_name] = func_signature_renamed
            # (callee_name, call_expr, ids_in_call_expr)
            call_expressions = get_call_expressions(func_entry, body, root)
            for callee_name, call_expr, ids_in_call_expr in call_expressions:
                new_call_expr = call_expr                    
                call_expr_renamed = rename_all(call_expr, my_names)
                if call_expr_renamed != call_expr:
                    if callee_name not in name_hints_from_callsite:
                        name_hints_from_callsite[callee_name] = []
                    name_hints_from_callsite[callee_name].append(call_expr_renamed)

    # associate hints via call graph
    hints_in_context = {}
    for stripped_name, func_entry in program.stripped_name2func.items():
        hints_in_context[stripped_name] = {
            'callsites': [],
            'callees': []
        }
        for current_func, callee in program.call_graph.out_edges(stripped_name):
            if current_func == callee:
                continue
            if callee in name_hints_from_callee:
                hints_in_context[stripped_name]['callees'].append((callee, name_hints_from_callee[callee]))
        if stripped_name in name_hints_from_callsite:
            hints_in_context[stripped_name]['callsites'] = name_hints_from_callsite[stripped_name]
    return {program.prog_name: hints_in_context, 'statistics': statistics}


prog_func2entry = {}
train_prog_func2entry = {}

if __name__ == "__main__":
    args = parse_args()
    print("FUNC_PREFIX", FUNC_PREFIX)
    # # dbg
    # args.for_training = True
    if not args.for_training:
        codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        codebert = AutoModel.from_pretrained("microsoft/codebert-base").eval().cuda()
    else:
        train_ds = datasets.load_dataset(args.train_ds)
        for entry in tqdm(train_ds['train'], desc="loading train ds"):
            prog_name = entry["prog_name"]
            func_name = entry["strip_func_name"]
            train_prog_func2entry[(prog_name, func_name)] = entry        
        print()

    if args.model_ret != "":
        lines = open(args.model_ret, "r").readlines()
        model_ret = [json.loads(l) for l in tqdm(lines)]
    else:
        model_ret = []
    binaries = pickle.load(open(args.binary, "rb"))
    if args.for_training:
        for entry in tqdm(model_ret):
            prog_name = entry["prog_name"]
            func_name = entry["func_name"]
            parsed_answers = []
            for ans_prob in  entry["answer_and_probs"]:
                ans = ans_prob[0]
            if prog_name not in prog_func2entry:
                prog_func2entry[prog_name] = {}
            if func_name not in prog_func2entry[prog_name]:
                prog_func2entry[prog_name][func_name] = entry
            else:        
                print("Duplicate entry", prog_name, func_name)
    else:
        for entry in tqdm(model_ret):
            prog_name = entry["prog_name"]
            func_name = entry["func_name"]
            if prog_name not in prog_func2entry:
                prog_func2entry[prog_name] = {}
            if func_name not in prog_func2entry[prog_name]:
                prog_func2entry[prog_name][func_name] = {}
            varname = entry["varname"]
            pred_name = entry["pred_name"]
            prog_func2entry[prog_name][func_name][varname] = pred_name



    if args.for_training:
        pool = multiprocessing.Pool(8)
        rets = pool.imap_unordered(process_one_binary, tqdm(binaries, desc="processing binaries"))
        # pool.close()
    else:
        rets = []
        for binary in tqdm(binaries, desc="processing binaries"):
            rets.append(process_one_binary(binary))
    
    prog_func2hints = {}
    statistics = {}
    for ret_dict in tqdm(rets):
        for k, v in ret_dict.items():
            if 'statistics' == k:
                for k2, v2 in v.items():
                    if k2 not in statistics:
                        statistics[k2] = 0
                    statistics[k2] += v2
                continue

            if k not in prog_func2hints:
                prog_func2hints[k] = {}
            prog_func2hints[k].update(v)
    print(statistics)

    with open(args.fout, "w") as f:
        json.dump(prog_func2hints, f, indent=2)    


