# isort: off
import sys

sys.path.append("common")
# isort: on

from transformers import (
    AutoModel,
    AutoTokenizer
)
import torch
import json
from tqdm import tqdm
import argparse
import pickle
import lmpa_ir
import copy
import time


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--default_name",
        type=str,
        default="",
    )
    args.add_argument(
        "--ds-in",
        type=str,
        default="",
    )
    args.add_argument(
        "--parsed-in",
        type=str,
        default="",
    )
    args.add_argument("--names", type=str, default="")
    args.add_argument("--fout", type=str, default="")
    args.add_argument("--prop_round", type=int, default=3)
    args.add_argument("--upper_bound", action="store_true")
    args = args.parse_args()

    return args


def _collect_all_exprs(parsed):
    return parsed['lmpa_ir']


class PropagationEntry:
    def __init__(self, prog_name, func_name, var_name, prop_level, prop_reason):
        self.prog_name = prog_name
        self.func_name = func_name
        self.var_name = var_name
        self.prop_level = prop_level
        self.prop_reason = prop_reason


class PropagationRecorder:
    """
    Record the propagation information for a given variable
    """

    def __init__(self, prog_name, func_name, var_name):
        self.prog_name = prog_name
        self.func_name = func_name
        self.var_name = var_name
        self.propagation_list = []
        self.prop_source_set = set()

    def has_prop_entry(self, entry):
        if (
            entry.prog_name == self.prog_name
            and entry.func_name == self.func_name
            and entry.var_name == self.var_name
        ):
            return True
        return (
            entry.prog_name,
            entry.func_name,
            entry.var_name,
        ) in self.prop_source_set

    def add_propagation(self, propagation_entry):
        if self.has_prop_entry(propagation_entry):
            return False

        self.propagation_list.append(propagation_entry)
        self.prop_source_set.add(
            (
                propagation_entry.prog_name,
                propagation_entry.func_name,
                propagation_entry.var_name,
            )
        )
        return True

    def receive_propagation_from(self, another, prop_reason=None):
        for entry in another.propagation_list:
            if self.has_prop_entry(entry):
                continue
            other_entry_copied = copy.deepcopy(entry)
            other_entry_copied.prop_level += 1
            self.add_propagation(other_entry_copied)


# callee(arg1, arg2...)
def _from_callee_args(
    current_prog_name,
    current_func_parsed,
    stripped_name2func,
    stripped_name2parsed,
    names,
    propagation_records,
):
    all_exprs = _collect_all_exprs(current_func_parsed)
    all_call_exprs = [expr for expr in all_exprs if type(expr) == lmpa_ir.LmPaCallExpr]
    for call_expr in all_call_exprs:
        callee_func_id = call_expr.func_id
        if callee_func_id not in stripped_name2parsed:
            continue
        callee_parsed = stripped_name2parsed[callee_func_id]
        callee_params = callee_parsed['lmpa_args']
        callsite_args = call_expr.args
        if len(callee_params) != len(callsite_args):
            continue
        for i in range(len(callsite_args)):
            callsite_arg = callsite_args[i]
            if (
                type(callsite_arg) != lmpa_ir.LmPaVarExpression
                and type(callsite_arg) != lmpa_ir.LmPaImplicitReturnVarExpr
            ):
                continue
            current_fully_qualified_name = (
                current_prog_name,
                current_func_parsed['func_name'],
                callsite_arg.var_name,
            )
            if current_fully_qualified_name not in propagation_records:
                propagation_records[current_fully_qualified_name] = PropagationRecorder(
                    prog_name=current_prog_name,
                    func_name=current_func_parsed['func_name'],
                    var_name=callsite_arg.var_name,
                )
            prop_fqn = (current_prog_name, callee_func_id, callee_params[i].var_name)
            if should_propagate_from(prop_fqn, names):
                prop_entry = PropagationEntry(
                    prog_name=current_prog_name,
                    func_name=callee_func_id,
                    var_name=callee_params[i].var_name,
                    prop_level=0,
                    prop_reason="from callee args",
                )
                propagation_records[current_fully_qualified_name].add_propagation(
                    prop_entry
                )


# var = sub_xxx(...)
def _from_callee_return(
    current_prog_name,
    current_func_parsed,
    stripped_name2func,
    stripped_name2parsed,
    names,
    propagation_records,
):
    all_exprs = _collect_all_exprs(current_func_parsed)
    interesting_exprs = []
    for expr in all_exprs:
        if type(expr) != lmpa_ir.LmPaBasicExpr:
            continue
        if not expr.is_direct_use:
            continue
        if len(expr.defs) == 0:
            continue
        use = expr.uses[0]
        if type(use) != lmpa_ir.LmPaImplicitReturnVarExpr:
            continue
        interesting_exprs.append(expr)
    for expr_to_analyze in interesting_exprs:
        defined_var = expr_to_analyze.defs[0]
        used_var = expr_to_analyze.uses[0]
        callee_func_id = used_var.func_id

        if callee_func_id not in stripped_name2parsed:
            continue
        callee_parsed = stripped_name2parsed[callee_func_id]
        callee_exprs = _collect_all_exprs(callee_parsed)
        callee_return_exprs = [
            expr for expr in callee_exprs if type(expr) == lmpa_ir.LmPaReturnStmt
        ]
        if len(callee_return_exprs) == 0:
            continue
        current_fully_qualified_name = (
            current_prog_name,
            current_func_parsed['func_name'],
            defined_var.var_name,
        )
        if current_fully_qualified_name not in propagation_records:
            propagation_records[current_fully_qualified_name] = PropagationRecorder(
                prog_name=current_prog_name,
                func_name=current_func_parsed['func_name'],
                var_name=defined_var.var_name,
            )

        for ret_expr in callee_return_exprs:
            if type(ret_expr.ret_val) != lmpa_ir.LmPaVarExpression:
                continue
            prop_fqn = (current_prog_name, callee_func_id, ret_expr.ret_val.var_name)
            if should_propagate_from(prop_fqn, names):
                propagation_entry = PropagationEntry(
                    prog_name=current_prog_name,
                    func_name=callee_func_id,
                    var_name=ret_expr.ret_val.var_name,
                    prop_level=0,
                    prop_reason="from callee return",
                )
                propagation_records[current_fully_qualified_name].add_propagation(
                    propagation_entry
                )

            if prop_fqn in propagation_records:
                prop_source_recorder = propagation_records[prop_fqn]
                propagation_records[
                    current_fully_qualified_name
                ].receive_propagation_from(
                    prop_source_recorder, prop_reason="from callee return"
                )


def _from_caller_args(binary_prog, current_func_parsed, names, propagation_records):
    if current_func_parsed['func_name'] not in binary_prog.call_graph:
        return {}
    callers = list(binary_prog.call_graph.predecessors(current_func_parsed['func_name']))
    if len(callers) == 0:
        return {}
    my_func_id = current_func_parsed['func_name']
    for caller in callers:
        if caller not in binary_prog.stripped_name2parsed:
            continue
        caller_parsed = binary_prog.stripped_name2parsed[caller]
        caller_exprs = _collect_all_exprs(caller_parsed)
        caller_call_exprs = [
            expr for expr in caller_exprs if type(expr) == lmpa_ir.LmPaCallExpr
        ]
        callsites = [expr for expr in caller_call_exprs if expr.func_id == my_func_id]
        if len(callsites) == 0:
            continue
        for callsite in callsites:
            callsite_args = callsite.args
            my_params = current_func_parsed['lmpa_args']
            if len(callsite_args) != len(my_params):
                continue
            for i in range(len(callsite_args)):
                callsite_arg = callsite_args[i]
                my_param_name = my_params[i].var_name
                if (
                    type(callsite_arg) != lmpa_ir.LmPaVarExpression
                    and type(callsite_arg) != lmpa_ir.LmPaImplicitReturnVarExpr
                ):
                    continue
                callsite_var_name = callsite_arg.var_name
                current_fully_qualified_name = (
                    binary_prog.prog_name,
                    current_func_parsed['func_name'],
                    my_param_name,
                )
                if current_fully_qualified_name not in propagation_records:
                    propagation_records[
                        current_fully_qualified_name
                    ] = PropagationRecorder(
                        prog_name=binary_prog.prog_name,
                        func_name=current_func_parsed['func_name'],
                        var_name=my_param_name,
                    )
                prop_fqn = (binary_prog.prog_name, caller, callsite_var_name)

                if should_propagate_from(prop_fqn, names):
                    prop_entry = PropagationEntry(
                        prog_name=binary_prog.prog_name,
                        func_name=caller,
                        var_name=callsite_var_name,
                        prop_level=0,
                        prop_reason="from caller args",
                    )
                    propagation_records[current_fully_qualified_name].add_propagation(
                        prop_entry
                    )
                if prop_fqn in propagation_records:
                    prop_source_recorder = propagation_records[prop_fqn]
                    propagation_records[
                        current_fully_qualified_name
                    ].receive_propagation_from(
                        prop_source_recorder, prop_reason="from caller args"
                    )


# (in caller) var = my_func(...)
def _from_caller_return(binary_prog, current_func_parsed, names, propagation_records):
    if current_func_parsed['func_name'] not in binary_prog.call_graph:
        return {}
    callers = list(binary_prog.call_graph.predecessors(current_func_parsed['func_name']))
    if len(callers) == 0:
        return {}
    my_exprs = _collect_all_exprs(current_func_parsed)
    my_return_exprs = [
        expr for expr in my_exprs if type(expr) == lmpa_ir.LmPaReturnStmt
    ]
    interesting_my_return_vars = []
    my_ret_var_set = set()
    for expr in my_return_exprs:
        if type(expr.ret_val) != lmpa_ir.LmPaVarExpression:
            continue
        ret_var_name = expr.ret_val.var_name
        if ret_var_name not in my_ret_var_set:
            interesting_my_return_vars.append(expr.ret_val)
            my_ret_var_set.add(ret_var_name)
    if len(interesting_my_return_vars) == 0:
        return

    my_func_id = current_func_parsed['func_name']
    for caller in callers:
        if caller not in binary_prog.stripped_name2parsed:
            continue
        caller_parsed = binary_prog.stripped_name2parsed[caller]
        caller_exprs = _collect_all_exprs(caller_parsed)
        interesting_caller_exprs = []
        for expr in caller_exprs:
            if type(expr) != lmpa_ir.LmPaBasicExpr:
                continue
            if not expr.is_direct_use:
                continue
            if len(expr.defs) == 0:
                continue
            use = expr.uses[0]
            if type(use) != lmpa_ir.LmPaImplicitReturnVarExpr:
                continue
            if use.func_id != my_func_id:
                continue
            interesting_caller_exprs.append(expr)
        for expr_to_analyze in interesting_caller_exprs:
            defined_var = expr_to_analyze.defs[0]
            caller_func_id = caller
            for ret_var in interesting_my_return_vars:
                ret_var_name = ret_var.var_name
                current_fqn = (
                    binary_prog.prog_name,
                    my_func_id,
                    ret_var_name,
                )
                if current_fqn not in propagation_records:
                    propagation_records[current_fqn] = PropagationRecorder(
                        prog_name=binary_prog.prog_name,
                        func_name=my_func_id,
                        var_name=ret_var_name,
                    )
                prop_fqn = (binary_prog.prog_name, caller, defined_var.var_name)
                if should_propagate_from(prop_fqn, names):
                    prop_entry = PropagationEntry(
                        prog_name=binary_prog.prog_name,
                        func_name=caller,
                        var_name=defined_var.var_name,
                        prop_level=0,
                        prop_reason="from caller return",
                    )
                    propagation_records[current_fqn].add_propagation(prop_entry)


# var1 = var2
def _among_direct_use(
    current_prog_name,
    current_func_parsed,
    stripped_name2func,
    stripped_name2parsed,
    names,
    propagation_records,
    prop_from_rhs=True,
):
    all_exprs = _collect_all_exprs(current_func_parsed)
    interesting_exprs = []
    for expr in all_exprs:
        if type(expr) != lmpa_ir.LmPaBasicExpr:
            continue
        if not expr.is_direct_use:
            continue
        if len(expr.defs) == 0:
            continue
        use = expr.uses[0]
        if type(use) != lmpa_ir.LmPaVarExpression:
            continue
        interesting_exprs.append(expr)
    for expr_to_analyze in interesting_exprs:
        defined_var_name = expr_to_analyze.defs[0].var_name
        used_var_name = expr_to_analyze.uses[0].var_name
        rhs_fqn = (current_prog_name, current_func_parsed['func_name'], used_var_name)
        lhs_fqn = (current_prog_name, current_func_parsed['func_name'], defined_var_name)
        if prop_from_rhs:
            prop_fqn = rhs_fqn
            receive_fqn = lhs_fqn
        else:
            prop_fqn = lhs_fqn
            receive_fqn = rhs_fqn
        if receive_fqn not in propagation_records:
            propagation_records[receive_fqn] = PropagationRecorder(
                prog_name=current_prog_name,
                func_name=current_func_parsed['func_name'],
                var_name=defined_var_name,
            )
        if should_propagate_from(prop_fqn, names):
            prop_entry = PropagationEntry(
                prog_name=current_prog_name,
                func_name=current_func_parsed['func_name'],
                var_name=prop_fqn[2],
                prop_level=0,
                prop_reason="from direct use",
            )
            propagation_records[receive_fqn].add_propagation(prop_entry)
        if prop_fqn in propagation_records and prop_from_rhs:
            prop_source_recorder = propagation_records[prop_fqn]
            propagation_records[receive_fqn].receive_propagation_from(
                prop_source_recorder, prop_reason="from direct use"
            )


CONFIDENT_THRESHOLD = 0
def _filter_new_names(new_names):
    filtered_new_names = {}
    # rule out names where the propagation source does not agree

    for k, v in new_names.items():
        prog_name, func_name, varname = k
        ori_preds, prop_preds = v["new_names"]
        if len(ori_preds) == 0:
            continue
        confident_props = []
        for one_prop_source in prop_preds:
            name_cnt = {}
            cnt = 0
            for entry in one_prop_source:
                if "<empty" in entry["pred_name"]:
                    continue
                pred_name = entry["pred_name"]
                if pred_name not in name_cnt:
                    name_cnt[pred_name] = 0
                name_cnt[pred_name] += 1
                cnt += 1
            total = cnt
            if total == 0:
                continue
            sorted_name_cnt = sorted(name_cnt.items(), key=lambda x: x[1], reverse=True)
            max_cnt = sorted_name_cnt[0][1]
            if max_cnt / total > CONFIDENT_THRESHOLD:
                confident_props.append(sorted_name_cnt[0][0])
        if len(confident_props) > 0:
            v["confident_new_names"] = confident_props
            filtered_new_names[k] = v
    return filtered_new_names


propagation_source_record = {}


def should_propagate_from(propagation_source, names):
    if propagation_source in propagation_source_record:
        return propagation_source_record[propagation_source]
    name_cnt = {}
    if propagation_source not in names:
        propagation_source_record[propagation_source] = False
        return False
    pred_names = names[propagation_source]["name_list"]
    # if len(pred_names) >= 2:
    #   if pred_names[0]["pred_name"] == pred_names[1]["pred_name"]:
    #       propagation_source_record[propagation_source] = True
    #       return True
    name_cnt = {}
    cnt = 0
    for entry in pred_names:
        pred_name = entry["pred_name"]
        if "<empty" in pred_name:
            continue
        if pred_name not in name_cnt:
            name_cnt[pred_name] = 0
        name_cnt[pred_name] += 1
        cnt += 1
    total = cnt
    if total == 0:
        propagation_source_record[propagation_source] = False
        return False
    

    sorted_name_cnt = sorted(name_cnt.items(), key=lambda x: x[1], reverse=True)
    max_cnt = sorted_name_cnt[0][1]
    if max_cnt / total > CONFIDENT_THRESHOLD:
        propagation_source_record[propagation_source] = True
        return True
    else:
        propagation_source_record[propagation_source] = False
        return False


def main():
    args = parse_args()
    codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    codebert = AutoModel.from_pretrained("microsoft/codebert-base").eval().cuda()

    default_names = {}
    fin = open(args.default_name, "r").readlines()
    for line in tqdm(fin, desc="Loading default names"):
        entry = json.loads(line)
        prog_name = entry["prog_name"]
        func_name = entry["func_name"]
        varname = entry["varname"]
        default_names[(prog_name, func_name, varname)] = entry

    names = {}
    fin = open(args.names, "r").readlines()
    for line in tqdm(fin, desc="Loading names"):
        entry = json.loads(line)
        name_list = eval(entry["name_list"])
        entry["name_list"] = name_list
        prog_name = entry["prog_name"]
        func_name = entry["func_name"]
        varname = entry["varname"]
        names[(prog_name, func_name, varname)] = entry

    data = pickle.load(open(args.ds_in, "rb"))
    parsed_prog = pickle.load(open(args.parsed_in, "rb"))
    prog_func_name2parsed = {}
    for entry in tqdm(parsed_prog, desc="Loading parsed prog"):
        if entry['prog_name'] not in prog_func_name2parsed:
            prog_func_name2parsed[entry['prog_name']] = {}
        prog_func_name2parsed[entry['prog_name']][entry['func_name']] = entry

    time_before_prop = time.time()
    new_names = {}
    prop_stats = {}
    propagation_records = {}
    for prog in tqdm(data, desc="Propagating names"):
        # propagate one prog
        stripped_name2func = prog.stripped_name2func
        stripped_name2parsed = prog_func_name2parsed[prog.prog_name]
        # stripped_name2parsed = prog.stripped_name2parsed
        func_name_list = list(stripped_name2func.keys())
        current_prog_name = prog.prog_name
        prog.stripped_name2parsed = stripped_name2parsed
        for prop_rnd in range(args.prop_round):
            for current_func_name in func_name_list:
                if current_func_name not in stripped_name2parsed:
                    continue
                current_func_parsed = stripped_name2parsed[current_func_name]

                _from_callee_args(
                    current_prog_name,
                    current_func_parsed,
                    stripped_name2func,
                    stripped_name2parsed,
                    names,
                    propagation_records,
                )
                _from_callee_return(
                    current_prog_name,
                    current_func_parsed,
                    stripped_name2func,
                    stripped_name2parsed,
                    names,
                    propagation_records,
                )
                _from_caller_args(prog, current_func_parsed, names, propagation_records)
                _from_caller_return(
                    prog, current_func_parsed, names, propagation_records
                )
                _among_direct_use(
                    current_prog_name,
                    current_func_parsed,
                    stripped_name2func,
                    stripped_name2parsed,
                    names,
                    propagation_records,
                    prop_from_rhs=True,
                )
                _among_direct_use(
                    current_prog_name,
                    current_func_parsed,
                    stripped_name2func,
                    stripped_name2parsed,
                    names,
                    propagation_records,
                    prop_from_rhs=False,
                )

    for k, v in propagation_records.items():
        for entry in v.propagation_list:
            if entry.prop_reason not in prop_stats:
                prop_stats[entry.prop_reason] = 0
            prop_stats[entry.prop_reason] += 1

    ret = {}
    for ori, prop_record in propagation_records.items():
        names_to_propagate = []
        ori_key = ori
        if ori_key not in names:
            ori_preds = []
        else:
            ori_preds = names[ori_key]["name_list"]
        for prop_entry in prop_record.propagation_list:
            prop_fqn = (
                prop_entry.prog_name,
                prop_entry.func_name,
                prop_entry.var_name,
            )
            if prop_fqn not in names:
                continue
            prop_preds = names[prop_fqn]["name_list"]
            names_to_propagate.append(prop_preds)

        if len(names_to_propagate) > 0:
            ret[ori] = (ori_preds, names_to_propagate)
    for k, v in ret.items():
        new_names[k] = {
            "varname": k[2],
            "new_names": v,
        }

    print("Before filtering, propagation stats: " + str(prop_stats))
    filtered_new_names = _filter_new_names(new_names)
    print("After filtering, remaining new names: " + str(len(filtered_new_names)))
    time_before_name_selection = time.time()
    new_name_selections = {}
    default_not_in = 0
    for k, v in tqdm(names.items(), desc="Selecting new names"):
        if args.upper_bound:
            sorted_name_list = sorted(
                v["name_list"], key=lambda x: x["precision"], reverse=True
            )
            selected = sorted_name_list[0]
            new_name_selections[k] = selected
            continue
        name_candidates_list = []
        name2name_list_entry = {}
        for name_list_entry in v["name_list"]:
            if "<empty" in name_list_entry["pred_name"]:
                continue
            name2name_list_entry[name_list_entry["pred_name"]] = name_list_entry
            name_candidates_list.append(name_list_entry["pred_name"])
        name_candidates = sorted(
            [n for n in list(name2name_list_entry.keys()) if "<empty" not in n]
        )
        if len(name_candidates) == 0:
            if k in default_names:
                selected = default_names[k]
                default_not_in += 1
                new_name_selections[k] = selected
            continue
        if len(set(name_candidates)) == 1:
            selected = name2name_list_entry[name_candidates[0]]
            new_name_selections[k] = selected
            continue
        tokenized_names = codebert_tokenizer(name_candidates, return_tensors="pt", padding=True, truncation=True)
        tokenized_names = {k: v.cuda() for k, v in tokenized_names.items()}
        with torch.no_grad():
            name_embs = codebert(**tokenized_names).last_hidden_state[:, 0, :]            
            name_embs_tensor = name_embs

        if k not in filtered_new_names:
            if k in default_names:
                selected = default_names[k]
                new_name_selections[k] = selected
            continue

        prop_names = filtered_new_names[k]["confident_new_names"]
        names_for_voting = prop_names + name_candidates_list
        # names_for_voting = prop_names
        prop_name_tokenized = codebert_tokenizer(names_for_voting, return_tensors="pt", padding=True, truncation=True)
        prop_name_tokens = {k: v.cuda() for k, v in prop_name_tokenized.items()}
        with torch.no_grad():
            prop_name_embs = codebert(**prop_name_tokens).last_hidden_state[:, 0, :]
            prop_name_embs_tensor = prop_name_embs
            name_embs_tensor = name_embs_tensor / name_embs_tensor.norm(
                dim=-1, keepdim=True
            )
            prop_name_embs_tensor = prop_name_embs_tensor / prop_name_embs_tensor.norm(
                dim=-1, keepdim=True
            )
            similarity = torch.matmul(
                name_embs_tensor, prop_name_embs_tensor.transpose(0, 1)
            )
            scores = torch.mean(similarity, dim=-1)
            sorted_scores = torch.argsort(scores, descending=True)
            selected = name2name_list_entry[name_candidates[sorted_scores[0]]]
            new_name_selections[k] = selected

    print("Default not in: " + str(default_not_in))
    with open(args.fout, "w") as f:
        for k, v in new_name_selections.items():
            prog_name, func_name, varname = k
            f.write(
                json.dumps(
                    {
                        "prog_name": prog_name,
                        "func_name": func_name,
                        "varname": varname,
                        **v,
                    }
                )
                + "\n"
            )
    print()
    final_time = time.time()
    print("Propagation time: " + str(time_before_name_selection - time_before_prop))
    print("Name selection time: " + str(final_time - time_before_name_selection))


if __name__ == "__main__":
    main()
