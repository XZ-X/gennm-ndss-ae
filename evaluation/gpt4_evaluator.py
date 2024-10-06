# isort: off
import sys

sys.path.append("common")
# isort: on
import argparse
import json
from tqdm import tqdm
from openai import OpenAI
import yaml
import pickle
import name_utils
import re

prompt_to_evaluator = """
You are an experienced C/C++ reverse engineer.
Please act as an impartial judge and evaluate the quality of names of variables in the given decompiled program.
You will be provided with 
(1) the decompiled code with groundtruth variable names,
(2) the variable names predicted by an AI assistant.

In the evaluation, you should answer the following questions:

A. Does the variable name reflect relevant context (domain)? Answer the question in range 5(best) to 1(worst).
Domain/context describes the high-level program context of the variable. 
It is more of the general high-level domain (e.g., network, memory, CPS, physics, GUI, etc) rather than specific semantics (e.g., len, size, pointer).
For 5, the predicted name and the groundtruth name should describe the same domain/context. 
Or, both the predicted name and the groundtruth name does not explicitly mention a specific domain.

For 4, the domains of the predicted name and the groundtruth name should be similar and relevant, although may not be exactly the same. The predicted name domain may be a superset or subset of the groundtruth. 
The predicted domain may be closely related to the groundtruth domain. The predicted name and groundtruth name may be two different perspectives of a same specific domain.

For 3, the predicted name does not explicitly mention a specific context, but the groundtruth name does. 
The predicted name only contains low level operations. 
From the predicted name, one cannot deduce the high-level purpose of the decompiled function/variable.

For 2, the predicted name is slightly misleading. 
The domain for predicted name is different and not relevant to the groundtruth domain. 
However, although misleading, the domain is only implied by the choice of words, and is not explicitly mentioned.

For 1, the predicted name is completely misleading. 
The name is irrelevant to the groundtruth domain, and it is explicitly mentioned in the name.

B. Does the predicted name reflect relevant semantics? Answer the question in range 1(worse) to 5(best).
Semantics means the specific high-level meanings denoted by a variable (e.g., len, errmsg, file).
For 5, the semantics of the name should be almost exactly the same to the groundtruth.
Or, both the predicted name and the groundtruth name do not have meaningful semantics.

For 4, the semantics of the predicted name are similar to the groundtruth name.
It may be vague, but the overall semantics and purpose is correct.

For 3, the predicted name does not specify meaningful semantics but the ground truth name does. 
It only indicates some low-level operations without high level abstractions.

For 2, the summary specify relevant but inaccurate semantics. 
The semantics specified in the predicted name may be relevant to the ground truth,
but they have significant differences.

For 1, the summary contains irrelevant semantics. 
It denotes a totally different semantics with the groundtruth.


You should first briefly summarize the provided decompiled code,
then for each predicted variable name, follow the workflow:

Step1: Output the placeholder variable name you are analyzing, and its ground truth name.
Step2: Explain the ground truth name. (Why it is named like that? What is the high-level context? What is the high-level semantics?)
Step3: Output the predicted name, and explain it.
Step4: Output your score in the format:

```json
{'var': (groundtruth name here), 'prediction': (predicted name here), 'score': {'Q-A': [1-5], 'Q-B': [1-5]}}
```

Repeat the process for each variable name in the predicted name map.
"""





class PromptComposer:
    def __init__(self, func_body, gt_varname_map, varname_preds):
        self.func_body = func_body
        self.gt_varname_map = gt_varname_map
        self.varname_preds = varname_preds
        self.sys_prompt = prompt_to_evaluator
        
        ground_truth_body = func_body
        for old_name, new_name in gt_varname_map.items():
            ground_truth_body = name_utils.replace_variable_names(ground_truth_body, old_name, new_name)

        pred_var_map = {}
        for k, v in varname_preds.items():
            if k not in gt_varname_map:
                continue
            if k in gt_varname_map:
                gt_varname = gt_varname_map[k]
                pred_var_map[gt_varname] = v
        
        self.reverse_map = {v: k for k, v in gt_varname_map.items()}


        self.prompt = """
<Decompiled code with groundtruth variable names>
{ground_truth_body}
</END>

<Predicted variable names>
{varname_preds}
</END>
""".format(ground_truth_body=ground_truth_body, varname_preds=pred_var_map)        

    def compose(self):
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": self.prompt},
        ]
        return messages

    def parse(self, rsp_str):
        # first, find all json blocks        
        # find ```json .* ```
        json_blocks = re.findall(r'```json(.*?)```', rsp_str, re.DOTALL)
        ret = []
        for block in json_blocks:
            entry = eval(block)
            if 'var' not in entry:
                continue
            groundtruth_name = entry['var']
            placeholder_name = self.reverse_map[groundtruth_name]
            entry['gt_varname'] = groundtruth_name
            entry['var'] = placeholder_name
            ret.append(entry)
        return ret

        


parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=str, default="")
parser.add_argument("--name-preds", type=str, default="")
parser.add_argument("--name-set", type=str, default="")
parser.add_argument("--fout", type=str, default="")
parser.add_argument("--from-idx", type=int, default=0)
parser.add_argument("--to-idx", type=int, default=101)

args = parser.parse_args()

api_key = yaml.load(open('api-key.yaml', 'r'), Loader=yaml.FullLoader)['key']

nameset_raw = [json.loads(l.strip()) for l in open(args.name_set, "r")]

nameset_dirty = {(e['prog_name'], e['func_name'], e['ori_name']) for e in nameset_raw}
nameset_varbert = {(e['binary'], e['funcname'], e['ori_name']) for e in nameset_raw}


class Prompter:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )
        return response


binary = pickle.load(open(args.binary, "rb"))

name_preds = [json.loads(l.strip()) for l in tqdm(open(args.name_preds, "r"), desc="Loading name preds")]

name_preds_by_prog_and_func = {}
for entry in name_preds:
    norm_prog_name = entry["prog_name"].split("_")[0]
    func_name = entry["func_name"]
    var_name = entry["varname"]
    if (norm_prog_name, func_name, var_name) not in nameset_dirty and (norm_prog_name, func_name, var_name) not in nameset_varbert:
        continue
    

    if (norm_prog_name, func_name) not in name_preds_by_prog_and_func:
        name_preds_by_prog_and_func[(norm_prog_name, func_name)] = []
    name_preds_by_prog_and_func[(norm_prog_name, func_name)].append(entry)

bin_func_entries_by_prog_and_func = {}
for prog in binary:    
    for func_name, func in prog.stripped_name2func.items():
        norm_prog_name = prog.prog_name.split("_")[0]
        if (norm_prog_name, func_name) not in name_preds_by_prog_and_func:
            continue
        bin_func_entries_by_prog_and_func[(norm_prog_name, func_name)] = func

sorted_bin_func_entries = sorted(bin_func_entries_by_prog_and_func.items(), key=lambda x: x[0])
if args.fout == "":
    args.fout = args.name_preds + ".gpt4eval.jsonl"
fout = open(args.fout, 'a+')
# check the length of the file
fout.seek(0)
lines = fout.readlines()
fout.seek(0, 2)
args.from_idx += len(lines)
print("Starting from %d" % args.from_idx)

prompter = Prompter(api_key, "gpt-4-turbo")
gen_config = {'temperature': 0.6}

for i in tqdm(range(len(sorted_bin_func_entries)), total=args.to_idx, desc="Asking evaluator"):
    if i < args.from_idx:
        continue
    if i > args.to_idx:
        break
    prog_func_name, bin_func_entry = sorted_bin_func_entries[i]
    predicted_entries = name_preds_by_prog_and_func[prog_func_name]
    pred_name_map = {}
    for entry in predicted_entries:
        if entry['varname'].startswith("sub_"):
            continue
        pred_name_map[entry["varname"]] = entry['pred_name']
    gt_name_map = {}
    for var_id, name in bin_func_entry.var_id_maps.items():
        gt_name_map[var_id] = name
    for func_id, name in bin_func_entry.func_id_maps.items():
        gt_name_map[func_id] = name


    prompt_composer = PromptComposer(func_body=bin_func_entry.body, gt_varname_map=gt_name_map, varname_preds=pred_name_map)
    prompt = prompt_composer.compose()
    parsed_ret = None
    retry = 0
    while retry < 5:
        response = prompter.generate(prompt, **gen_config)
        rsp_str = response.choices[0].message.content
        try:
            parsed_ret = prompt_composer.parse(rsp_str)
            if len(parsed_ret) == len(pred_name_map):
                break
        except:
            ...
        retry += 1
    if parsed_ret is None:
        parsed_ret = {}
        error = True
    else:
        error = False
    
    out_entry = {
        "prog_name": prog_func_name[0],
        "func_name": prog_func_name[1],
        "pred_varname": pred_name_map,
        "gt_varname": gt_name_map,
        "prompt": prompt,
        "response": rsp_str,
        "scored_varname": parsed_ret,
        "error": error        
    }
    fout.write(json.dumps(out_entry) + "\n")

fout.close()






print()