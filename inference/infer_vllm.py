# isort: off
import sys
sys.path.append("common")
# isort: on
from transformers import (
    HfArgumentParser,
    GenerationConfig
)
import os
import json
from tqdm import tqdm
import re
from dataclasses import dataclass, field
from typing import Optional
import pickle
import hints_utils
from vllm import LLM, SamplingParams


@dataclass
class LmpaArguments:
    model_id: Optional[str] = field(
        default="",
    )
    subfolder: Optional[str] = field(
        default="",
    )

    progs: Optional[str] = field(
        default="",
    )

    fout: Optional[str] = field(
        default="",
    )

    from_idx: Optional[int] = field(
        default=0,
    )
    to_idx: Optional[int] = field(
        default=999999,
    )
    temperature: Optional[float] = field(
        default=0.8,
    )
    max_func_strlen: Optional[int] = field(
        default=5120,
    )
    num_return_sequences: Optional[int] = field(
        default=3,
    )
    interesting_only: Optional[str] = field(
        default="",
    )
    no_prompt: Optional[bool] = field(
        default=False,
    )

    prop_callee_name: Optional[bool] = field(
        default=False,
    )
    dtype: Optional[str] = field(
        default="float16",
    )
    hint: Optional[str] = field(
        default="",
    )
    # vllm args
    gpu_util: Optional[float] = field(
        default=0.8,
    )

    tensor_parallel_size: Optional[int] = field(
        default=1,
    )

    swap_space: Optional[int] = field(
        default=16,
    )

class AskFTConfig(GenerationConfig):
    def __init__(self, 
                 prompt='prompt-ft.txt',
                 suffix=None,                 
                 max_func_strlen=5120,
                 max_ids=10,
                 **kwargs):
        super().__init__(**kwargs)
        if prompt is None:
            self.prompt = ''
        else:
            self.prompt = open(prompt, 'r').read()
        self.max_func_strlen = max_func_strlen
        self.max_ids = max_ids
        # self.prefix = "[INST]\n" + self.prompt + "\n"
        self.prefix = self.prompt + "\n"
        if suffix is None:
            self.suffix = """
            What would be a meaningful name for `%s`?
            [/INST]

            A meaningful name for `%s` would be `
            """.rstrip()
            self.answer_pattern = re.compile(r'A meaningful name for `[^`]*` would be `([^`]*)`')
        else:            
            self.suffix = suffix            



class AskFTManager:

    def __init__(self, config, func, hints_entry, tokenizer):
        self.config = config
        self.func_to_ask = func
        self.var_id_maps = func.var_id_maps
        self.func_id_maps = func.func_id_maps   
        self.hints_entry = hints_entry             
        self.tokenizer = tokenizer
    

    def ask_ids(self):
        stripped_func_code = self.func_to_ask.body
        stripped_func_code = stripped_func_code[:self.config.max_func_strlen]
        ids = list(self.var_id_maps.keys()) + list(self.func_id_maps.keys())
        ask_text = "\n\n"
        if self.hints_entry is not None:
            stripped_func_code, hint_string = hints_utils.mix_hints_w_code(stripped_func_code, self.hints_entry)            
        # if self.hint_string != "":
        #     ask_text += self.hint_string + "\n"
        ask_text += "\n\nQ:["
        for var_id in ids[:self.config.max_ids]:
            ask_text += var_id + ","
        # ask_text = ask_text+ "]\n"
        # ask_text += "Assistant: "
        ask_text = ask_text+ "]A"
        # ask_text += " "
        all_text = self.config.prefix + stripped_func_code + ask_text
        return all_text

    

def parse_answer(answer):
    answer = answer.lstrip(':').strip()
    try:            
        ret = eval(answer)
    except:
        ret = None
        answer_pattern = re.compile(r'{[^}]*}')
        found = answer_pattern.findall(answer)
        if len(found) > 0:
            second_try = found[-1]
        else:
            second_try = answer
    if ret is None:
        try:
            ret = eval(second_try)
        except:
            ret = None
    if ret is not None:
        if not isinstance(ret, dict):
            ret = {'Error': answer}
        new_ret = {}
        for k, v in ret.items():
            if not ( isinstance(k, str) and isinstance(v, str)):
                new_ret[str(k)] = str(v)
            else:
                new_ret[k] = v
        ret = new_ret
    else:
        ret = {'Error': answer}
    return ret        


def gen_prompts(lmpa_args, prog, pbar, hints, interesting_only):
    ret_prompts = []
    call_graph = prog.call_graph
    stripped_name2func = prog.stripped_name2func
    # calculate out degree
    nodes_out_degree = []
    for node in call_graph.nodes:
        nodes_out_degree.append((node, call_graph.out_degree(node)))
    nodes_out_degree = sorted(nodes_out_degree, key=lambda x: x[1])
    for node in nodes_out_degree:
        if node[0] not in stripped_name2func:
            continue
        else:
            func_to_ask = stripped_name2func[node[0]]
            if pbar.n < lmpa_args.from_idx:
                pbar.update(1)
                continue
            if pbar.n > lmpa_args.to_idx:
                break
            pbar.update(1)
            if interesting_only is not None and (prog.prog_name, func_to_ask.func_name) not in interesting_only:
                continue
            ask_config = AskFTConfig(       
                prompt=None,
                max_new_tokens=100,
                max_ids=20,
                do_sample=True if lmpa_args.temperature > 0.01 else False,
                top_k=50,
                # top_p=0.95,
                temperature=lmpa_args.temperature if lmpa_args.temperature > 0.01 else None,
                max_func_strlen=lmpa_args.max_func_strlen,
                num_return_sequences=lmpa_args.num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True,
                suffix=""
            )
            prog_name = prog.prog_name
            func_name = func_to_ask.func_name
            # if (prog_name, func_name) in hints and lmpa_args.hint != "":
            #     hint_string = hints[prog_name, func_name]
            if prog_name in hints and func_name in hints[prog_name] and lmpa_args.hint != "":
                current_hints = hints[prog_name][func_name]
                
            else: 
                current_hints = None               
                # hint_string = ""
                # if lmpa_args.hint != "":
                #     print("Hint not found for %s %s" % (prog_name, func_name))
            ask_manager = AskFTManager(ask_config, func_to_ask, hints_entry=current_hints, tokenizer=None)
            prompt = ask_manager.ask_ids()
            ret_prompts.append({
                'prog_name': prog_name,
                'func_name': func_name,
                'func_body': func_to_ask.body,
                'rename_map': func_to_ask.rename_map,
                'var_id_maps': func_to_ask.var_id_maps,
                'func_id_maps': func_to_ask.func_id_maps,
                'hints': current_hints if current_hints is not None else {},
                'ask_str': prompt,
            })

    return ret_prompts
           


def main():
    parser = HfArgumentParser((LmpaArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        lmpa_args = parser.parse_yaml_file(
            json_file=os.path.abspath(sys.argv[1])[0]
        )
    else:
        lmpa_args = parser.parse_args_into_dataclasses()[0]
    if lmpa_args.interesting_only != "":
        interesting_only_file = json.load(open(lmpa_args.interesting_only, 'r'))
        interesting_only = set([(i[0], i[1]) for  i in interesting_only_file])
        print("Loaded %d interesting functions" % len(interesting_only))
    else:
        interesting_only = None

    with open(lmpa_args.progs, "rb") as f:
        progs = pickle.load(f)
    model_id = lmpa_args.model_id    
    


    # fout = open(lmpa_args.fout, 'w')
    # if fout exists, count existing lines, and append to last
    fout = open(lmpa_args.fout, 'a+')
    # check the length of the file
    fout.seek(0)
    lines = fout.readlines()
    fout.seek(0, 2)
    lmpa_args.from_idx += len(lines)
    print("Starting from %d" % lmpa_args.from_idx)
    

    prompts_all = []
    total_funcs = 0
    for prog in progs:
        total_funcs += len(prog.stripped_name2func)
    pbar = tqdm(total=min(total_funcs, lmpa_args.to_idx), desc="Generating prompts")
    if lmpa_args.hint != "":
        hints = json.load(open(lmpa_args.hint, 'r'))
        # hints = {}
        # hints_raw = [json.loads(l) for l in open(lmpa_args.hint, 'r')]
        # for entry in hints_raw:
        #     prog_name = entry['prog_name']
        #     func_name = entry['func_name']
        #     hints[prog_name, func_name] = entry['hints']
        print("Loaded %d hints" % len(hints))
    else:
        hints = {}

    for prog in progs:                        
        if pbar.n > lmpa_args.to_idx:
            break
        prompts = gen_prompts(
            lmpa_args=lmpa_args,
            prog=prog,
            pbar=pbar,
            hints=hints,
            interesting_only=interesting_only)
        prompts_all.extend(prompts)

    pbar.close()
    prompts_texts = [p['ask_str'] for p in prompts_all]
    print("In total %d prompts" % len(prompts_texts))
 
    llm = LLM(model=model_id, dtype=lmpa_args.dtype, swap_space=lmpa_args.swap_space, tensor_parallel_size=lmpa_args.tensor_parallel_size, gpu_memory_utilization=lmpa_args.gpu_util)
    print("Model Loaded")

    sampling_params = SamplingParams(
        temperature=lmpa_args.temperature,
        n=lmpa_args.num_return_sequences,
        top_k=50,
        seed=42,
        max_tokens=256)

    BZ = 1024
    for i in tqdm(range(0, len(prompts_texts), BZ), desc="Batch generation"):
        batch_prompt_texts = prompts_texts[i:i+BZ]
        batch_prompts = prompts_all[i:i+BZ]
        batch_outputs = llm.generate(batch_prompt_texts, sampling_params)
        for k, out_entry in enumerate(batch_outputs):
            completions = out_entry.outputs
            answers = [parse_answer(out.text) for out in completions]
            answer_scores = [(a, 0.2333) for a in answers]
            prompt = batch_prompts[k]
            ret_entry = {
                'prog_name': prompt['prog_name'],
                'func_name': prompt['func_name'],
                'func_body': prompt['func_body'],
                'rename_map': prompt['rename_map'],
                'var_id_maps': prompt['var_id_maps'],
                'func_id_maps': prompt['func_id_maps'],
                'hints': prompt['hints'],
                'ask_str': prompt['ask_str'],
                'answer_and_probs': answer_scores,
            }
            fout.write(json.dumps(ret_entry) + "\n")
            fout.flush()
            

        


if __name__ == "__main__":
    main()

