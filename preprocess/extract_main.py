# isort: off
import sys

sys.path.append("common")
# isort: on
import argparse
import sys
from tqdm import tqdm
import pickle
import tree_sitter_utils as ts_utils
from tree_sitter import Language, Parser
import tree_sitter_cpp
import lmpa_ir
from analysis_utils import parser_sanitize_for_ida, parser_sanitize_for_ghidra
from lmpa_parser import LmpaParser
import multiprocessing


argparser = argparse.ArgumentParser(description='generate analysis for dataset')
argparser.add_argument('--ds-bin-in', type=str, default='', help='dataset path')
argparser.add_argument('--mode', type=str, default='ida', help='mode')
argparser.add_argument('--fout', type=str, default='', help='analysis-out')

args = argparser.parse_args()

if args.mode == '':
    if 'ghidra' in args.ds_bin_in:
        args.mode = 'ghidra'
    elif 'ida' in args.ds_bin_in:
        args.mode = 'ida'
    else:
        args.mode = 'ida'


CPP_LANGUAGE = Language(tree_sitter_cpp.language())
parser = Parser(CPP_LANGUAGE)


binaries = pickle.load(open(args.ds_bin_in, 'rb'))
prog_name2bin = {bin_prog.prog_name: bin_prog for bin_prog in binaries}

functions = []
for bin_prog in tqdm(binaries, desc='loading binaries'):
    for stripped_name, func in bin_prog.stripped_name2func.items():
        if args.mode == 'ida':
            sanitized_body = parser_sanitize_for_ida(func.body)
        else:
            sanitized_body = parser_sanitize_for_ghidra(func.body)
        functions.append((bin_prog.prog_name, stripped_name, func, sanitized_body))


def parse_one_function(entry):
    prog_name, func_name, func, body = entry
    root = parser.parse(bytes(body, 'utf-8'))
    params = ts_utils.find_all_recursively(root.root_node, 'parameter_declaration')
    lmpa_params = []
    for param in params:
        param_name = ts_utils.get_first_opt(param, 'identifier')
        if param_name is not None:            
            param_name = param_name.text.decode('utf-8')
            lmpa_params.append(lmpa_ir.LmPaVarExpression(var_name=param_name))
        else:
            lmpa_params.append(lmpa_ir.LmPaVarExpression(var_name='unknown'))
    
    lmpa_parser = LmpaParser()
    lmpa_parser.parse_body(root)
    return {
        'prog_name': prog_name,
        'func_name': func_name,
        'sanitized_body': body,
        'lmpa_args': lmpa_params,
        'lmpa_ir': lmpa_parser.all_lmpa_ir
    }

pool = multiprocessing.Pool(16)
rets = pool.imap_unordered(parse_one_function, tqdm(functions, desc='parsing functions'))

ret_all = []
for ret in rets:
    ret_all.append(ret)

pool.close()

pickle.dump(ret_all, open(args.fout, 'wb'))



print()