import re
import cxxfilt

# v1 v2 v4...
dummy_var_pattern = re.compile(r'v\d+')
# a1 a2 a4...
dummy_arg_pattern = re.compile(r'a\d+')

dummy_funcs = set(
    [
        '__libc_csu_init',
        '__libc_csu_fini',
        '_start',
        # 'main',
        '__libc_start_main',
        '__gmon_start__',
        '__cxa_finalize',
        '__cxa_atexit',
        'frame_dummy',        
        '_dl_relocate_static_pie',
        'deregister_tm_clones',
        'register_tm_clones',
        'nullsub',
        'nullsub_1',
        '.init_proc'
    ]
)

dummy_func_patterns = [
    re.compile(r'__do_global_dtors_.*'), 
    re.compile(r'__do_global_ctors_.*'),
    re.compile(r'\..*'),
    re.compile(r'__.*'),
]

def _is_interesting_function(name):
    if type(name) != str:
        return False
    if name in dummy_funcs:
        return False
    for pattern in dummy_func_patterns:
        if pattern.match(name):
            return False
    return True



def is_dummy_name(name):
    if name.strip() == '':
        return True
    return dummy_var_pattern.match(name) or dummy_arg_pattern.match(name)


def is_trivial_name(name):
    return name in set(['i', 'j', 'k', 'ret'])

def is_interesting_name(name):
    return not is_dummy_name(name) and not is_trivial_name(name)


def replace_variable_names(code, ori_variable_name, new_variable_name):
  variable_name_pattern = re.compile(
          r'([^a-zA-Z0-9_@]|^)(%s)([^a-zA-Z0-9_@])' % ori_variable_name)
  return variable_name_pattern.sub(
      r'\g<1>%s\g<3>'%new_variable_name, code)
  

def prepare_func_str(func_entry, vars_to_rename, max_len=10240):    
    current_str = func_entry['stripped_code']
    for k, v in func_entry['id_maps'].items():
        if k in vars_to_rename:
            continue
        current_str = replace_variable_names(
            current_str, k, v)
            
        current_str = current_str[:max_len]
    return current_str



def is_interesting_func(func_entry):
    gt_func = func_entry['b']
    stripped_func = func_entry['c']    
    gt_name = gt_func['n']
    if not _is_interesting_function(gt_name):
        # dummy funcs inserted by compiler
        return False
    if len(stripped_func['c'].split('\n')) < 4:
        # function with empty body
        return False
    
    return True


demangled_func_name_pattern = re.compile(r'.*([^0-9A-Za-z_~]|^)([~0-9a-zA-Z_]+)(<.*>)?(\[abi.*\])?\(.*\)')
demangled_operator_name_pattern = re.compile(r'.*([^0-9A-Za-z_~]|^)(operator ?[^0-9a-zA-Z_]+|new|new\[\]|delete|delete\[\])(<.*>)?(\[abi.*\])?\(.*\)')

def try_demangle(name, silent=False):
  try:
    demangled_name = cxxfilt.demangle(name)  
  except:
    if name.endswith('_0') or name.endswith('_1') or name.endswith('_2') or name.endswith('_3'):
      name = name[:-2]
    try:
        demangled_name = cxxfilt.demangle(name)
    except:
        return name
  if name != demangled_name:    
    matched = demangled_func_name_pattern.match(demangled_name)
    if matched:
      return matched.group(2)
    else:
      matched = demangled_operator_name_pattern.match(demangled_name)
      if matched:
        return matched.group(2)
    if not silent:
        print("Error parsing demangled name: %s" % demangled_name)
        print("Original name: %s" % name)
    return name
    # raise Exception("Error parsing demangled name!")        
  else:
    return name
  
