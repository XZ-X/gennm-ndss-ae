# isort: off
import sys
sys.path.append("common")
# isort: on

import re
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import cxxfilt

lem = WordNetLemmatizer()

# v1 v2 v4...
dummy_var_pattern = re.compile(r"\bv\d+\b")
# a1 a2 a4...
dummy_arg_pattern = re.compile(r"\ba\d+\b")

dummy_var_pattern2 = re.compile(r"\b[_0-9]+\b")


def is_dummy_name(name):
    if name.strip() == "":
        return True
    return (
        # when whole name is a dummy name        
        dummy_var_pattern.match(name)
        or dummy_arg_pattern.match(name)
        or dummy_var_pattern2.match(name)
    )


def is_trivial_name(name):
    return name in set(["i", "j", "k", "ret"])


def is_interesting_name(name):
    if name.strip() == "":
        return False

    return not is_dummy_name(name) and not is_trivial_name(name)


def preprocess(name):
    """
    Preprocess function name by:
        - tokenize whole name into words
        - remove digits
    """

    # split whole name into words and remove digits
    name = name.replace("_", " ")
    tmp = ""
    for c in name:
        if (
            not c.isalpha()
        ):  # filter out numbers and other special characters, e.g. '_' and digits
            tmp = tmp + " "
        elif c.isupper():
            tmp = tmp + " " + c
        else:
            tmp = tmp + c
    tmp = tmp.strip()
    tmp = tmp.split()

    res = []
    i = 0
    while i < len(tmp):
        cap = ""
        t = tmp[i]

        # handle series of capital letters: e.g., SHA, MD
        while i < len(tmp) and len(tmp[i]) == 1:
            cap = cap + tmp[i]
            i += 1
        if len(cap) == 0:
            res.append(t)
            i += 1
        else:
            res.append(cap)

    res_lower = " ".join(res).lower()
    return res_lower


def get_pos(treebank_tag):
    """
    get the pos of a treebank tag
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None  # for easy if-statement


def lemmatize_name_tokens(name_tokens):
    name_tokens_tagged = nltk.pos_tag(name_tokens)
    result_name_tokens = []
    for word, tag in name_tokens_tagged:
        wordnet_tag = get_pos(tag)
        if wordnet_tag is None:
            word = lem.lemmatize(word)
        else:
            word = lem.lemmatize(word, pos=wordnet_tag)
        result_name_tokens.append(word)
    return result_name_tokens


def replace_variable_names(code, ori_variable_name, new_variable_name):
  variable_name_pattern = re.compile(
          r'([^a-zA-Z0-9_]|^)(%s)([^a-zA-Z0-9_])' % ori_variable_name)
  return variable_name_pattern.sub(
      r'\g<1>%s\g<3>'%new_variable_name, code)


demangled_func_name_pattern = re.compile(r'.*([^0-9A-Za-z_~]|^)([~0-9a-zA-Z_]+)(<.*>)?(\[abi.*\])?\(.*\)')
demangled_operator_name_pattern = re.compile(r'.*([^0-9A-Za-z_~]|^)(operator ?[^0-9a-zA-Z_]+|new|new\[\]|delete|delete\[\])(<.*>)?(\[abi.*\])?\(.*\)')

def try_demangle(name):
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
    
    print("Error parsing demangled name: %s" % demangled_name)
    print("Original name: %s" % name)
    return name
    # raise Exception("Error parsing demangled name!")        
  else:
    return name
  
