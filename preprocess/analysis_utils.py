# isort: off
import sys

sys.path.append("common")
# isort: on
import name_utils


def parser_sanitize_for_ida(body):
    subs_map = {
        "__int64": "long",
        "__int32": "int",
        "__int16": "short",
        "__int8": "char",
        "__int8": "char",
        "_QWORD": "long",
        "_DWORD": "int",
        "_WORD": "short",
        "_BYTE": "char",
        "QWORD": "long",
        "DWORD": "int",
        "WORD": "short",
        "BYTE": "char",
        "__fastcall": "",
        "__cdecl": "",
        "__stdcall": "",
        "__thiscall": "",        
        "__noreturn": "",
    }
    new_body = body
    for k, v in subs_map.items():
        if k in new_body:
            new_body = name_utils.replace_variable_names(new_body, k, v)
    return new_body

def parser_sanitize_for_ghidra(body):
    subs_map = {
        "undefined8": "long",
        "undefined4": "int",
        "undefined2": "short",
        "undefined1": "char",
        "undefined": "char",
        "code": "void",  
        "byte": "char",
        "word": "short",
        "dword": "int",
        "qword": "long",              
        "__int64": "long",
        "__int32": "int",
        "__int16": "short",
        "__int8": "char",
        "__int8": "char",
        "_QWORD": "long",
        "_DWORD": "int",
        "_WORD": "short",
        "_BYTE": "char",
        "QWORD": "long",
        "DWORD": "int",
        "WORD": "short",
        "BYTE": "char",
        "__fastcall": "",
        "__cdecl": "",
        "__stdcall": "",
        "__thiscall": "",        
        "__noreturn": "",
    }
    new_body = body
    for k, v in subs_map.items():
        if k in new_body:
            new_body = name_utils.replace_variable_names(new_body, k, v)
    return new_body