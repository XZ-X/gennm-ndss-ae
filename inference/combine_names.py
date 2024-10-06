import json
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_list", type=str, default="name_ret_list.txt")
    parser.add_argument("--fout", type=str, default="combined-name-list.jsonl")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    name_list = [l.strip() for l in open(args.name_list, "r").readlines()]
    seen_files = set()
    ret_name_map = {}
    for fname in tqdm(name_list, desc="Combining names"):
        if fname in seen_files:
            continue
        seen_files.add(fname)
        fin = open(fname, "r").readlines()
        data = [json.loads(l.strip()) for l in fin]
        for entry in data:
            prog_name = entry["prog_name"]
            func_name = entry["func_name"]
            varname = entry["varname"]
            if (prog_name, func_name, varname) not in ret_name_map:
                ret_name_map[(prog_name, func_name, varname)] = []
            gt_varname = entry["gt_varname"]
            ret_entry = {
                "gt_varname": entry["gt_varname"],
                "pred_name": entry["pred_name"],
                "precision": entry["precision"],
                "recall": entry["recall"],
                "from": fname,
            }
            ret_name_map[(prog_name, func_name, varname)].append(ret_entry)

    fout = open(args.fout, "w")
    for key, val in tqdm(ret_name_map.items(), desc="Writing"):
        prog_name, func_name, varname = key
        entry = {
            "prog_name": prog_name,
            "func_name": func_name,
            "varname": varname,            
            "name_list": str(val),
        }
        fout.write(json.dumps(entry) + "\n")
    fout.close()
    print()


if __name__ == "__main__":
    main()
