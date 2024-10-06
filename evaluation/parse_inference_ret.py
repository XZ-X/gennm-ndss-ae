# isort: off
import sys
sys.path.append("common")
# isort: on
import argparse
import json
import utils
import eval_utils
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--fin', type=str, default='')
parser.add_argument('--fout', type=str, default='')
parser.add_argument('--best', action='store_true')
parser.add_argument('--ori', action='store_true')
parser.add_argument('--topk', type=int, default=3)
parser.add_argument('--force-parse', action='store_true')
parser.add_argument('--max-sample', type=int, default=999999999)
parser.add_argument('--case-study', action='store_true')

args = parser.parse_args()
# # # XXX: for debug
# args.force_parse = True

def evaluate_batch(data):
  ret = []
  error_cnt = 0
  total_cnt = 0
  fixed_cnt = 0  
  dbg_error_entries = []
  for entry in tqdm(data):
      prog_name = entry['prog_name']      
      func_name = entry['func_name']
      if not args.case_study:
        if not func_name.startswith('sub_') and not func_name.startswith('FUN_'):
          continue
      answer_and_probs = entry['answer_and_probs']
      # "transpose" answer_and_probs
      ids_answer = {}
      for answer, prob in answer_and_probs:
        total_cnt += 1
        # dbg
        if type(answer) != dict:
          error_cnt += 1
          dbg_error_entries.append((entry, answer))
          continue
        if 'Error' in answer:
          error_cnt += 1
          dbg_error_entries.append((entry, answer))
          if not args.force_parse:
            continue
          else:
            parsed = None
            # for dbg
            ori_answer_str = answer['Error']            
            # find last {
            last_left_bracket = answer['Error'].rfind('{')
            # find last }
            last_right_bracket = answer['Error'].rfind('}')
            if last_left_bracket != -1 and last_right_bracket != -1:
              try_recover_str = answer['Error'][last_left_bracket:last_right_bracket+1]
              answer['Error'] = try_recover_str
            elif last_left_bracket != -1:
              try_recover_str = answer['Error'][last_left_bracket:]
              answer['Error'] = try_recover_str
            try:
              parsed = eval(answer['Error'])
            except:
              pass            
            if parsed is None:
              if len(answer['Error'].strip()) == 0:
                continue
              if answer['Error'].strip()[-1] == '}':
                first_colon = answer['Error'].find(':')
                if first_colon == -1:
                  continue                                
                # skip the first few
                first_comma = answer['Error'].find(',')
                try_recover_str = answer['Error']
                while first_comma != -1:
                  next_comma = answer['Error'].find(',', first_comma + 1)
                  try_recover_str = answer['Error'][first_comma + 1:]
                  if next_comma != -1 and next_comma > first_colon:
                    break
                  first_comma = next_comma
                if try_recover_str.strip()[0] != "'":
                  try_recover_str = "'"+try_recover_str.strip()
                try_recover_str = '{'+try_recover_str.strip()
              else:
                # remove the last few
                last_comma = answer['Error'].rfind(',')
                if last_comma != -1:
                  try_recover_str = answer['Error'][:last_comma]
                  try_recover_str = try_recover_str.strip()+'}'              
              try:
                parsed = eval(try_recover_str)
              except:
                # print("1: Try to recover %s from %s failed" % (try_recover_str, answer['Error']))
                pass
            if parsed is None:
              if '{' in answer['Error'] and '}' in answer['Error']:              
                left_idx = answer['Error'].find('{')
                right_idx = answer['Error'].rfind('}')
                try_recover_str = answer['Error'][left_idx:right_idx+1]              
              try:
                parsed = eval(try_recover_str)
              except:
                # print("2: Try to recover %s from %s failed" % (try_recover_str, answer['Error']))
                continue
            answer = parsed
            if type(answer) != dict:              
              continue
            fixed_cnt += 1
        for old_name, new_name in answer.items():
          if old_name not in ids_answer:
            ids_answer[old_name] = []
          ids_answer[old_name].append((new_name, prob))
      entry['vars_answer'] = ids_answer
      for varname, gt_varname in list(entry['func_id_maps'].items()) + list(entry['var_id_maps'].items()):
        gt_varname = utils.try_demangle(gt_varname)
        if not utils.is_interesting_name(gt_varname):
          continue
        if gt_varname == varname:
          continue
        if varname not in entry['vars_answer']:
          # pred_name = "<emptyname>"
          continue
        elif len(entry['vars_answer'][varname]) == 0:
          pred_name = "<emptyname>"
        else:
          if args.best:            
            highest_pr = -1
            highest_recall = -1
            current_pred_name = "<emptyname>"
            all_names_sorted = sorted(entry['vars_answer'][varname], key=lambda x: x[1], reverse=True)
            for pred_name, _ in all_names_sorted[:args.topk]:
              score = eval_utils.score_name(gt_varname, pred_name)
              if score[0] >= highest_pr and score[1] >= highest_recall:
                highest_pr = score[0]
                highest_recall = score[1]
                current_pred_name = pred_name
            pred_name = current_pred_name
            score = (highest_pr, highest_recall)
          else:            
            pred_name = entry['vars_answer'][varname][0][0]
            if args.ori:
              score = eval_utils.score_name_ori(gt_varname, pred_name)
            else:
              score = eval_utils.score_name(gt_varname, pred_name)
        if pred_name == "<emptyname>":
          score = (0, 0)
        ret.append({
          'prog_name': prog_name,
          'func_name': func_name,
          'varname': varname,
          'gt_varname': gt_varname,
          'pred_name': pred_name,
          'precision': score[0],
          'recall': score[1],
        })

  return ret, error_cnt, total_cnt, dbg_error_entries, fixed_cnt


data_raw = open(args.fin, 'r').readlines()
data = []
for entry_str in data_raw[:args.max_sample]:
  try:
    entry = json.loads(entry_str)
    data.append(entry)
  except:
    print("Error parsing json:", entry_str)
    continue

NUM_THREADS = 8
data_segs = []
for i in tqdm(range(NUM_THREADS)):
  data_segs.append(data[i::NUM_THREADS])



# multi-threaded
from multiprocessing import Pool

pool = Pool(NUM_THREADS)
rets = list(tqdm(pool.imap_unordered(evaluate_batch, data_segs), total=NUM_THREADS))

rets_all = []
error_cnt = 0
total_cnt = 0
total_fixed_cnt = 0
dbg_error_entries_all = []
for r, ec, tc, dbg_err_entries, fixed_cnt in rets:
  rets_all.extend(r)
  error_cnt += ec
  total_cnt += tc
  total_fixed_cnt += fixed_cnt
  dbg_error_entries_all.extend(dbg_err_entries)

print("Error cnt:", error_cnt)
if error_cnt > 0:
  print("Ratio of error %f=%d/%d" % (error_cnt / total_cnt, error_cnt, total_cnt))
  print("Fixed cnt:", total_fixed_cnt)
  print("Ratio of fixed %f=%d/%d" % (total_fixed_cnt / error_cnt, total_fixed_cnt, error_cnt))

print("Writing to file...")
fout = open(args.fout, 'w')
for line in tqdm(rets_all):
  fout.write(json.dumps(line) + '\n')
fout.close()

print("Writing to dbg file...")
fout = open(args.fout + '.dbg', 'w')
for entry, answer in tqdm(dbg_error_entries_all):
  fout.write(json.dumps({
    'entry': entry,
    'answer': answer,
  }) + '\n')
fout.close()


print()