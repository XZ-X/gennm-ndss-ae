import json
import sentencepiece as spm
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from utils import preprocess, lemmatize_name_tokens
from tqdm import tqdm
import argparse


sp = spm.SentencePieceProcessor()
sp.load("evaluation/tmp.segmentation.model")

lem = WordNetLemmatizer()


word_cluster = json.load(open("evaluation/word_cluster.json", "r"))


word_cluster_fast = {}
for word, cluster in word_cluster.items():
    word_cluster_fast[word] = set()
    for idx in cluster:
        word_cluster_fast[word].add(idx)


def tokenize_name(name):
    preprocessed_name = preprocess(name)
    name_tokens = preprocessed_name.split()
    result_name_tokens = lemmatize_name_tokens(name_tokens)
    split = sp.encode_as_pieces(" ".join(result_name_tokens))    
    ret = []
    for w in split:
        if w.startswith("\u2581"):
            w = w[1:]
        ret.append(w)
    return ret


def _same_token(gt, pred):
    if gt == pred:
        return True
    shorter = gt if len(gt) < len(pred) else pred
    longer = gt if len(gt) >= len(pred) else pred
    if len(shorter) >= 2 and longer.startswith(shorter):
        return True
    if len(shorter) >= 2 and longer.endswith(shorter):
        return True    
    
    if gt in word_cluster_fast and pred in word_cluster_fast:
        return len(word_cluster_fast[gt] & word_cluster_fast[pred]) > 0
    return False


def score_name(gt, pred, dbg=False):
    # return precision and recall
    gt = gt.lower()
    if type(pred) != str:
        pred = str(pred)
    pred = pred.lower()
    if gt == pred:
        if dbg:
            return 1, 1, "exact match"
        else:
            return 1, 1
    gt_tokens = tokenize_name(gt)
    if len(gt_tokens) == 0:
        print("gt_tokens is empty for gt = {}".format(gt))
        return 0, 0
                
    pred_tokens = tokenize_name(pred)
    if len(pred_tokens) == 0:
        print("pred_token is empty, pred = %s, gt = %s" % (pred, gt))
        return 0, 0
    matched_gt = set()
    matched_pred = set()
    for gt_t in gt_tokens:
        for pred_t in pred_tokens:
            if _same_token(gt_t, pred_t):
                matched_gt.add(gt_t)
                matched_pred.add(pred_t)
    precision = len(matched_pred) / len(pred_tokens)
    recall = len(matched_gt) / len(gt_tokens)
    if dbg:
        return precision, recall, {
                'matched_gt': matched_gt,
                'matched_pred': matched_pred,
                'gt_tokens': gt_tokens,
                'pred_tokens': pred_tokens
            }
    else:
        return precision, recall
            

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
    return dp[m][n]


def _same_token_ori(gt, pred):
    if gt == pred:
        return True
    shorter = gt if len(gt) < len(pred) else pred
    longer = gt if len(gt) >= len(pred) else pred
    if longer.startswith(shorter):
        return True
    
    if gt in word_cluster_fast and pred in word_cluster_fast:
        return len(word_cluster_fast[gt] & word_cluster_fast[pred]) > 0
    
    if len(shorter) > 1 and shorter[0] == longer[0]:
        d = edit_distance(shorter, longer)
        dr = d / len(longer)
        if dr < 1/3:
            return True

    return False     


def score_name_ori(gt, pred, dbg=False):
    # return precision and recall
    gt = gt.lower()
    pred = pred.lower()
    if gt == pred:
        if dbg:
            return 1, 1, "exact match"
        else:
            return 1, 1
    gt_tokens = tokenize_name(gt)
    if len(gt_tokens) == 0:
        print("gt_tokens is empty for gt = {}".format(gt))
        return 0, 0
                
    pred_tokens = tokenize_name(pred)
    if len(pred_tokens) == 0:
        print("pred_token is empty, pred = %s, gt = %s" % (pred, gt))
        return 0, 0
    matched_gt = set()
    matched_pred = set()
    for gt_t in gt_tokens:
        for pred_t in pred_tokens:
            if _same_token_ori(gt_t, pred_t):
                matched_gt.add(gt_t)
                matched_pred.add(pred_t)
    precision = len(matched_pred) / len(pred_tokens)
    recall = len(matched_gt) / len(gt_tokens)
    if dbg:
        return precision, recall, {
                'matched_gt': matched_gt,
                'matched_pred': matched_pred,
                'gt_tokens': gt_tokens,
                'pred_tokens': pred_tokens
            }
    else:
        return precision, recall
            
