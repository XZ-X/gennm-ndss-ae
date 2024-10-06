#!/bin/bash -x

python3 evaluation/compare_varbert_and_gennm.py \
    --test_bins data/data_preproc/ghidra-O0-test.preproc.pkl \
    --ds_origin data/dataset_origin/C_CPP_binaries_O0.map.jsonl \
    --ds_overlap data/dataset_overlap/ghidra-O0-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/ghidra-O0-no-overlap-varbert.json \
    --training Alex-xu/gennm-ghidra-O0 \
    --ret_evaluated data/inference_results/ghidra-O0-gennm.jsonl \
    --varbert_evaluated data/inference_results/ghidra-O0-varbert.jsonl \
    --func_prefix FUN_