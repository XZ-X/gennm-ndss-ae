#!/bin/bash -x

python3 evaluation/compare_varbert_and_gennm_freq.py \
    --test_bins data/data_preproc/ida-O0-test.preproc.pkl \
    --ds_origin data/dataset_origin/C_CPP_binaries_O0.map.jsonl \
    --ds_overlap data/dataset_overlap/ida-O0-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/ida-O0-no-overlap-varbert.json \
    --training Alex-xu/gennm-ida-O0 \
    --ret_evaluated data/inference_results/ida-O0-gennm.jsonl \
    --varbert_evaluated data/inference_results/ida-O0-varbert.jsonl