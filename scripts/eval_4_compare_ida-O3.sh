#!/bin/bash -x

python3 evaluation/compare_varbert_and_gennm.py \
    --test_bins data/data_preproc/ida-O3-test.preproc.pkl \
    --ds_origin data/dataset_origin/C_CPP_binaries_O3.map.jsonl \
    --ds_overlap data/dataset_overlap/ida-O3-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/ida-O3-no-overlap-varbert.json \
    --training Alex-xu/gennm-ida-O3 \
    --ret_evaluated data/inference_results/ida-O3-gennm.jsonl \
    --varbert_evaluated data/inference_results/ida-O3-varbert.jsonl