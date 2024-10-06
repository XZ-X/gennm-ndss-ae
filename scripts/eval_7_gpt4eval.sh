#!/bin/bash -x

python3 evaluation/compare_gpt4evaluator.py \
    --test_bins data/data_preproc/dirty-test.preproc.pkl \
    --ds_origin data/dataset_origin/dirty-ghcc.map.jsonl \
    --ds_overlap data/dataset_overlap/dirty-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/dirty-no-overlap-varbert.json \
    --training Alex-xu/gennm-dirty-dedup \
    --ret_evaluated data/inference_results/dirty-test5k-gpt4eval-gennm.jsonl \
    --varbert_evaluated data/inference_results/dirty-test5k-gpt4eval-varbert.jsonl \
    --is_dirty


