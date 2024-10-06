#!/bin/bash


echo "3-shot GPT-4"
python3 evaluation/compare_gennm_and_llm.py \
    --test_bins data/data_preproc/dirty-test.preproc.pkl \
    --ds_origin data/dataset_origin/dirty-ghcc.map.jsonl \
    --ds_overlap data/dataset_overlap/dirty-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/dirty-no-overlap-varbert.json \
    --training Alex-xu/gennm-dirty-dedup \
    --gennm_evaluated data/inference_results/dirty-test-gennm.jsonl \
    --llm_evaluated data/inference_results/3shot-gpt4.jsonl \
    --varbert_evaluated data/inference_results/dirty-test-varbert.jsonl \
    --is_dirty

echo "=================================="
echo "3-shot GPT-3.5"
python3 evaluation/compare_gennm_and_llm.py \
    --test_bins data/data_preproc/dirty-test.preproc.pkl \
    --ds_origin data/dataset_origin/dirty-ghcc.map.jsonl \
    --ds_overlap data/dataset_overlap/dirty-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/dirty-no-overlap-varbert.json \
    --training Alex-xu/gennm-dirty-dedup \
    --gennm_evaluated data/inference_results/dirty-test-gennm.jsonl \
    --llm_evaluated data/inference_results/3shot-gpt3.5.jsonl \
    --varbert_evaluated data/inference_results/dirty-test-varbert.jsonl \
    --is_dirty

echo "=================================="

echo "0-shot GPT-4"
python3 evaluation/compare_gennm_and_llm.py \
    --test_bins data/data_preproc/dirty-test.preproc.pkl \
    --ds_origin data/dataset_origin/dirty-ghcc.map.jsonl \
    --ds_overlap data/dataset_overlap/dirty-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/dirty-no-overlap-varbert.json \
    --training Alex-xu/gennm-dirty-dedup \
    --gennm_evaluated data/inference_results/dirty-test-gennm.jsonl \
    --llm_evaluated data/inference_results/zeroshot-gpt4.jsonl \
    --varbert_evaluated data/inference_results/dirty-test-varbert.jsonl \
    --is_dirty

echo "=================================="

echo "0-shot GPT-3.5"
python3 evaluation/compare_gennm_and_llm.py \
    --test_bins data/data_preproc/dirty-test.preproc.pkl \
    --ds_origin data/dataset_origin/dirty-ghcc.map.jsonl \
    --ds_overlap data/dataset_overlap/dirty-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/dirty-no-overlap-varbert.json \
    --training Alex-xu/gennm-dirty-dedup \
    --gennm_evaluated data/inference_results/dirty-test-gennm.jsonl \
    --llm_evaluated data/inference_results/zeroshot-gpt3.5.jsonl \
    --varbert_evaluated data/inference_results/dirty-test-varbert.jsonl \
    --is_dirty
