#!/bin/bash



echo "RND 0/3"

python3 inference/infer_vllm.py \
    --model_id data/model_ckpts/gennm-codegemma2b-600 \
    --progs data/data_preproc/dirty-test.preproc.pkl \
    --fout data/tmp-out.jsonl \
    --to_idx 5000 \
    --num_return_sequences 1 \
    --temperature 0


python3 evaluation/parse_inference_ret.py --fin data/tmp-out.jsonl --fout data/tmp-out-parsed.jsonl
python3 inference/gen_hints.py --binary data/data_preproc/dirty-test.preproc.pkl  --model-ret data/tmp-out-parsed.jsonl --fout data/tmp-hints.jsonl


echo "RND 1/3"
python3 inference/infer_vllm.py \
    --model_id data/model_ckpts/gennm-codegemma2b-600 \
    --progs data/data_preproc/dirty-test.preproc.pkl \
    --fout data/tmp-rnd1-out.jsonl \
    --to_idx 5000 \
    --num_return_sequences 1 \
    --hint data/tmp-hints.jsonl \
    --temperature 0

python3 evaluation/parse_inference_ret.py --fin data/tmp-rnd1-out.jsonl --fout data/tmp-rnd1-out-parsed.jsonl
python3 inference/gen_hints.py --binary data/data_preproc/dirty-test.preproc.pkl  --model-ret data/tmp-rnd1-out-parsed.jsonl --fout data/tmp-rnd1-hints.jsonl



echo "RND 2/3"
python3 inference/infer_vllm.py \
    --model_id data/model_ckpts/gennm-codegemma2b-600 \
    --progs data/data_preproc/dirty-test.preproc.pkl \
    --fout data/tmp-rnd2-out.jsonl \
    --to_idx 5000 \
    --num_return_sequences 1 \
    --hint data/tmp-rnd1-hints.jsonl \
    --temperature 0

python3 evaluation/parse_inference_ret.py --fin data/tmp-rnd2-out.jsonl --fout data/tmp-rnd2-out-parsed.jsonl
python3 inference/gen_hints.py --binary data/data_preproc/dirty-test.preproc.pkl  --model-ret data/tmp-rnd2-out-parsed.jsonl --fout data/tmp-rnd2-hints.jsonl


echo "RND 3/3"
python3 inference/infer_vllm.py \
    --model_id data/model_ckpts/gennm-codegemma2b-600 \
    --progs data/data_preproc/dirty-test.preproc.pkl \
    --fout data/tmp-rnd3-out.jsonl \
    --to_idx 5000 \
    --num_return_sequences 1 \
    --hint data/tmp-rnd2-hints.jsonl \
    --temperature 0

python3 evaluation/parse_inference_ret.py --fin data/tmp-rnd3-out.jsonl --fout data/tmp-rnd3-out-parsed.jsonl
python3 inference/gen_hints.py --binary data/data_preproc/dirty-test.preproc.pkl  --model-ret data/tmp-rnd3-out-parsed.jsonl --fout data/tmp-rnd3-hints.jsonl


echo "data/tmp-out-parsed.jsonl" > data/tmp-namelist.txt
echo "data/tmp-rnd1-out-parsed.jsonl" >> data/tmp-namelist.txt
echo "data/tmp-rnd2-out-parsed.jsonl" >> data/tmp-namelist.txt
echo "data/tmp-rnd3-out-parsed.jsonl" >> data/tmp-namelist.txt

python3 inference/combine_names.py --name_list data/tmp-namelist.txt --fout data/tmp-name-combined.jsonl


echo "Parsing dataset"
python3 preprocess/extract_main.py --ds-bin-in data/data_preproc/dirty-test.preproc.pkl --fout data/data_preproc/dirty-test.parsed.pkl

echo "Validating names"

python3 inference/prop_names.py \
  --ds-in data/data_preproc/dirty-test.preproc.pkl \
  --parsed-in data/data_preproc/dirty-test.parsed.pkl \
  --default_name data/tmp-rnd3-out-parsed.jsonl \
  --names data/tmp-name-combined.jsonl \
  --fout data/tmp-name-prop.jsonl

python3 evaluation/compare_varbert_and_gennm.py \
    --test_bins data/data_preproc/dirty-test.preproc.pkl \
    --ds_origin data/dataset_origin/dirty-ghcc.map.jsonl \
    --ds_overlap data/dataset_overlap/dirty-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/dirty-no-overlap-varbert.json \
    --training Alex-xu/gennm-dirty-dedup \
    --ret_evaluated data/tmp-name-prop.jsonl \
    --varbert_evaluated data/inference_results/dirty-test-varbert.jsonl \
    --is_dirty