#!/bin/bash



echo "RND 0/3"

python3 inference/infer_vllm.py \
    --model_id data/model_ckpts/gennm-ida-O0-codegemma2b-1200 \
    --progs data/data_preproc/ida-O0-test.preproc.pkl \
    --fout data/ida-O0-tmp-out.jsonl \
    --to_idx 5000 \
    --num_return_sequences 1 \
    --temperature 0


python3 evaluation/parse_inference_ret.py --fin data/ida-O0-tmp-out.jsonl --fout data/ida-O0-tmp-out-parsed.jsonl
python3 inference/gen_hints.py --binary data/data_preproc/ida-O0-test.preproc.pkl  --model-ret data/ida-O0-tmp-out-parsed.jsonl --fout data/ida-O0-tmp-hints.jsonl


echo "RND 1/3"
python3 inference/infer_vllm.py \
    --model_id data/model_ckpts/gennm-ida-O0-codegemma2b-1200 \
    --progs data/data_preproc/ida-O0-test.preproc.pkl \
    --fout data/ida-O0-tmp-rnd1-out.jsonl \
    --to_idx 5000 \
    --num_return_sequences 1 \
    --hint data/ida-O0-tmp-hints.jsonl \
    --temperature 0

python3 evaluation/parse_inference_ret.py --fin data/ida-O0-tmp-rnd1-out.jsonl --fout data/ida-O0-tmp-rnd1-out-parsed.jsonl
python3 inference/gen_hints.py --binary data/data_preproc/ida-O0-test.preproc.pkl  --model-ret data/ida-O0-tmp-rnd1-out-parsed.jsonl --fout data/ida-O0-tmp-rnd1-hints.jsonl



echo "RND 2/3"
python3 inference/infer_vllm.py \
    --model_id data/model_ckpts/gennm-ida-O0-codegemma2b-1200 \
    --progs data/data_preproc/ida-O0-test.preproc.pkl \
    --fout data/ida-O0-tmp-rnd2-out.jsonl \
    --to_idx 5000 \
    --num_return_sequences 1 \
    --hint data/ida-O0-tmp-rnd1-hints.jsonl \
    --temperature 0

python3 evaluation/parse_inference_ret.py --fin data/ida-O0-tmp-rnd2-out.jsonl --fout data/ida-O0-tmp-rnd2-out-parsed.jsonl
python3 inference/gen_hints.py --binary data/data_preproc/ida-O0-test.preproc.pkl  --model-ret data/ida-O0-tmp-rnd2-out-parsed.jsonl --fout data/ida-O0-tmp-rnd2-hints.jsonl


echo "RND 3/3"
python3 inference/infer_vllm.py \
    --model_id data/model_ckpts/gennm-ida-O0-codegemma2b-1200 \
    --progs data/data_preproc/ida-O0-test.preproc.pkl \
    --fout data/ida-O0-tmp-rnd3-out.jsonl \
    --to_idx 5000 \
    --num_return_sequences 1 \
    --hint data/ida-O0-tmp-rnd2-hints.jsonl \
    --temperature 0

python3 evaluation/parse_inference_ret.py --fin data/ida-O0-tmp-rnd3-out.jsonl --fout data/ida-O0-tmp-rnd3-out-parsed.jsonl
python3 inference/gen_hints.py --binary data/data_preproc/ida-O0-test.preproc.pkl  --model-ret data/ida-O0-tmp-rnd3-out-parsed.jsonl --fout data/ida-O0-tmp-rnd3-hints.jsonl


echo "data/ida-O0-tmp-out-parsed.jsonl" > data/ida-O0-tmp-namelist.txt
echo "data/ida-O0-tmp-rnd1-out-parsed.jsonl" >> data/ida-O0-tmp-namelist.txt
echo "data/ida-O0-tmp-rnd2-out-parsed.jsonl" >> data/ida-O0-tmp-namelist.txt
echo "data/ida-O0-tmp-rnd3-out-parsed.jsonl" >> data/ida-O0-tmp-namelist.txt

python3 inference/combine_names.py --name_list data/ida-O0-tmp-namelist.txt --fout data/ida-O0-tmp-name-combined.jsonl


echo "Parsing dataset"
python3 preprocess/extract_main.py --ds-bin-in data/data_preproc/ida-O0-test.preproc.pkl --fout data/data_preproc/dirty-test.parsed.pkl

echo "Validating names"

python3 inference/prop_names.py \
  --ds-in data/data_preproc/ida-O0-test.preproc.pkl \
  --parsed-in data/data_preproc/dirty-test.parsed.pkl \
  --default_name data/ida-O0-tmp-rnd3-out-parsed.jsonl \
  --names data/ida-O0-tmp-name-combined.jsonl \
  --fout data/ida-O0-tmp-name-prop.jsonl

python3 evaluation/compare_varbert_and_gennm.py \
    --test_bins data/data_preproc/ida-O0-test.preproc.pkl \
    --ds_origin data/dataset_origin/C_CPP_binaries_O0.map.jsonl \
    --ds_overlap data/dataset_overlap/ida-O0-no-overlap.json \
    --ds_overlap_varbert_format data/dataset_overlap/ida-O0-no-overlap-varbert.json \
    --training Alex-xu/gennm-ida-O0 \
    --ret_evaluated data/ida-O0-tmp-name-prop.jsonl \
    --varbert_evaluated data/inference_results/ida-O0-varbert.jsonl