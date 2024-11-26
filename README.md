

# Artifact Evaluation for GenNm

[![DOI](https://zenodo.org/badge/867965814.svg)](https://doi.org/10.5281/zenodo.14220041)

This repository contains the artifact evaluation for our NDSS submission titled "Unleashing the Power of Generative Model in Recovering Variable Names from Stripped Binary". 

GenNm proposes a large language model based reverse engineering technique that recovers variable names from stripped binaries.
Specifically, it takes as input the decompiled code of a stripped bianry program.
Decompiled binary program has a syntax that is similar to the C programming language.
However, it does not contain meaningful variable names.
The names in the variables are just placeholders like `var_1`, `var_2`, etc.
GenNm aims to recover meaningful variable names for those variables.


## Instructions to Artifact Evaluation

We apply for the Available badge, the Functional badge, and the Reproduced badge.

### Available Badge

The README file contains detailed explanations for our artifact, and step-by-step instructions for running the artifact.
We will upload it to the Zenodo for a DOI after the major revision. Currently, we host the artifact on GitHub before finalizing it.

### Functional Badge

We provide all the implementations and scripts we use to run experiments of the paper.
We also include detailed explanations how each component works and how it is related to the paper.


### Reproduced Badge

Due to the essence of GenNm, it requires GPU to run the experiments from scratch.
Unfortunately, we are not able to provide GPU access for the reviewers.
Therefore, to demonstrate the functionality of GenNm, we uploaded intermediate data obtained from GenNm to Google Drive.
The reviewers can download the data and run the evaluation scripts to reproduce the main results in the paper.

We also provide all the scripts and detailed instructions to run GenNm from scratch.
We design a subset of our evaluation so that it can be finished on one A6000 GPU in 2 hours.
Optionally, the reviewers can observe similar results to the paper by running the scripts.


## Overview of the Artifact

Please download the data package from the following [Google Drive Link](https://drive.google.com/file/d/1XO7InFCgAowZacu3ZAhh2JIzBDmMsmgf).

And unzip the data package under the root directory of the artifact repo.

After that, the artifact repo should have the following structure:
```bash
README.md
reqruirements.txt
common/
evaluation/
inference/
preprocess/
scripts/
training_scripts/
data/
 |--data_preproc
 |--dataset_origin
 |--dataset_overlap
 |--inference_results
 |--model_ckpts
```

`README.md` is this file.

`README-detail.md` contains the detailed explanations for each component of GenNm.

`requirements.txt` contains the required dependencies.

`scripts/` contains the scripts for the artifact evaluation.

`common/` contains the common utility functions and data structures used in GenNm.

`evaluation/` contains the evaluation scripts for GenNm.

`inference/` contains the inference scripts to run GenNm.

`preprocess/` contains the preprocessing scripts to prepare the dataset for GenNm.

`training/` contains the scripts, configurations, and training scripts to train GenNm.

`data/` contains the data package for the artifact evaluation.

## Supported Environments

Ubuntu >=20.04. To run inference, you may need GPUs with at least 24GB VRAMs in total.
To run training, it is recommended to have GPUs with 96 GB VRAMs.

## Configuration

Please install `anaconda` (a [Python package manager](https://www.anaconda.com/download/success)) on your machine.
Then create a new environment and install the dependencies by running the following commands:

```bash
conda create -n gennm-artifact python=3.10
conda activate gennm-artifact
pip install -r requirements.txt
```

## Detailed Explanations of Each Component

Please refer to the `README-detail.md` for detailed explanations of each component of GenNm.



## Core Claims of the Paper

1. Table 1, GenNm outperforms the state-of-the-art technique, VarBERT, in terms of precision and recall.
2. Fig 13, GenNm can generalize to different decompilers and different optimization levels.
3. Fig 12, GenNm has better generalizability on names that are rarely seen in the training dataset.
4. Fig 11, GenNm outperforms VarBERT when evaluated by a GPT-Evaluator that mimics how a human would perceive the results.
5. Table 2, GenNm outperforms state-of-the-art black-box LLMs.


### Table 1

To reproduce the results in Table 1, please run the following script:
```bash
scripts/eval_1_compare_dirty.sh
```
This script will first load the output of both GenNm-2B and VarBERT.
Then it calculates the average precision and recall for both GenNm-2B and VarBERT.
The results should be in the following format:

```bash
               gennm_precision  gennm_recall  varbert_precision  varbert_recall
proj_in_train                                                                  
False                 0.305068      0.287518           0.235534        0.217368
True                  0.416923      0.395864           0.313501        0.296283
```

Each row denotes whether the function is in a project that is overlapped with the training dataset.
For example, the first row denotes the functions that are not in the training dataset.
`gennm_precision` and `gennm_recall` corresponding to the row `DIRTY-GenNm-CG-2B` and `Proj. NIT` columns of Table 1.
`varbert_precision` and `varbert_recall` corresponding to the row `DIRTY-VarBERT` and `Proj. NIT` columns of Table 1.

Similarly, the second row denotes the functions that are in the training dataset.
It corresponds to the row `DIRTY-GenNm-CG-2B` and `Proj. IT` columns of Table 1 and the row `DIRTY-VarBERT` and `Proj. IT` columns of Table 1, respectively.

Similarly, the following script print the performance for GenNm-CLM-7B.
```bash
scripts/eval_2_compare_dirty-7b.sh
```

Note that the results of VarBERT are printed again. They denote the same results for VarBERT as the previous script.
There might be minor differences (less than 0.005) than the results in the paper due to the randomness of the inference process.

To reproduce the rows for VarCorpus in Table 1, please run the following script:
```bash
scripts/eval_3_compare_ida-O0.sh
```
The results can be interpreted in the same way as the previous scripts.

### Fig 13

Fig.13 shows that GenNm outperforms VarBERT in different decompilers and optimization levels.
In Fig.13, the x-axis labels `In-PR` and `In-RC` denote the precision and recall for `proj_in_train=True`, 
and `Not-PR` and `Not-RC` denote the precision and recall for `proj_in_train=False`.

The following script computes the results for `IDA-O3` (the left sub-fig of Fig.13).

```bash
scripts/eval_4_compare_ida-O3.sh
```


The following script computes the results for `Ghidra-O0` (the right sub-fig) of Fig.13.
```bash
scripts/eval_5_compare_ghidra-O0.sh
```

The outputs of both scripts have the same format as the scripts for Table 1.

### Fig 12

Fig.12 shows that GenNm has better performance than the baseline for variables that are rarely seen during training.

The results can be reproduced by the following script:
```bash
scripts/eval_6_frequency.sh
```

It first computes the frequency of names in the training dataset, and aggregates the performance of both GenNm and the baseline by the name frequencies.
The output should look similar to the following:
```bash
Frequency range: 0, GenNM precision: 0.22675390887666336, VarBERT precision: 0.08463872255489022
Frequency range: 10, GenNM precision: 0.26418103974485163, VarBERT precision: 0.13218434545745872
Frequency range: 100, GenNM precision: 0.29191163604549425, VarBERT precision: 0.1817339238845144
Frequency range: 1000, GenNM precision: 0.2890527047785112, VarBERT precision: 0.20581014482224158
Frequency range: 999999, GenNM precision: 0.3734545184193896, VarBERT precision: 0.3100419129400394
```

The difference should be minor (less than 0.005) than the expected results.


### Fig 11

Fig.11 shows the performance of both GenNm and the baseline evaluated by GPT4.
We ask ChatGPT to evaluate each name by two questions, that is, Context Relevance (noted as Q-A in our scripts) and
Semantics Relevance (noted as Q-B in our scripts).

The following script reproduces the results:
```
scripts/eval_7_gpt4eval.sh
```

It outputs the distribution of the score for each question. The output looks similar to the following:
```bash
gennm_qa_score
1    189
2    139
3    319
4    257
5    728
Name: count, dtype: int64
gennm_qb_score
1    345
2    151
3    214
4    321
5    601
Name: count, dtype: int64
varbert_qa_score
1    325
2    157
3    405
4    215
5    530
Name: count, dtype: int64
varbert_qb_score
1    526
2    189
3    239
4    219
5    459
Name: count, dtype: int64
```

For example, `gennm_qa_score` denotes the aggregated scores of variables names generated by GenNm.
`1  189` denotes there are 189 names given the score `1`. 


### Table 2

Table 2 compares the performance of GenNm with black-box LLMs.
The results can be reproduced by the following script:

```bash
scripts/eval_8_compare_blackbox_llm.sh
```


### (Optional) Running inference from scratch

Please use the following script to run inference on a small subset of the DIRTY dataset.
It will run the inference process and print out the results at the end.
It takes less than 2 hours to finish on our machine with one A6000 GPU.
```bash
scripts/infer_1_generation.sh 
```

The expected results printed at the end should be similar to the following, with a variance of less than 0.01

```bash
               gennm_precision  gennm_recall  varbert_precision  varbert_recall
proj_in_train                                                                  
False                 0.338509      0.317039           0.259116        0.241244
True                  0.424468      0.404409           0.270418        0.279624
```


Similarly, the following script runs inference on a small subset of the VarCorpus dataset.
```bash
scripts/infer_2_generation-ida-O0.sh
```

The expected results are as follows, with a variance of less than 0.01:

```bash
               gennm_precision  gennm_recall  varbert_precision  varbert_recall
proj_in_train                                                                  
False                 0.386675      0.352520           0.286263        0.262375
True                  0.516616      0.498127           0.400037        0.390214
```

## Citation

Please cite our paper with the following BibTeX.

```text
@misc{xu2024gennm,
      title={Symbol Preference Aware Generative Models for Recovering Variable Names from Stripped Binary}, 
      author={Xiangzhe Xu and Zhuo Zhang and Zian Su and Ziyang Huang and Shiwei Feng and Yapeng Ye and Nan Jiang and Danning Xie and Siyuan Cheng and Lin Tan and Xiangyu Zhang},
      year={2024},
      eprint={2306.02546},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2306.02546}, 
}
```
