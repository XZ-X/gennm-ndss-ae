# Detailed Explanations about the Artifact

## Preprocssing

We use IDA to extract the call graph of a binary program.
For the DIRTY dataset, we reuse their preprocessing pipeline to generate the decompiled code and the ground truth variable names.
For the VarCorpus dataset, we directly use the preprocessed data from the VarBERT authors.
Our artifact pipline does not inlcude the aforementioned steps.

The input to our pipeline is the `pickle` file containing a list of binary program objects (as defined in `common/binary_prog.py:BinaryProgram`).
For each binary program, it contains a list of decompiled functions and the call graph.
Each decompiled function (`common/binary_prog.py:Function`) contains the decompiled code and a map from the placeholder variable names to the ground truth variable names.

Our preprocessing contains three key steps:
1. Deduplicating binary programs.
2. Identifying potential overlapped functions between training and test sets.
3. Use tree-sitter to parse the decompiled code and extract the AST.


### Deduplicating binary programs

As pointed out by previous work VarBERT, binary programs compiled from GitHub repositories may contain duplicated binaries.
Thus we use `preprocess/dedup_dataset.py` to identify and remove deduplicated binaries.
Specifically, everytime we add a new binary program to the dataset, we will record the names of all functions in the binary program.
For each new binary program, we will compare the names of all functions with the names of all functions in the existing binary programs.
We include a binary program only if it has at least 70% of its functions not overlapping with any existing binary functions.

The corresponding code lines are lines 44--70.

### Identifying potential overlapped functions

We use string similarity to identify potential overlapped functions between the training and test sets.
In lines 76--163 of `preprocess/dedup_dataset.py`, we first normalize all decompiled functions with their ground truth names,
and then calculate the string similarity between each pair of functions across the training and test sets.
To make the process tractable, we first use function signatures to identify the potentially overlapped functions.
Then we calculate the full-function similarity only for the potentially overlapped functions.

### Extracting ASTs

We use `preprocess/extract_main.py` to extract the ASTs of the decompiled code.
We use the tree-sitter parser to parse the decompiled code.

## Training

The training scripts of GenNm are under the `training/` directory.
We use [llama-factory](https://github.com/hiyouga/LLaMA-Factory) to train our models.
We did not include the code of llama-factory in our artifact. One can simply clone llama-factory from their repository.

**Please install the dependency of llama-factory following the `README` of llama-factory.** 
To use our training script directly, please make sure you have access to the base model [CodeGemma](https://huggingface.co/google/codegemma-2b). 
After that, please use `huggingface-cli login` to login to your huggingface account before running the script.

We include all the necessary scripts to train GenNm with llama-factory.
`training/dataset_info.json` defines our dataset configurations, `training/deep_speed_config.json` how we use deepspeed to facilitate training,
and `training/train-gennm.sh` is the script to train GenNm.


After training the model with SFT loss, we further use the model to inference on the training dataset.
Then we construct the SymPO dataset by selecting pairs of better name predictions and worse name predictions.
The construction of SymPO dataset is in `training/gen_sympo_ds.py`.

Lines 120--293 construct one SymPO entry. Specifically, we keep track of all possible predictions for each placeholder variable. Then we identify the best combination of names that achieves the highest precision and recall; and the worst combination of names that achieves the lowest precision and recall. A SymPO data entry contains the query function and variables, and the best and worst predictions.

Furthermore, to reduce noise from the dataset and make the training more efficient, we use heuristics to filter out SymPO entries that may introduce undesirable noise. The implementation is in lines 308--372.

## Inference

SymPO features an iterative inference process.
The script `scripts/infer_1_generation.sh` shows an example of the inference process.
It first uses `inference/infer_vllm.py` to predict variable names from the local context of functions individually.
Then it uses `inference/gen_hints.py` to propagate program contexts along the call graph.
After that, it calls `inference/infer_vllm.py` again, but with an additional argument specifying the propagated contextual hints.
The process repeats until the variable names converge.
Empirically, we find 2 iterations are sufficient.
Then we use `inference/combine_names.py` to collect all predictions for a given variable across all iterations.
Finally, we use `inference/prop_names.py` to identify the most consistent prediction for each variable based on the predictions from all iterations and program properties (e.g., data-flow).

## Evaluation

We evaluate the performance of GenNm by two metrics.

First, we follow previous work SymLM to compute the token-level precision and recall for each variable.
The implementation is in `evaluation/parse_inference_ret.py`.
Second, we use GPT4 as an evaluator to mimic how a human developer would evaluate the variable names.
The implementation is in `evaluation/gpt4_evaluator.py`.
