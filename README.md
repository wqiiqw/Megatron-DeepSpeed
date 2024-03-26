# LLaMA for PyTorch

This directory provide scripts to train the GPT-based LLaMA models in the Megatron-DeepSpeed repository on Intel® Gaudi® 2 AI accelerator.
Before you get started, make sure to review the [Supported Configuration](#supported-configuration).

## Table of Contents
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview
This implementation is based on https://github.com/microsoft/Megatron-DeepSpeed at 61d5d619802de2373a93f33a99b5439da28a5b0a.
Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf) and [2](https://arxiv.org/pdf/2104.04473.pdf)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository is for training large transformer language models such as LLaMA at scale. Codebase is capable of efficiently training very large (hundreds of billions of parameters) language models with both model and data parallelism.
LLaMA training is based on https://arxiv.org/abs/2302.13971
LLaMA2 training is based on https://arxiv.org/pdf/2307.09288.pdf

### How to use
Users bear sole liability and responsibility to follow and comply with any third party licenses, and Habana Labs disclaims and will bear no liability with respect to users’ use or compliance with third party licenses.


## Setup
Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi 2.

### Install Habana DeepSpeed-fork
Please follow the instructions provided in the [DeepSpeed Installation Guide](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/DeepSpeed_User_Guide/DeepSpeed_User_Guide.html#installing-deepspeed-library) to install deepspeed-fork.

### Clone Habana Megatron-DeepSpeed
In the docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.
```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Megatron-DeepSpeed
```

```
export MEGATRON_DEEPSPEED_ROOT=/path/to/Megatron-DeepSpeed
export PYTHONPATH=$MEGATRON_DEEPSPEED_ROOT:$PYTHONPATH
```
### Install Megatron-DeepSpeed Requirements
* In the docker container, go to the Megatron-DeepSpeed directory:
  ```bash
  cd $MEGATRON_DEEPSPEED_ROOT
  ```

* Install the required packages using pip:
  ```bash
  pip install -r megatron/core/requirements.txt
  ```

### Dataset Preparation
Follow the instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar to download oscar-en full dataset. Note that the dataset takes around 550G of disk space. This dataset is used for training LLaMA & LLaMA 2.
#### Dataset Preparation Examples
The below provides the steps required to prepare your dataset. It is based on instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar.  The dataset in the example is intended to be `zh`
# Step 0 :
  ```bash
git clone https://github.com/bigscience-workshop/bigscience.git
cd bigscience/data/oscar
# Edit the `oscar-to-jsonl.py` in the list language_subsets and remove the comment on unshuffled_deduplicated_zh and comment out unshuffled_deduplicated_en
vi oscar-to-jsonl.py
  ```
# Step 1 :
  ```bash
  # -s can be added for subset of data
$PYTHON oscar-to-jsonl.py
  ```
# Step 2 :
  ```bash
mkdir -p zh
mv oscar*.jsonl zh
cd zh
cat oscar-[0-4].jsonl > oscar-zh.jsonl
  ```
# Step 3 :
  ```bash
$PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/preprocess_data.py --input zh/oscar-zh.jsonl --output-prefix $MEGATRON_DEEPSPEED_ROOT/zh/tokenized --vocab-file gpt2-vocab.json --merge-file gpt2-merges.txt --append-eod --tokenizer-type GPT2BPETokenizer --workers 64
# use the tokenized files from above step to train
  ```


## Training and Examples
* Training of LLaMA is based on https://arxiv.org/abs/2302.13971
* Training of LLaMA 2 is based on https://arxiv.org/pdf/2307.09288

### Multi-Card Training Examples
* Update data root dir with the path of your choice:
  ```
  HL_DATA_DIR_ROOT=/data/bigscience/oscar-en
  ```

Note: For the below commands, make sure to change the IP addresses in hostsfile according to your setup

* Run LLaMA 13B on 8 HPUs with BF16 precision:
  ```
  HL_NUM_NODES=1 HL_PP=2 HL_TP=2 HL_DP=2 scripts/run_llama.sh
  ```

* Run LLaMA 13B on 64 HPUs with BF16 precision:
  ```
  HL_HOSTSFILE=scripts/hostsfile HL_NUM_NODES=8 HL_PP=2 HL_TP=2 HL_DP=16 scripts/run_llama1.sh
  ```

* Run LLaMA 2 70B on 32 HPUs with BF16 precision:
  ```
  HL_HOSTSFILE=scripts/hostsfile HL_LLAMA_VER=2 HL_LLAMA_MODEL_SIZE=70 HL_MICRO_BATCH=1 HL_NUM_NODES=4 HL_PP=4 HL_TP=8 HL_DP=1 scripts/run_llama.sh
  ```


## Supported Configuration
| Validated on  | Intel Gaudi software Version | PyTorch Version | Mode     |
|---------------|------------------------------|-----------------|----------|
| Gaudi 2       | 1.15.0                       | 2.2.0           | Training |


## Changelog
### 1.15.0
 - Initial release.

### Script Modifications
Major changes done to the original model from [microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/commit/61d5d619802de2373a93f33a99b5439da28a5b0a) repository:
* Changed README file content.
* TFLOPs calculation changed.
* Added HPU FP8 support.
* Flash attention support via FusedSDPA is added for HPU Accelerator.
* Added RMSNorm support for HPU Accelerator.
* Added checkpoint verification.
* Added kill-switch mechanism to gracefully stop training.

## Known Issues
* Only scripts and configurations mentioned in this README are supported and verified.
* If checkpoint is already saved with adamw optimizer, training cannot be resumed with fusedadamw.
* While running LLaMA 2 7B training using TP+PP+DP with sequence parallelism disabled, a failure during checkpoint verification causes a crash after about 24 hours. This will be fixed in a subsequent release. To mitigate this issue, make sure sequence parallelism is enabled (default). Training can be resumed from last valid checkpoint.
