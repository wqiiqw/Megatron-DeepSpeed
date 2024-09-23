# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

#!/bin/bash
set -ex

BASE_DATA_PATH=${HL_DATA_DIR_ROOT:-/scratch-1/llama-training/bigscience/data/oscar/zh/}  #  tokenized_text_document.bin  tokenized_text_document.idx
DATA_PATH=${BASE_DATA_PATH}/tokenized_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
DS_CONFIG=ds_config.json

TP=1  # Tensor Parallelism across 8 HPUs
PP=1  # No Pipeline Parallelism

# GPT3 6.7B
NLAYERS=32
HIDDEN=4096
NUMATTNHEADS=32

GLOBAL_BATCH=8  # Adjusted for single node with 8 HPUs
MICRO_BATCH=8    # Adjusted for single node with 8 HPUs

ZERO_STAGE=2

OUTPUT_DIR=ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

export NCCL_DEBUG=warn 

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}" 
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


deepspeed ../pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $NUMATTNHEADS \
    --seq-length 256 \
    --loss-scale 12 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH  \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters 1000 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000 \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_PATH \
    --merge-file $MERGE_PATH \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16 \
    --checkpoint-activations \
    --recompute-granularity=full \
    --recompute-method=uniform \
    --no-gradient-accumulation-fusion \
    --tensorboard-dir $OUTPUT_DIR \
    $ds_args \
    --exit-interval 5000 | tee ${OUTPUT_DIR}/output.log