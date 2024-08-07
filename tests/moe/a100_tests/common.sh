#!/bin/bash


function config_env() {
  local n="$1"

  mds_base_path=/workdisk/misland/work/mds-fork
  export PATH=$mds_base_path/deepspeed-fork/bin:$PATH
  export PYTHONPATH=$mds_base_path/deepspeed-fork:$mds_base_path/Megatron-DeepSpeed-fork/
  export HL_DEVICES_PER_NODE=${n}
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
}

function config_dataset() {
  export HL_DATA_DIR_ROOT=/datasets/red_pajama
  export HL_DATA_CACHE_DIR=/workdisk/misland/work/datasets/cache/llama_rp_medium/
  export HL_DATA_FILE_PREFIX=medium
}

function config_training_regime() {
  export HL_GBS=128
  export HL_TRAIN_ITERS=4000
  export HL_SAVE_INTERVAL=4000
  export HL_EVAL_INTERVAL=1000
  export HL_EVAL_ITERS=10
}

function config_model() {
  export HL_MIXTRAL_MODEL=small
}

function config_capacity_bins() {
  local n_bins="$1"
  export HL_MOE_NUM_CAPACITY_BINS=${n_bins}
  export HL_MOE_CAPACITY_BINS=
  export HL_CAPACITY_BINS_EXP_BASE=1.5
  export HL_MOE_CAPACITY_BINS_ALIGNMENT=16
  export HL_MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL=10
  export HL_MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP=2
}

function config_init_capacity_bins() {
  local capacity_bins="$1"
  export HL_MOE_CAPACITY_BINS="${capacity_bins}"
}

function config_clear_init_capacity_bins() {
  unset HL_MOE_CAPACITY_BINS
}

function config_set_optimize_interval() {
  local optimize_interval="$1"
  export HL_MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL=${optimize_interval}
}

function config_n_experts() {
  local n_experts="$1"
  export HL_MOE_NUM_EXPERTS=${n_experts}
}

function config_gpt_pipe() {
  local use_gpt_pipe="$1"
  export HL_USE_GPT_PIPE=${use_gpt_pipe}
}

function config_scaling_dp_tp_pp_ep_mbs_moe_tp() {
  export HL_DP="$1"
  export HL_TP="$2"
  export HL_PP="$3"
  export HL_MOE_EP="$4"
  export HL_MICRO_BATCH="$5"
  export HL_MOE_ENABLE_EXPERT_TP="$6"
}

function run() {
  local tag="$1"
  now=`date +"%Y%m%d_%H%M"`
  gpt_type="gpt_2d"
  if [ "$HL_USE_GPT_PIPE" -eq "1" ]; then
    gpt_type="gpt_3d"
  fi
  run_name=${tag}_${gpt_type}_d${HL_DP}_t${HL_TP}_p${HL_PP}_e${HL_MOE_EP}_moe_t${HL_MOE_ENABLE_EXPERT_TP}_mbs${HL_MICRO_BATCH}_experts${HL_MOE_NUM_EXPERTS}_bins${HL_MOE_NUM_CAPACITY_BINS}  # _${now}
  export HL_RESULTS_DIR=${BASE_OUTPUT}/${run_name}
  mkdir -p ${HL_RESULTS_DIR}

  echo "Start ${run_name}"
  ../../../scripts/run_mixtral.sh  2>&1 | tee ${HL_RESULTS_DIR}/log.txt

  # check processes exit ok
  num_exit_ok=$(grep -e "Process .* exits successfully" ${HL_RESULTS_DIR}/log.txt | wc -l)
  if [[ $num_exit_ok -eq 0 ]]; then
    echo "-------------------------------------------------------------------------------"
    echo "Failed running ${run_name}"
    echo "-------------------------------------------------------------------------------"
    exit 1
  fi

  echo "DONE ${run_name}"
  sleep 10
}
