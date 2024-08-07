source common.sh

# -------------------------------------------------------------------------------
# Test specific functions
# -------------------------------------------------------------------------------

# use shorter regime
function config_training_regime() {
  export HL_GBS=128
  export HL_TRAIN_ITERS=500
  export HL_LR_WARMUP=10
  export HL_SAVE_INTERVAL=500
  export HL_EVAL_INTERVAL=100
  export HL_EVAL_ITERS=10
}

function run() {
  local tag="$1"

  export RUN_NAME_ED=${tag}_ed_d${HL_DP}_t${HL_TP}_p${HL_PP}_e${HL_MOE_EP}_moe_t${HL_MOE_ENABLE_EXPERT_TP}_mbs${HL_MICRO_BATCH}_experts${HL_MOE_NUM_EXPERTS}_bins${HL_MOE_NUM_CAPACITY_BINS}
  export RUN_NAME_DE=${tag}_de_d${HL_DP}_t${HL_TP}_p${HL_PP}_e${HL_MOE_EP}_moe_t${HL_MOE_ENABLE_EXPERT_TP}_mbs${HL_MICRO_BATCH}_experts${HL_MOE_NUM_EXPERTS}_bins${HL_MOE_NUM_CAPACITY_BINS}

  if [ "$HL_MOE_USE_DATA_BEFORE_EXPERT_PARALLEL" -eq "0" ]; then
    run_name=${RUN_NAME_ED}
  else
    run_name=${RUN_NAME_DE}
  fi

  export HL_CHECKPOINTS_DIR=${BASE_OUTPUT}/${run_name}/checkpoints
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

function enable_use_dp_before_ep() {
  export HL_MOE_USE_DATA_BEFORE_EXPERT_PARALLEL=1
}

function disable_use_dp_before_ep() {
  export HL_MOE_USE_DATA_BEFORE_EXPERT_PARALLEL=0
}

# -------------------------------------------------------------------------------
# Static configuration throughout the tests
# -------------------------------------------------------------------------------
export BASE_OUTPUT=/workdisk/misland/work/mds-fork/Megatron-DeepSpeed-fork/out/moe/tests/dp_before_ep${BASE_OUTPUT_EXT}
config_dataset
config_training_regime
config_model
config_n_experts 4
config_capacity_bins 0
config_gpt_pipe 1

# -------------------------------------------------------------------------------
# Test data before expert parallel with 3d configurations
# -------------------------------------------------------------------------------

# DP=2 TP=1 PP=1 EP=2
config_env 2
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 1 1 2 8 0
disable_use_dp_before_ep; run x2
enable_use_dp_before_ep; run x2

# DP=1 TP=2 PP=1 EP=1
config_env 2
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 2 1 1 8 1
disable_use_dp_before_ep; run x2
enable_use_dp_before_ep; run x2

# DP=2 TP=2 PP=1 EP=2
config_env 4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 1
disable_use_dp_before_ep; run x4
enable_use_dp_before_ep; run x4

# DP=2 TP=2 PP=1 EP=1
config_env 4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 1 8 1
disable_use_dp_before_ep; run x4
enable_use_dp_before_ep; run x4

# DP=2 TP=1 PP=2 EP=2
config_env 4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 1 2 2 8 0
disable_use_dp_before_ep; run x4
enable_use_dp_before_ep; run x4

# DP=4 TP=1 PP=2 EP=2
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 2 2 8 0
disable_use_dp_before_ep; run x8
enable_use_dp_before_ep; run x8

# DP=2 TP=2 PP=2 EP=2
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 2 2 8 1
disable_use_dp_before_ep; run x8
enable_use_dp_before_ep; run x8

if [ $ALLOW_ONLY_NON_MOE_TP -eq 1 ]; then

  # DP=1 TP=2 PP=1 EP=1
  config_env 2
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 2 1 1 8 0
  disable_use_dp_before_ep; run x2
  enable_use_dp_before_ep; run x2

  # DP=2 TP=2 PP=1 EP=2
  config_env 4
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 0
  disable_use_dp_before_ep; run x4
  enable_use_dp_before_ep; run x4

  # DP=2 TP=2 PP=2 EP=2
  config_env 8
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 2 2 8 0
  disable_use_dp_before_ep; run x8
  enable_use_dp_before_ep; run x8

fi
