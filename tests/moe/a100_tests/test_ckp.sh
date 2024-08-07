source common.sh

# -------------------------------------------------------------------------------
# Override common functions
# -------------------------------------------------------------------------------
function config_training_regime() {
  export HL_GBS=128
  export HL_TRAIN_ITERS=200
  export HL_LR_WARMUP=10
  export HL_SAVE_INTERVAL=100
  export HL_EVAL_INTERVAL=1000
  export HL_EVAL_ITERS=10
}

function config_load_tag() {
  export HL_CHECKPOINT_LOAD_TAG=global_step100
}

function unset_load_tag() {
  unset HL_CHECKPOINT_LOAD_TAG
}

function run() {
  local tag="$1"

  gpt_type="gpt_2d"
  if [ "$HL_USE_GPT_PIPE" -eq "1" ]; then
    gpt_type="gpt_3d"
  fi

  export LOAD_RUN_NAME=${tag}_load_${gpt_type}_d${HL_DP}_t${HL_TP}_p${HL_PP}_e${HL_MOE_EP}_moe_t${HL_MOE_ENABLE_EXPERT_TP}_mbs${HL_MICRO_BATCH}_experts${HL_MOE_NUM_EXPERTS}_bins${HL_MOE_NUM_CAPACITY_BINS}  # _${now}
  export SAVE_RUN_NAME=${tag}_save_${gpt_type}_d${HL_DP}_t${HL_TP}_p${HL_PP}_e${HL_MOE_EP}_moe_t${HL_MOE_ENABLE_EXPERT_TP}_mbs${HL_MICRO_BATCH}_experts${HL_MOE_NUM_EXPERTS}_bins${HL_MOE_NUM_CAPACITY_BINS}  # _${now}

  if [ -z "$HL_CHECKPOINT_LOAD_TAG" ]; then
    run_name=${SAVE_RUN_NAME}
    export HL_SAVE=1
  else
    run_name=${LOAD_RUN_NAME}
    export HL_SAVE=0
  fi
  export HL_CHECKPOINTS_DIR=${BASE_OUTPUT}/${SAVE_RUN_NAME}/checkpoints

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

# -------------------------------------------------------------------------------
# Specific testing methods
# -------------------------------------------------------------------------------
function verify_same_final_loss() {
  save_log=${BASE_OUTPUT}/${SAVE_RUN_NAME}/log.txt
  load_log=${BASE_OUTPUT}/${LOAD_RUN_NAME}/log.txt

  # final training log line for 200 steps looks like the following:
  # steps: 200 loss: 5.5508 lm loss: 5.5103 moe loss: 0.0405 iter time (s): 0.936 samples/sec: 136.706

  # extract save lm_loss
  save_final_training_log=$(grep "steps: 200" ${save_log})
  save_final_lm_loss=$(echo "$save_final_training_log" | awk '{print $4}')

  # extract load lm_loss
  load_final_training_log=$(grep "steps: 200" ${load_log})
  load_final_lm_loss=$(echo "$load_final_training_log" | awk '{print $4}')

  echo "Final training loss report for ${SAVE_RUN_NAME}"
  echo "save: ${save_final_training_log}"
  echo "load: ${load_final_training_log}"

  # compare
  if [ -z $save_final_lm_loss ]; then
    echo "FAIL - could not get lm loss at ${save_log}"
    exit 1
  fi

  if [ -z $load_final_lm_loss ]; then
    echo "FAIL - could not get lm loss at ${load_log}"
    exit 1
  fi

  if [ $save_final_lm_loss == $load_final_lm_loss ]; then
    echo "----"
    echo "PASS"
    echo "----"
    sleep 5
  else
    echo "FAIL"
    exit 1
  fi
}

# -------------------------------------------------------------------------------
# Static configuration throughout the tests
# -------------------------------------------------------------------------------
export BASE_OUTPUT=/workdisk/misland/work/mds-fork/Megatron-DeepSpeed-fork/out/moe/tests/ckp${BASE_OUTPUT_EXT}
config_dataset
config_training_regime
config_model
config_gpt_pipe 1
config_n_experts 4

# capacity bins are tested at the end of this file
config_capacity_bins 0

# -------------------------------------------------------------------------------
# Test 3d save/load from checkpoint, 1 device, no capacity bins
# -------------------------------------------------------------------------------
config_env 1

# DP=1 with 4 local experts
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 1 1 1 8 0
unset_load_tag;  run gpu1
config_load_tag; run gpu1
verify_same_final_loss

# -------------------------------------------------------------------------------
# Test 3d save/load from checkpoint, 2 devices, no capacity bins
# -------------------------------------------------------------------------------
config_env 2

# DP=2 TP=1 EP=2 (2 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 1 1 2 8 0
unset_load_tag;  run gpu2
config_load_tag; run gpu2
verify_same_final_loss

# DP=1 TP=2 EP=1 (4 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 2 1 1 8 1
unset_load_tag;  run gpu2
config_load_tag; run gpu2
verify_same_final_loss

# DP=1 TP=2 EP=1 (4 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 2 1 1 8 1
unset_load_tag;  run gpu2
config_load_tag; run gpu2
verify_same_final_loss

# -------------------------------------------------------------------------------
# Test 3d save/load from checkpoint,4 devices, no capacity bins
# -------------------------------------------------------------------------------
config_env 4

# DP=4 TP=1 EP=4 (1 local expert)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 4 8 0
unset_load_tag;  run gpu4
config_load_tag; run gpu4
verify_same_final_loss

# DP=2 TP=2 EP=2 (2 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 1
unset_load_tag;  run gpu4
config_load_tag; run gpu4
verify_same_final_loss

if [ $ALLOW_ONLY_NON_MOE_TP -eq 1 ]; then
  # DP=2 TP=2 with 2 local experts
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 0
  unset_load_tag;  run gpu4
  config_load_tag; run gpu4
  verify_same_final_loss
fi

# -------------------------------------------------------------------------------
# Test 3d save/load from checkpoint,4 devices, with capacity bins
# -------------------------------------------------------------------------------
config_env 4
config_capacity_bins 8

# DP=4 TP=1 EP=4 (1 local expert)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 4 8 0
unset_load_tag;  run gpu4_bins
config_load_tag; run gpu4_bins
verify_same_final_loss

# DP=2 TP=2 EP=2 (2 local expert)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 1
unset_load_tag;  run gpu4_bins
config_load_tag; run gpu4_bins
verify_same_final_loss

# DP=2 TP=2 EP=2 (2 local expert) with initialized bins
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 1
config_capacity_bins 4
config_init_capacity_bins "2,1,2,3,4 4,1024,1120,2240,4096"
config_set_optimize_interval 0
unset_load_tag;  run gpu4_initialized_bins
config_load_tag; run gpu4_initialized_bins
verify_same_final_loss
config_capacity_bins 8; # reset before next test

# DP=2 TP=2 EP=2 (2 local expert) with initialized bins with optimize bins
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 1
config_capacity_bins 4
config_init_capacity_bins "2,1,2,3,4 4,1024,1120,2240,4096"
config_set_optimize_interval 10
unset_load_tag;  run gpu4_initialized_optimized_bins
config_load_tag; run gpu4_initialized_optimized_bins
verify_same_final_loss
config_capacity_bins 8; # reset before next test

if [ $ALLOW_ONLY_NON_MOE_TP -eq 1 ]; then
  # DP=2 TP=2 EP=2 (2 local experts)
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 0
  unset_load_tag;  run gpu4_bins
  config_load_tag; run gpu4_bins
  verify_same_final_loss
fi
