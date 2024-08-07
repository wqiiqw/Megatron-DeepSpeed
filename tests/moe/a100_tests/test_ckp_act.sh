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

function config_ckp_act_disabled() {
  export HL_CKP_ACT=0
}

function config_ckp_act_full() {
  export HL_CKP_ACT=1
}

function config_ckp_act_selective() {
  export HL_CKP_ACT=2
}

function run() {
  local tag="$1"

  gpt_type="gpt_2d"
  if [ "$HL_USE_GPT_PIPE" -eq "1" ]; then
    gpt_type="gpt_3d"
  fi

  export RUN_NAME_DIS=${tag}_ckp_act_dis_${gpt_type}_d${HL_DP}_t${HL_TP}_p${HL_PP}_e${HL_MOE_EP}_moe_t${HL_MOE_ENABLE_EXPERT_TP}_mbs${HL_MICRO_BATCH}_experts${HL_MOE_NUM_EXPERTS}_bins${HL_MOE_NUM_CAPACITY_BINS}
  export RUN_NAME_FUL=${tag}_ckp_act_ful_${gpt_type}_d${HL_DP}_t${HL_TP}_p${HL_PP}_e${HL_MOE_EP}_moe_t${HL_MOE_ENABLE_EXPERT_TP}_mbs${HL_MICRO_BATCH}_experts${HL_MOE_NUM_EXPERTS}_bins${HL_MOE_NUM_CAPACITY_BINS}
  export RUN_NAME_SEL=${tag}_ckp_act_sel_${gpt_type}_d${HL_DP}_t${HL_TP}_p${HL_PP}_e${HL_MOE_EP}_moe_t${HL_MOE_ENABLE_EXPERT_TP}_mbs${HL_MICRO_BATCH}_experts${HL_MOE_NUM_EXPERTS}_bins${HL_MOE_NUM_CAPACITY_BINS}

  if [ "$HL_CKP_ACT" -eq "0" ]; then
    run_name=${RUN_NAME_DIS}
  elif [ "$HL_CKP_ACT" -eq "1" ]; then
    run_name=${RUN_NAME_FUL}
  elif [ "$HL_CKP_ACT" -eq "2" ]; then
    run_name=${RUN_NAME_SEL}
  else
    echo "Bad configuration of HL_CKP_ACT=${HL_CKP_ACT}"
    exit 1
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

# -------------------------------------------------------------------------------
# Specific testing methods
# -------------------------------------------------------------------------------
function verify_same_final_loss() {
  ckp_act_dis_log=${BASE_OUTPUT}/${RUN_NAME_DIS}/log.txt
  ckp_act_ful_log=${BASE_OUTPUT}/${RUN_NAME_FUL}/log.txt
  ckp_act_sel_log=${BASE_OUTPUT}/${RUN_NAME_SEL}/log.txt

  # final training log line for 200 steps looks like the following:
  # steps: 200 loss: 5.5508 lm loss: 5.5103 moe loss: 0.0405 iter time (s): 0.936 samples/sec: 136.706

  # extract lm_loss ckp_act=disabled
  ckp_act_dis_final_training_log=$(grep "steps: 200" ${ckp_act_dis_log})
  ckp_act_dis_final_lm_loss=$(echo "$ckp_act_dis_final_training_log" | awk '{print $4}')

  # extract load ckp_act=full
  ckp_act_ful_final_training_log=$(grep "steps: 200" ${ckp_act_ful_log})
  ckp_act_ful_final_lm_loss=$(echo "$ckp_act_ful_final_training_log" | awk '{print $4}')

  # extract load ckp_act=selective
  ckp_act_sel_final_training_log=$(grep "steps: 200" ${ckp_act_sel_log})
  ckp_act_sel_final_lm_loss=$(echo "$ckp_act_sel_final_training_log" | awk '{print $4}')

  echo "Final training loss report for ${SAVE_RUN_NAME}"
  echo "Disabled:  ${ckp_act_dis_final_training_log}"
  echo "Full:      ${ckp_act_ful_final_training_log}"
  echo "Selective: ${ckp_act_sel_final_training_log}"

  # compare
  if [ -z $ckp_act_dis_final_lm_loss ]; then
    echo "FAIL - could not get lm loss at ${ckp_act_dis_log}"
    exit 1
  fi

  if [ -z $ckp_act_ful_final_lm_loss ]; then
    echo "FAIL - could not get lm loss at ${ckp_act_ful_log}"
    exit 1
  fi

  if [ -z $ckp_act_sel_final_lm_loss ]; then
    echo "FAIL - could not get lm loss at ${ckp_act_sel_log}"
    exit 1
  fi

  if [[ $ckp_act_dis_final_lm_loss == $ckp_act_ful_final_lm_loss ]] && \
     [[ $ckp_act_ful_final_lm_loss == $ckp_act_sel_final_lm_loss ]]; then
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
export BASE_OUTPUT=/workdisk/misland/work/mds-fork/Megatron-DeepSpeed-fork/out/moe/tests/ckp_act${BASE_OUTPUT_EXT}
config_dataset
config_training_regime
config_model
config_gpt_pipe 1
config_n_experts 4

# capacity bins are tested at the end of this file
config_capacity_bins 0

# -------------------------------------------------------------------------------
# Test single device
# -------------------------------------------------------------------------------
config_env 1

# DP=1 TP=1 PP=1 EP=1 (4 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 1 1 1 8 0
config_ckp_act_disabled;  run gpu1
config_ckp_act_full;      run gpu1
config_ckp_act_selective; run gpu1
verify_same_final_loss

# -------------------------------------------------------------------------------
# Test 4 devices
# -------------------------------------------------------------------------------
config_env 4

# DP=4 TP=1 PP=1 EP=4 (1 local expert)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 4 8 0
config_ckp_act_disabled;  run gpu4
config_ckp_act_full;      run gpu4
config_ckp_act_selective; run gpu4
verify_same_final_loss

# DP=2 TP=2 PP=1 EP=2 (2 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 1
config_ckp_act_disabled;  run gpu4
config_ckp_act_full;      run gpu4
config_ckp_act_selective; run gpu4
verify_same_final_loss

# DP=2 TP=1 PP=2 EP=2 (2 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 1 2 2 8 0
config_ckp_act_disabled;  run gpu4
config_ckp_act_full;      run gpu4
config_ckp_act_selective; run gpu4
verify_same_final_loss

# DP=1 TP=2 PP=2 EP=1 (4 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 2 2 1 8 1
config_ckp_act_disabled;  run gpu4
config_ckp_act_full;      run gpu4
config_ckp_act_selective; run gpu4
verify_same_final_loss

if [ $ALLOW_ONLY_NON_MOE_TP -eq 1 ]; then
  # DP=2 TP=2 PP=1 EP=2 (2 local experts)
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 0
  config_ckp_act_disabled;  run gpu4
  config_ckp_act_full;      run gpu4
  config_ckp_act_selective; run gpu4
  verify_same_final_loss
fi

# -------------------------------------------------------------------------------
# Test 8 devices
# -------------------------------------------------------------------------------
config_env 8

# DP=2 TP=2 PP=2 EP=2 (2 local experts)
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 2 2 8 1
config_ckp_act_disabled;  run gpu8
config_ckp_act_full;      run gpu8
config_ckp_act_selective; run gpu8
verify_same_final_loss

if [ $ALLOW_ONLY_NON_MOE_TP -eq 1 ]; then
  # DP=2 TP=2 PP=2 EP=2 (2 local experts)
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 2 2 8 0
  config_ckp_act_disabled;  run gpu8
  config_ckp_act_full;      run gpu8
  config_ckp_act_selective; run gpu8
  verify_same_final_loss
fi
