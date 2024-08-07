source common.sh

# -------------------------------------------------------------------------------
# Static configuration throughout the tests
# -------------------------------------------------------------------------------
export BASE_OUTPUT=/workdisk/misland/work/mds-fork/Megatron-DeepSpeed-fork/out/moe/tests/dp_tp${BASE_OUTPUT_EXT}
config_dataset
config_training_regime
config_model
config_n_experts 4
config_capacity_bins 0
config_gpt_pipe 1

# -------------------------------------------------------------------------------
# Test dp vs tp, 2 devices, no capacity bins
# -------------------------------------------------------------------------------
config_env 2

# DP=2 TP=1 EP=2
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 1 1 2 8 0
run x2

# DP=1 TP=2 EP=1
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 2 1 1 8 1
run x2

if [ $ALLOW_ONLY_NON_MOE_TP -eq 1 ]; then
  # DP=1 TP=2 EP=2
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 2 1 2 8 0
  run x2
fi

# -------------------------------------------------------------------------------
# Test dp vs tp vs dp+tp, 4 devices, no capacity bins
# -------------------------------------------------------------------------------
config_env 4

# DP=4 TP=1 EP=4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 4 8 0
run x4

# DP=2 TP=2 EP=2
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 1
run x4

# DP=1 TP=4 EP=1
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 4 1 1 8 1
run x4

if [ $ALLOW_ONLY_NON_MOE_TP -eq 1 ]; then
  # DP=1 TP=4 EP=4
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 4 1 4 8 0
  run x4

  # DP=2 TP=2 EP=2
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 1 2 8 0
  run x4
fi

# -------------------------------------------------------------------------------
# Test dp vs tp vs dp+tp, 8 devices, no capacity bins
# -------------------------------------------------------------------------------
config_env 8

# DP=2 TP=4 EP=2
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 4 1 2 8 1
run x8

# DP=4 TP=2 EP=4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 2 1 4 8 1
run x8

if [ $ALLOW_ONLY_NON_MOE_TP -eq 1 ]; then
  # DP=2 TP=4 EP=2
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 4 1 2 8 0
  run x8

  # DP=4 TP=2 EP=4
  config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 2 1 4 8 0
  run x8
fi
