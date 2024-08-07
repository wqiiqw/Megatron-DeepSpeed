source common.sh

function config_seq_parallel() {
  local enabled="$1"

  if [ "$enabled" -eq "1" ]; then
    export HL_SEQ_PARALLEL=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
  else
    export HL_SEQ_PARALLEL=0
    unset CUDA_DEVICE_MAX_CONNECTIONS
  fi
}

# -------------------------------------------------------------------------------
# Static configuration throughout the tests
# -------------------------------------------------------------------------------
export BASE_OUTPUT=/workdisk/misland/work/mds-fork/Megatron-DeepSpeed-fork/out/moe/tests/sp${BASE_OUTPUT_EXT}
config_dataset
config_training_regime
config_model
config_capacity_bins 0
config_gpt_pipe 1

# -------------------------------------------------------------------------------
# Test sequence parallel for dense model (no moe)
# -------------------------------------------------------------------------------

# DP=2 TP=2 PP=2
config_n_experts 1
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 2 1 8 1
config_seq_parallel 0
run dense_base
config_seq_parallel 1
run dense_seq

# -------------------------------------------------------------------------------
# Test sequence parallel for MoE model
# -------------------------------------------------------------------------------
config_n_experts 4

# DP=2 TP=2 PP=2 EP=2 ETP
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 2 2 8 1
config_seq_parallel 0
run moe_base
config_seq_parallel 1
run moe_sp

# DP=2 TP=2 PP=2 EP=1 ETP
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 2 2 1 8 1
config_seq_parallel 0
run moe_base
config_seq_parallel 1
run moe_sp

# DP=4 TP=2 PP=1 EP=2 ETP
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 2 1 2 8 1
config_seq_parallel 0
run moe_base
config_seq_parallel 1
run moe_sp

# DP=4 TP=2 PP=1 EP=4 ETP
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 2 1 4 8 1
config_seq_parallel 0
run moe_base
config_seq_parallel 1
run moe_sp

# DP=2 TP=4 PP=1 EP=2 ETP
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 4 1 2 8 1
config_seq_parallel 0
run moe_base
config_seq_parallel 1
run moe_sp

# DP=2 TP=4 PP=1 EP=1 ETP
config_env 8
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 4 1 1 8 1
config_seq_parallel 0
run moe_base
config_seq_parallel 1
run moe_sp
