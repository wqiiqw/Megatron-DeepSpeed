source common.sh

export BASE_OUTPUT=/workdisk/misland/work/mds-fork/Megatron-DeepSpeed-fork/out/moe/tests/dp${BASE_OUTPUT_EXT}

# -------------------------------------------------------------------------------
# Test 2d vs 3d with 1 device, no capacity bins
# -------------------------------------------------------------------------------
config_env 1
config_dataset
config_training_regime
config_model
config_capacity_bins 0

# No-MoE
config_n_experts 1
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 1 1 1 8 0
config_gpt_pipe 0 ; run dense
config_gpt_pipe 1 ; run dense

# DP=1 with multiple local experts
config_n_experts 4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 1 1 1 8 0
config_gpt_pipe 0 ; run dp_only
config_gpt_pipe 1 ; run dp_only

# -------------------------------------------------------------------------------
# Test 2d vs 3d with 2 devices, no capacity bins
# -------------------------------------------------------------------------------
config_env 2
config_dataset
config_training_regime
config_model
config_capacity_bins 0

# No-MoE
config_n_experts 1
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 1 1 1 8 0
config_gpt_pipe 1 ; run dense

# DP=2 with multiple local experts
config_n_experts 4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 2 1 1 2 8 0
config_gpt_pipe 1 ; run dp_only

# -------------------------------------------------------------------------------
# Test 2d vs 3d with 4 devices, no capacity bins
# -------------------------------------------------------------------------------
config_env 4
config_dataset
config_training_regime
config_model
config_capacity_bins 0

# No-MoE
config_n_experts 1
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 1 8 0
config_gpt_pipe 1 ; run dense

# DP=4 with single local expert
config_n_experts 4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 4 8 0
config_gpt_pipe 1 ; run dp_only

# DP=4 with multiple local experts
config_n_experts 4
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 2 8 0
config_gpt_pipe 1 ; run dp_only
