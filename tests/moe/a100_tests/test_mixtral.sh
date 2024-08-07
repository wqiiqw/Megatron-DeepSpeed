source common.sh

# -------------------------------------------------------------------------------
# Small Mixtral with default Mixtral script configuration
# -------------------------------------------------------------------------------
export BASE_OUTPUT=/workdisk/misland/work/mds-fork/Megatron-DeepSpeed-fork/out/moe/tests/mixtral${BASE_OUTPUT_EXT}
config_dataset
config_training_regime
config_model
#config_n_experts 8
config_gpt_pipe 1

config_env 1
config_scaling_dp_tp_pp_ep_mbs_moe_tp 1 1 1 1 8 0
run small
