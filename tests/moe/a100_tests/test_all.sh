# For now, test only when both Non-MoE and MoE layers have same TP configuration
export ALLOW_ONLY_NON_MOE_TP=0

# Use below to configure an extension to the base output path
# export BASE_OUTPUT_EXT=

# Run tests
./test_dp.sh
./test_dp_tp.sh
./test_3d.sh
./test_ckp.sh
./test_bins.sh
./test_ckp_act.sh
./test_dp_before_ep.sh
