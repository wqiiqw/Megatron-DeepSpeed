source common.sh

# Running combinations of capacity bins with max capacity groups
# Testing:
#   1. No failures
#   2. Loss curve should be similar, but not exactly the same
#   3. Iter time should decrease as number of bins goes up
#   4. Iter time should be similar when changing max group (maybe a bit higher)

# -------------------------------------------------------------------------------
# Override common functions
# -------------------------------------------------------------------------------
function config_training_regime() {
  export HL_GBS=128
  export HL_TRAIN_ITERS=200
  export HL_LR_WARMUP=10
  export HL_SAVE_INTERVAL=1000
  export HL_EVAL_INTERVAL=1000
  export HL_EVAL_ITERS=10
}

function config_capacity_bins() {
  local n_bins="$1"
  export HL_MOE_NUM_CAPACITY_BINS=${n_bins}
  export HL_CAPACITY_BINS_EXP_BASE=1.5
  export HL_MOE_CAPACITY_BINS_ALIGNMENT=16
  export HL_MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL=50
}

function config_capacity_bins_max_group() {
  local max_group="$1"
  export HL_MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP=${max_group}
}

function get_logged_bins() {
  log_file=${HL_RESULTS_DIR}/log.txt

  # get moe_0 stats
  # resulting string should be similar to: 'bin0_1024': 0, 'bin1_1120': 0, 'bin2_2240': 160, 'bin3_4096': 0
  res=$(grep "moe_0 stats:" ${log_file} | sed 's/^.*moe_0 stats: //' | sed 's/ |.*$/ XXX /' | tr -d "{}")
  # remove bin number, resulting in: 'bin_1024': 0, 'bin_1120': 0, 'bin_2240': 160, 'bin_4096': 0
  res=$(echo "$res" | sed 's/bin[0-9]*_/bin_/g')
  # remove counts, resulting in: 'bin_1024', 'bin_1120', 'bin_2240', 'bin_4096'
  res=$(echo "$res" | sed 's/: [0-9]*,/,/g' | sed 's/: [0-9]//g')
  # cleanup bin_ and "'"
  res=$(echo "$res" | sed 's/bin_//g' | tr -d "'")
  echo ${res}
}

function get_initial_bins() {
  res=$(get_logged_bins | sed 's/ XXX.*//')
  echo ${res}
}

function get_final_bins() {
  res=$(get_logged_bins | sed 's/.*XXX \([0-9, ]*\) XXX$/\1/')
  echo ${res}
}

function verify_initial_eq_final_bins() {
  initial=$(get_initial_bins)
  final=$(get_final_bins)
  if [ "$initial" != "$final" ]; then
      echo "FAIL: Initial (${initial}) != Final (${final})"
      exit 1
  fi
}

function verify_initial_ne_final_bins() {
  initial=$(get_initial_bins)
  final=$(get_final_bins)
  if [ "$initial" == "$final" ]; then
      echo "FAIL: Initial (${initial}) == Final (${final})"
      exit 1
  fi
}

function get_final_sps_time() {
  log_file=${HL_RESULTS_DIR}/log.txt

  # get samples/sec from line formatted as:
  #    steps: 130 loss: 5.6950 lm loss: 5.6543 moe loss: 0.0407 iter time (s): 0.239 samples/sec: 535.571
  res=$(grep "steps: " ${log_file} | sed 's/^.*samples\/sec: //')
  # convert to array and pick the last value
  res=(${res// / })
  res=${res[-1]}
  echo ${res}
}

function verify_spsX_le_spsY() {
  local tag=$1
  local spsX=$2
  local spsY=$3
  res=$(awk -v f1="$spsX" -v f2="$spsY" 'BEGIN {printf(f1 <= f2 ? "OK" : "FAIL")}')
  if [ "$res" == "FAIL" ]; then
    echo "FAIL: $tag - spsX=$spsX expected to be LE spsY=$spsY"
    exit 1
  fi
  echo "PASS: $tag: $spsX <= $spsY"
}

# -------------------------------------------------------------------------------
# Static configuration throughout the tests
# -------------------------------------------------------------------------------
export BASE_OUTPUT=/workdisk/misland/work/mds-fork/Megatron-DeepSpeed-fork/out/moe/tests/bins${BASE_OUTPUT_EXT}
config_env 4
config_dataset
config_training_regime
config_model
config_gpt_pipe 1
config_n_experts 4

# -------------------------------------------------------------------------------
# DP only tests
# -------------------------------------------------------------------------------
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 4 8 0

# test bins in (1, 2, 4, 8) + max_group=1
config_capacity_bins_max_group 1
config_capacity_bins 1; run dp_max1; sps1=$(get_final_sps_time)
config_capacity_bins 2; run dp_max1; sps2=$(get_final_sps_time)
config_capacity_bins 4; run dp_max1; sps4=$(get_final_sps_time)
config_capacity_bins 8; run dp_max1; sps8=$(get_final_sps_time)
verify_spsX_le_spsY "groups=1 bins=(2 vs 8)" $sps2 $sps8
verify_spsX_le_spsY "groups=1 bins=(1 vs 4)" $sps1 $sps4

# test bins in (1, 2, 4, 8) + max_group=2
config_capacity_bins_max_group 2
config_capacity_bins 1; run dp_max2; sps1=$(get_final_sps_time)
config_capacity_bins 2; run dp_max2; sps2=$(get_final_sps_time)
config_capacity_bins 4; run dp_max2; sps4=$(get_final_sps_time)
config_capacity_bins 8; run dp_max2; sps8=$(get_final_sps_time)
verify_spsX_le_spsY "groups=1 bins=(2 vs 8)" $sps2 $sps8
verify_spsX_le_spsY "groups=1 bins=(1 vs 4)" $sps1 $sps4

# test bins in (1, 2, 4, 8) + max_group=4
config_capacity_bins_max_group 4
config_capacity_bins 1; run dp_max4; sps1=$(get_final_sps_time)
config_capacity_bins 2; run dp_max4; sps2=$(get_final_sps_time)
config_capacity_bins 4; run dp_max4; sps4=$(get_final_sps_time)
config_capacity_bins 8; run dp_max4; sps8=$(get_final_sps_time)
verify_spsX_le_spsY "groups=1 bins=(2 vs 8)" $sps2 $sps8
verify_spsX_le_spsY "groups=1 bins=(1 vs 4)" $sps1 $sps4

# -------------------------------------------------------------------------------
# Test hardcoded configuration
# -------------------------------------------------------------------------------
config_scaling_dp_tp_pp_ep_mbs_moe_tp 4 1 1 4 8 0

# test 8 capacity bins with max_group=1
config_capacity_bins_max_group 1
config_capacity_bins 4

# Configured fixed bins
config_init_capacity_bins "2,1,2,3,4 4,1024,1120,2240,4096"
config_set_optimize_interval 0
run config_fixed
verify_initial_eq_final_bins

# Configured bins followed by auto-optimization
config_init_capacity_bins "2,1,2,3,4 4,1024,1120,2240,4096"
config_set_optimize_interval 50
run config_auto
verify_initial_ne_final_bins
