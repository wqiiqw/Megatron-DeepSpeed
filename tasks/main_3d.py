# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main 3D parallel tasks functionality."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import torch
import megatron
from deepspeed.checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from main import get_tasks_args
from megatron.arguments import parse_args, _print_args
from megatron.global_vars import get_args, set_args
from megatron.initialize import initialize_megatron


def override_args(base_args, override, skip_keys, skip_if_specified_keys):
    for k, v in vars(override).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(base_args, k) is not None:
            continue
        setattr(base_args, k, v)


# Below function is adapted from tasks/eval_harness/evaluate.py
def init_args(extra_args_provider):
    # parse the megatorn args, but wait with initializing megatron as they will be overridden later.
    args = parse_args(extra_args_provider)

    # we set below values as we don't validate args
    model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    args.data_parallel_size = args.world_size // model_parallel_size
    args.eval_micro_batch_size = args.micro_batch_size
    args.global_batch_size = args.micro_batch_size * args.data_parallel_size
    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    args.curriculum_learning_legacy = False

    # load DeepSpeed checkpoint
    ds_checkpoint = DeepSpeedCheckpoint(args.load,
                                        tp_degree=args.tensor_model_parallel_size,
                                        pp_degree=args.pipeline_model_parallel_size,
                                        dp_degree=args.data_parallel_size)

    # Merge the current args with the checkpoint args.
    cp_args = ds_checkpoint.get_args()

    # update arguments due to name difference from ckpt
    old_to_new_arg = {"apply_layernorm_weight_plus_one": "apply_layernorm_1p"}
    for key in old_to_new_arg.keys():
        if hasattr(cp_args, key):
            setattr(args, old_to_new_arg[key], getattr(cp_args, key))

    skip_keys = ['world_size', 'rank', 'local_rank','device_count', 'micro_batch_size','global_batch_size',
                 'batch_size', 'tensorboard_dir', 'deepspeed', 'deepspeed_config', 'deepspeed_configuration',
                 'data_parallel_size', 'pipeline_model_parallel_size', 'tensor_model_parallel_size',
                 'moe_expert_parallel_size', 'moe_token_dropping', 'load', 'rampup_batch_size', 'iteration',
                 'inference', 'bias_dropout_fusion', 'masked_softmax_fusion', 'bias_dropout_fusion',
                 'gradient_accumulation_fusion', 'fp16', 'bf16', 'use_seq_len_plus_one_tokens', 'log_interval',
                 'seq_length', 'max_position_embeddings', 'encoder_seq_length', 'distributed_backend', 'device',
                 'recompute_granularity', 'deepspeed_activation_checkpointing', 'eval_micro_batch_size',
                 'use_fused_sdpa', 'use_fused_rmsnorm']

    if args.checkpoint_override_tokenizer:
        skip_keys += ['merge_file', 'tokenizer_model', 'tokenizer_type',
                      'vocab_extra_ids', 'vocab_file']

    skip_if_specified = ['merge_file', 'vocab_file']

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # stop megatron from reparsing the arguments.
    set_args(args)
    initialize_megatron(allow_parsing=False, allow_validating_args=False)
    torch.distributed.barrier()

    # Initializing megatron will update eg. tokenizer size. Override again.
    override_args(args, cp_args, skip_keys, skip_if_specified)

    # Create minimal deepspeed configuration
    if args.deepspeed_config is None:
        args.deepspeed_config_dict = {
            'train_batch_size': args.global_batch_size,
            'train_micro_batch_size_per_gpu': args.micro_batch_size,
            'bf16': {'enabled': args.bf16},
            'fp16': {'enabled': args.fp16},
            'zero_optimization': {'stage': 0},
        }

    # print final arguments.
    _print_args("arguments", args)


if __name__ == '__main__':

    def my_rebuild(data, dtype, device, requires_grad):
        device = 'cpu' if 'hpu' in device else device
        tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
        tensor.requires_grad = requires_grad
        return tensor

    torch._utils._rebuild_device_tensor_from_numpy = my_rebuild
    init_args(get_tasks_args)
    args = get_args()

    if args.task in ['LAMBADA', 'WIKITEXT103']:
        from zeroshot_gpt.evaluate import main
    else:
        raise NotImplementedError(f'Task {args.task} is not implemented.')

    main()
