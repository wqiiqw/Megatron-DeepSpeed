# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from deepspeed.accelerator import get_accelerator
from megatron import get_args

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
except:
    FusedRMSNorm = None


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5, sequence_parallel=False):
        super().__init__()
        self.epsilon = eps
        self.weight = Parameter(torch.empty(dim,
                                device=get_accelerator().current_device_name(),
                                dtype=get_args().params_dtype))
        init.ones_(self.weight)
        self.use_fused_rmsnorm = get_args().use_fused_rmsnorm

        if sequence_parallel:
            setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def forward(self, x):
        if self.use_fused_rmsnorm and x.device.type == "hpu":
            assert FusedRMSNorm is not None, "failed to import FusedRMSNorm"
            return FusedRMSNorm.apply(x, self.weight, self.epsilon)
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x**2, -1, keepdim=True)
        norm = x.mul(norm.add_(self.epsilon).rsqrt_())
        return self.weight * norm.to(dtype)
