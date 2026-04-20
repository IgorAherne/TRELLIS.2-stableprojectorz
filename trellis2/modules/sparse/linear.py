# File: trellis2/modules/sparse/linear.py
# trellis2/modules/sparse/linear.py
import torch
import torch.nn as nn
from . import VarLenTensor

__all__ = [
    'SparseLinear'
]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        return input.replace(super().forward(input.feats))
