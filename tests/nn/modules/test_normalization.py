import pytest
import torch
from torch.nn import LayerNorm
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm3d
from torch.nn.parameter import UninitializedParameter


from torch_lazy.nn import LazyBatchNorm


class TestLazyBatchNorm:

    def test_batch_norm(self):
        m = LazyBatchNorm()
        assert isinstance(m, LazyBatchNorm)
        assert isinstance(m.weight, UninitializedParameter)
        assert isinstance(m.bias, UninitializedParameter)
        output = m(torch.randn(20, 100))
        assert isinstance(m, BatchNorm1d)
        assert not isinstance(m, LazyBatchNorm)
        assert m.weight.size() == torch.Size((100,))

        m = LazyBatchNorm()
        assert isinstance(m, LazyBatchNorm)
        assert isinstance(m.weight, UninitializedParameter)
        assert isinstance(m.bias, UninitializedParameter)
        output = m(torch.randn(20, 100, 35, 45))
        assert isinstance(m, BatchNorm2d)
        assert not isinstance(m, LazyBatchNorm)
        assert m.weight.size() == torch.Size((100,))

        m = LazyBatchNorm()
        assert isinstance(m, LazyBatchNorm)
        assert isinstance(m.weight, UninitializedParameter)
        assert isinstance(m.bias, UninitializedParameter)
        output = m(torch.randn(20, 100, 35, 45, 10))
        assert isinstance(m, BatchNorm3d)
        assert not isinstance(m, LazyBatchNorm)
        assert m.weight.size() == torch.Size((100,))


class TestLazyLayerNorm:
    def test_layer_norm(self):
        m = LazyLayerNorm()

        m(torch.empty(1, 2, 3))

        assert isinstance(m, LayerNorm)
        assert m.normalized_shape == (2, 3)
