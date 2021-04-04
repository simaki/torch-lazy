import pytest
import torch
from torch.nn import Bilinear
from torch.nn.parameter import UninitializedParameter

from torch_lazy.nn import LazyBilinear


class TestLazyBilinear:
    """
    Test `LazyBilinear`.
    """

    def test_bilinear(self):
        m = LazyBilinear(10)
        assert isinstance(m.weight, UninitializedParameter)
        assert isinstance(m.bias, UninitializedParameter)

        input1 = torch.ones(5, 6)
        input2 = torch.ones(5, 7)

        m(input1, input2)

        assert isinstance(m, Bilinear)
        assert not isinstance(m, LazyBilinear)

        assert m.weight.size() == torch.Size((10, 6, 7))
        assert m.bias.size() == torch.Size((10,))

        y = m(input1, input2)
        assert torch.equal(
            torch.nn.functional.bilinear(input1, input2, m.weight, m.bias), y
        )
