import torch
from torch.nn import LazyLinear
from torch.nn import Linear

from torch_lazy.nn import MLP
from torch_lazy.nn import LazyMLP


class TestLazyMLP:
    """
    Test `LazyMLP`.
    """

    def test_mlp(self):
        m = LazyMLP(10, n_units=32)
        assert isinstance(m[0], LazyLinear)

        m(torch.empty(1, 5))

        assert isinstance(m, MLP)
        assert isinstance(m[0], Linear)
        assert m[0].weight.size() == torch.Size((32, 5))
        assert m[0].bias.size() == torch.Size((32,))
