import pytest

import torch
from torch_lazy.nn import LazyLayerNorm
from torch.nn import LayerNorm


class TestLazyLayerNorm:

    def test_layer_norm(self):
        m = LazyLayerNorm()

        m(torch.empty(1, 2, 3))

        assert isinstance(m, LayerNorm)
        assert m.normalized_shape == (2, 3)
       
