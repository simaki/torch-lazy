from copy import deepcopy

import torch
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn.modules.lazy import LazyModuleMixin


class MLP(Sequential):
    """
    Multi-layer perceptron.

    Parameters
    ----------
    - in_features : int
        size of each input sample.
    - out_features : int
        size of each output sample.
    - n_layers : int
        number of hidden layers.
    - n_units : int
        number of units in each hidden layer.
    - activation : Module
        activation module in hidden layers.

    Shape
    -----
    - Input : (N, *, H_in)
        where where * means any number of
        additional dimensions and `H_in = in_features`.
    - Output : (N, *, H_out)
        where all but the last dimension
        are the same shape as the input and `H_out = out_features`.

    Examples
    --------
    >>> x = torch.empty(1, 2)
    >>> m = MLP(2, 3)
    >>> m
    MLP(
      (0): Linear(in_features=2, out_features=32, bias=True)
      (1): ReLU()
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ReLU()
      (4): Linear(in_features=32, out_features=3, bias=True)
    )
    >>> m(x).size()
    torch.Size([1, 3])
    """

    def __init__(
        self, in_features, out_features, n_layers=2, n_units=32, activation=ReLU()
    ):

        layers = []
        for i_layer in range(n_layers):
            layers.append(Linear(in_features if i_layer == 0 else n_units, n_units))
            layers.append(deepcopy(activation))
        layers.append(Linear(n_units, out_features))

        super().__init__(*layers)


class LazyMLP(LazyModuleMixin, Sequential):
    """
    A feed-forward neural network.

    Number of input features is lazily determined.

    Parameters
    ----------
    - out_features : int
        size of each output sample.
    - n_layers : int
        number of hidden layers.
    - n_units : int
        number of units in each hidden layer.
    - activation : Module
        activation module in hidden layers.

    Shape
    -----
    - Input : (N, *, H_in)
        where where * means any number of
        additional dimensions and `H_in = in_features`.
    - Output : (N, *, H_out)
        where all but the last dimension
        are the same shape as the input and `H_out = out_features`.

    Examples
    --------
    >>> m = LazyMLP(3)
    >>> m
    LazyMLP(
      (0): LazyLinear(in_features=0, out_features=32, bias=True)
      (1): ReLU()
      (2): LazyLinear(in_features=0, out_features=32, bias=True)
      (3): ReLU()
      (4): LazyLinear(in_features=0, out_features=3, bias=True)
    )

    >>> x = torch.empty(1, 2)
    >>> m(x).size()
    torch.Size([1, 3])
    >>> m
    MLP(
      (0): Linear(in_features=2, out_features=32, bias=True)
      (1): ReLU()
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ReLU()
      (4): Linear(in_features=32, out_features=3, bias=True)
    )
    """

    cls_to_become = MLP

    def __init__(self, out_features=1, n_layers=2, n_units=32, activation=ReLU()):
        layers = []
        for _ in range(n_layers):
            layers.append(LazyLinear(n_units))
            layers.append(deepcopy(activation))
        layers.append(LazyLinear(out_features))

        super().__init__(*layers)

    def initialize_parameters(self, *args, **kwargs):
        pass
