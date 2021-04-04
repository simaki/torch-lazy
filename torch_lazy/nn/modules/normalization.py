import torch
from torch.nn import LayerNorm
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


class LazyLayerNorm(LazyModuleMixin, Module):
    """
    A `LayerNorm` with lazy initialization.

    See `LayerNorm` for details:
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm

    Parameters
    ----------
    - eps : float, default 1e-5
        a value added to the denominator for numerical stability.
    - elementwise_affine : bool, default True
        a boolean value that when set to True, this module has learnable per-element
        affine parameters initialized to ones (for weights) and zeros (for biases).

    Examples
    --------
    >>> input = torch.randn(20, 5, 10, 10)
    >>> # With Learnable Parameters
    >>> m = LazyLayerNorm()
    >>> # Without Learnable Parameters
    >>> m = LazyLayerNorm(elementwise_affine=False)
    >>> m
    LazyLayerNorm(eps=1e-05, elementwise_affine=False)
    >>> output = m(input)
    >>> output.size()
    torch.Size([20, 5, 10, 10])
    >>> m
    LayerNorm((5, 10, 10), eps=1e-05, elementwise_affine=False)
    """

    cls_to_become = LayerNorm

    def __init__(self, eps=1e-5, elementwise_affine=True) -> None:
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def extra_repr(self):
        return "eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )

    def initialize_parameters(self, input) -> None:
        self.normalized_shape = tuple(input.size()[1:])
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(self.normalized_shape)
                self.bias.materialize(self.normalized_shape)
