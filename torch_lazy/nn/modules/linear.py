import torch
from torch.nn import Bilinear
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


class LazyBilinear(LazyModuleMixin, Bilinear):
    """
    A `torch.nn.Bilinear` module with lazy initialization.

    See `torch.nn.Bilinear` for more details:
    https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html

    Parameters
    ----------
    - out_features : int
        size of each output sample
    - bias: bool, default True
        If set to ``False``, the layer will not learn an additive bias.

    Examples
    --------
    >>> m = LazyBilinear(3)
    >>> m
    LazyBilinear(in1_features=0, in2_features=0, out_features=3, bias=True)

    >>> input1 = torch.empty(1, 4)
    >>> input2 = torch.empty(1, 5)
    >>> m(input1, input2).size()
    torch.Size([1, 3])
    >>> m
    Bilinear(in1_features=4, in2_features=5, out_features=3, bias=True)
    """

    cls_to_become = Bilinear
    weight: UninitializedParameter
    bias: UninitializedParameter

    def __init__(self, out_features: int, bias: bool = True):
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__(0, 0, out_features, bias=False)
        self.weight = UninitializedParameter()
        if bias:
            self.bias = UninitializedParameter()

    def reset_parameters(self) -> None:
        if (
            not self.has_uninitialized_params()
            and self.in1_features != 0
            and self.in2_features != 0
        ):
            super().reset_parameters()

    def initialize_parameters(self, input1, input2) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in1_features = input1.shape[-1]
                self.in2_features = input2.shape[-1]
                self.weight.materialize(
                    (self.out_features, self.in1_features, self.in2_features)
                )
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
