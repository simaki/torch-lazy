import torch
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm3d
from torch.nn import LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer
from torch.nn.parameter import UninitializedParameter


class LazyBatchNorm(LazyModuleMixin, _BatchNorm):
    """

    Parameters
    ----------
    - eps
    - momentum
    - affine
    - track_running_stats
    - process_group

    Examples
    --------
    >>> m = LazyBatchNorm()
    >>> m
    LazyBatchNorm(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >>> output = m(torch.randn(20, 100))
    >>> m
    BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> m = LazyBatchNorm()
    >>> output = m(torch.randn(20, 100, 35, 45))
    >>> m
    BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> m = LazyBatchNorm()
    >>> output = m(torch.randn(20, 100, 35, 45, 10))
    >>> m
    BatchNorm3d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    """

    # cls_to_become is determined in initialize_parameters
    cls_to_become = None

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group=None,
    ) -> None:
        super().__init__(0, eps, momentum, affine, track_running_stats)

        if self.affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", UninitializedBuffer())
            self.register_buffer("running_var", UninitializedBuffer())
            self.register_buffer("num_batches_tracked", UninitializedBuffer())
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3, 4, 5):
            raise ValueError(
                "expected 2D, 3D, 4D or 5D input (got {}D input)".format(input.dim())
            )

    def initialize_parameters(self, input) -> None:
        self._check_input_dim(input)

        if input.dim() == 2 or input.dim() == 3:
            self.cls_to_become = BatchNorm1d
        if input.dim() == 4:
            self.cls_to_become = BatchNorm2d
        if input.dim() == 5:
            self.cls_to_become = BatchNorm3d

        self.num_features = input.size(1)

        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(self.num_features)
                self.bias.materialize(self.num_features)

        if (
            isinstance(self.running_mean, UninitializedBuffer)
            or isinstance(self.running_var, UninitializedBuffer)
            or isinstance(self.num_batches_tracked, UninitializedBuffer)
        ):
            with torch.no_grad():
                self.register_buffer("running_mean", torch.zeros(self.num_features))
                self.register_buffer("running_var", torch.ones(self.num_features))
                self.register_buffer(
                    "num_batches_tracked", torch.tensor(0, dtype=torch.long)
                )


class LazyBatchNorm1d(LazyBatchNorm):
    """

    >>> m = LazyBatchNorm1d()
    >>> m
    LazyBatchNorm1d(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> input = torch.randn(20, 100)
    >>> output = m(input)
    >>> m
    BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    """

    cls_to_become = BatchNorm1d

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


class LazyBatchNorm2d(LazyBatchNorm):
    """
    >>> m = LazyBatchNorm2d()
    >>> m
    LazyBatchNorm2d(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> input = torch.randn(20, 100, 35, 45)
    >>> output = m(input)
    >>> m
    BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    """

    cls_to_become = BatchNorm2d

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class LazyBatchNorm3d(LazyBatchNorm):
    """
    >>> m = LazyBatchNorm3d()
    >>> m
    LazyBatchNorm3d(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> input = torch.randn(20, 100, 35, 45, 10)
    >>> output = m(input)
    >>> m
    BatchNorm3d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    """

    cls_to_become = BatchNorm3d

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))


class LazyLayerNorm(LazyModuleMixin, LayerNorm):
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
    LazyLayerNorm((0,), eps=1e-05, elementwise_affine=False)
    >>> output = m(input)
    >>> output.size()
    torch.Size([20, 5, 10, 10])
    >>> m
    LayerNorm((5, 10, 10), eps=1e-05, elementwise_affine=False)
    """

    cls_to_become = LayerNorm

    def __init__(self, eps=1e-5, elementwise_affine=True) -> None:
        super().__init__(0, eps, elementwise_affine)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def initialize_parameters(self, input) -> None:
        self.normalized_shape = tuple(input.size()[1:])
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(self.normalized_shape)
                self.bias.materialize(self.normalized_shape)
