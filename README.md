# torch-lazy

Lazy Modules for PyTorch

## Install

```sh
pip install git+https://github.com/simaki/torch-lazy
```

## Lazy Modules

### `LazyBilinear`

A [`torch.nn.Bilinear`](https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html) module with lazy initialization.

```python
m = LazyBilinear(out_features=3)
m
# LazyBilinear(in1_features=0, in2_features=0, out_features=3, bias=True)

input1 = torch.empty(1, 4)
input2 = torch.empty(1, 5)
output = m(input1, input2)
m
# Bilinear(in1_features=4, in2_features=5, out_features=3, bias=True)
```

### `LazyMLP`

A multi-layer perceptron module with lazy initialization.

```python
m = LazyMLP(out_features=3)
m
# LazyMLP(
#     (0): LazyLinear(in_features=0, out_features=32, bias=True)
#     (1): ReLU()
#     (2): LazyLinear(in_features=0, out_features=32, bias=True)
#     (3): ReLU()
#     (4): LazyLinear(in_features=0, out_features=3, bias=True)
# )

input = torch.empty(1, 2)
m
# MLP(
#     (0): Linear(in_features=2, out_features=32, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=32, out_features=32, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=32, out_features=3, bias=True)
# )
```

### `LazyBatchNorm`

A [`torch.nn.BatchNorm[1-3]d`](https://pytorch.org/docs/stable/nn.html#normalization-layers) module with lazy initialization.

```python
m = LazyBatchNorm()
m
LazyBatchNorm(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

input = torch.randn(20, 100, 35, 45)
output = m(input)

m
# BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

### `LazyLayerNorm`

A [`torch.nn.LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) module with lazy initialization.

```python
m = LazyLayerNorm()
m
# LazyLayerNorm(eps=1e-05, elementwise_affine=False)

input = torch.randn(20, 5, 10, 10)
output = m(input)

m
# LayerNorm((5, 10, 10), eps=1e-05, elementwise_affine=True)
```
