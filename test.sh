#!/bin/sh

python3 -m pytest --doctest-modules torch_lazy
python3 -m pytest --doctest-modules tests

python3 -m black --check torch_lazy || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m black torch_lazy
python3 -m isort --check --force-single-line-imports torch_lazy || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m isort --force-single-line-imports torch_lazy
