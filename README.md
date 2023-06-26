# NewtonianVAE

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This is a PyTorch implementation that expands on [NewtonianVAE](https://arxiv.org/abs/2006.01959).

## Prerequisites

Install the necessary packages.

```bash
python -m pip install -r requirements.txt
```

### Tips
- Recommend using a virtual environment such as [venv](https://docs.python.org/3/library/venv.html)
- The installation of matplotlib, opencv, and their associated Qt is likely to be environmentally dependent.
- You may need to install PyTorch according to [official site](https://pytorch.org/).

## Run

These should be run under the [src/exec](src/exec) directory. To see examples other than those listed below, run `python xxx.py -h`.

### Create Data

Example:

```bash
python create_data.py -c config/reacher2d.json5 --mode save-data --save-dir data/reacher2d
```

If you want to see what kind of data you are looking for:

```bash
python create_data.py -c config/reacher2d.json5 --mode show-plt
```

### Train

Example:

```bash
python train.py -c config/reacher2d.json5
```

### Correlation

Example:

```bash
python correlation.py -c config/reacher2d.json5
```

## References

- TS-NVAE
  - paper: https://arxiv.org/abs/2203.05955
- VRNN
  - paper: https://arxiv.org/abs/1506.02216
  - impl: https://github.com/emited/VariationalRecurrentNeuralNetwork
- World Models
  - paper: https://arxiv.org/abs/1803.10122
  - impl: https://github.com/ctallec/world-models
- PlaNet (RSSM):
  - paper: https://arxiv.org/abs/1811.04551
  - impl: https://github.com/Kaixhin/PlaNet
- Spatial Broadcast Decoder
  - paper: https://arxiv.org/abs/1901.07017
  - impl: https://github.com/dfdazac/vaesbd
