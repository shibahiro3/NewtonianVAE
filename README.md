# Scratch of NewtonianVAE

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This is NOT an official implementation.

Point Mass and Reacher2D show high correlation between latent space and physical location.
But I'm still working on fixing some details. Destructive changes may be made (2022/12/14).

You can view this document from [index.html](docs/generated/index.html).

Paper

- Original: https://arxiv.org/abs/2006.01959
- TS-NVAE: https://arxiv.org/abs/2203.05955

Other References (All implementations are PyTorch)

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

## Prerequisites

Install the necessary packages.

```bash
pip install -r requirements.txt
```

And install PyTorch manually by referring to the [official site](https://pytorch.org/).

### Tips

The installation of matplotlib, opencv, and their associated Qt is likely to be environmentally dependent.

## Run

These should be run under the [exec](exec) directory.

### Collect Data

To see what kind of data you can get before saving an episode

```bash
./collect.sh [environment (directory name)] --watch plt (or render)
```

Example:

```bash
./collect.sh reacher2d --watch plt
```

If you want to save the data, please remove the --watch option.

```bash
./collect.sh [environment (directory name)]
```

### Train

```bash
./train.sh [environment (directory name)] train
```

If [visdom](https://github.com/fossasia/visdom) is used:

```bash
./train.sh [environment (directory name)] train_visdom
```

Model weights are saved per `save_per_epoch`.

### Reconstruction

Sequentially feed the trained model with $\mathbf{u}_{t-1}$ and $\mathbf{I}_t$ of the validation data to see how Reconstructed $\mathbf{I}_t$, etc., transitions.

```bash
./reconstruction.sh [environment (directory name)]
```

### Control in simulation

You can give the target image in the source code and see how it behaves.

```bash
./control_sim.sh [environment (directory name)]
```

## Acknowledgements

[@ItoMasaki](https://github.com/ItoMasaki) helped me understand the detailed formulas.
