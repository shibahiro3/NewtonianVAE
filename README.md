# Scratch of NewtonianVAE

This is NOT an official implementation.

Still under development. Destructive changes may be made. But at this time, Reacher2D shows a high correlation (2022/12/13).

Paper:
* Original: https://arxiv.org/abs/2006.01959
* TS-NVAE: https://arxiv.org/abs/2203.05955

Referenced implementation
* VRNN: https://github.com/emited/VariationalRecurrentNeuralNetwork
* World Models: https://github.com/ctallec/world-models
* PlaNet (RSSM): https://github.com/Kaixhin/PlaNet

You can view the document from [index.html](docs/generated/index.html).

## Prerequisites
Install the necessary packages.
```bash
pip install -r requirements.txt
```

And install PyTorch manually by referring to the [official site](https://pytorch.org/).


### Tips
If you build the environment directly into your environment, it may not work properly. So I recommend using a virtual environment like [venv](https://docs.python.org/3/library/venv.html).


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
Model weights are saved per ``save_per_epoch``.


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
