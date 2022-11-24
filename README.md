# Scratch of NewtonianVAE

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
By default, parameters can be adjusted by ``**/params_env.json5``.

If you want to save the data, please remove the --watch option.
```bash
./collect.sh [environment (directory name)]
```


### Train
```bash
python train.py
```
Model weights are saved per ``save_per_epoch``.
The reconstruction.py, described below, for example, requires a trained model to be loaded via ``--path-model [Directory path for models managed by date and time]``. 


### Reconstruction
Sequentially feed the trained model with $\mathbf{u}_{t-1}$ and $\mathbf{I}_t$ of the validation data to see how Reconstructed $\mathbf{I}_t$, etc., transitions.
```bash
./reconstruction.sh [environment (directory name)]
```


### Control
You can give the target image in the source code and see how it behaves.
```bash
./control_*.sh
```
