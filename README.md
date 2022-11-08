# Scratch of NewtonianVAE
paper: https://arxiv.org/abs/2006.01959


## Prerequisites
Install the necessary packages.
The package includes [mypython](https://github.com/SatohKazuto/mypython), part of my old "toolbox" that I have been creating.
I did not include it in this repository because many of them are not directly related to NewtonianVAE and I wanted to keep the environment separate.

It is recommended that a virtual environment be used.
```bash
pip install -r requirements.txt
```

And install PyTorch manually by referring to the [official site](https://pytorch.org/).


## Run
These should be run under the [exec](exec) directory.

### Collect Data
To see what kind of data you can get before saving an episode
```bash
./collect_*.sh --watch plt (or render)
```
By default, parameters can be adjusted by ```**/params_env.json5```.

If you want to save the data, please remove the --watch option.
```bash
./collect_*.sh
```

### Train
```bash
python train.py
```
The reconstruction.py, described below, for example, requires a trained model to be loaded. If you want to run it immediately without learning it on your computer, please add the ```--path-model saves_trained/``` option.

### Reconstruction
Sequentially feed the trained model with $\mathbf{u}_{t-1}$ and $\mathbf{I}_t$ of the validation data to see how Reconstructed $\mathbf{I}_t$, etc., transitions.
```bash
./reconstruction_*.sh
```

### Control
You can give the target image in the source code and see how it behaves.
```bash
./control_*.sh
```
