# Scratch of NewtonianVAE
paper: https://arxiv.org/abs/2006.01959

## Prerequisites
To keep your environment clean, I recommend that you set up your virtual environment as follows:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools
```
Then install the necessary packages.
The package includes [mypython](https://github.com/SatohKazuto/mypython), part of my old "toolbox" that I have been creating.
I did not include it in this repository because many of them are not directly related to NewtonianVAE and I wanted to keep the environment separate.
```bash
pip install -r requirements.txt
```

Please install PyTorch manually by referring to the [official site](https://pytorch.org/).

### Create Task
Overwrite the two files in [dm_control](https://arxiv.org/abs/2006.12983) to create the Reacher-2D environment in the paper. In fact, we are only modifying the original files slightly. This enables to change the number of balls the range of motion of the joints.

Please do the following:
```bash
cd make_reacher2d
python override_reacher.py
```

## Run with Reacher-2D
These should be run under the [reacher2d](reacher2d) directory.

### Collect Data
To see what kind of data you can get before saving an episode
```bash
python collect_data.py --watch plt
```
By default, parameters can be adjusted by ```params_reacher2d.json5```.

If you want to save the data, please remove the --watch option.
```bash
python collect_data.py
```

### Train
```bash
python train.py
```
The reconstruction.py, described below, for example, requires a trained model to be loaded. If you want to run it immediately without learning it on your computer, please add the ```--path-model saves_trained/``` option.

### Reconstruction
Sequentially feed the trained model with $\mathbf{u}_{t-1}$ and $\mathbf{I}_t$ of the validation data to see how Reconstructed $\mathbf{I}_t$, etc., transitions.
```bash
python reconstruction.py
```

### Control
You can give the target image in the source code and see how it behaves.
```bash
python control.py
```
