# M-NewtonianVAE

This is an extension of [NewtonianVAE](https://arxiv.org/abs/2006.01959).

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

### Create Data

Example:

```bash
./src/create_data.py -c config/sim/reacher2d.json5 --mode save-data --save-dir data/reacher2d
```

If you want to see what kind of data you are looking for:

```bash
./src/create_data.py -c CONFIG_FILE.json5 --mode show-plt
```

### Train

```bash
./src/train.py -c CONFIG_FILE.json5
```

### Correlation

```bash
./src/correlation.py -c CONFIG_FILE.json5
```
