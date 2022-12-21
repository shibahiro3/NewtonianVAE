#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, point_mass, ...
env=$1
# train, train_visdom, train_tensorboard, ... (python file)
trainer=$2


cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"


opts=(
	--config exec/config/$env.json5
	# --resume
	${@:3}
)

if [ "$trainer" == "train" ]; then
	python source/newtonianvae/train.py ${opts[@]}
else
	python source/view/$trainer.py ${opts[@]}
fi
