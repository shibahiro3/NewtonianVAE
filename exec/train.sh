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
	--cf        exec/config/$env.json5
	# --path-data environment/$env/data
	--path-data environment/$env/data_center
	--path-save environment/$env/saves
	# --resume
	${@:3}
)

if [ "$trainer" == "train" ]; then
	python source/newtonianvae/train.py ${opts[@]}
else
	python source/view/$trainer.py ${opts[@]}
fi
