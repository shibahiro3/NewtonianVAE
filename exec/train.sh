#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, pointmass, pointmass_big, ... 
env=$1
# train, train_visdom, train_tensorboard, ... (python file)
trainer=$2


cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"


opts=(
	--cf environment/$env/cf/params.json5
	--path-save environment/$env/saves
	# --path-data environment/$env/data
	--path-data environment/$env/data_handmade2
	# --resume
	${@:3}
)

if [ "$trainer" == "train" ]; then
	python source/newtonianvae/$trainer.py ${opts[@]}
else
	python source/view/$trainer.py ${opts[@]}
fi
