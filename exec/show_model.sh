#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, pointmass, pointmass_big, ... 
env=$1


cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"


# tensorboard --logdir=../log_[model name]

mkdir -p log_model

opts=(
	--cf environment/$env/cf/params.json5
	${@:2}
)

python -O source/view/show_model.py ${opts[@]}

