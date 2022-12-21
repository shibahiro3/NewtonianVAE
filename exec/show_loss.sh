#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, pointmass, pointmass_big, ... 
env=$1


cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"


opts=(
	--path_model  environment/$env/saves
	--path_result environment/$env/results
	--start_iter 100
	${@:2}
)

python source/newtonianvae/show_loss.py ${opts[@]}

