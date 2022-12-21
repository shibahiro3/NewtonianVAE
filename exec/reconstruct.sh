#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, pointmass, pointmass_big, ... 
env=$1


cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"


opts=(
	--config exec/config/$env.json5
	# --path_model  environment/$env/saves
	# --path_result environment/$env/results
	
	--episodes 10
	# --fix_xmap_size 20
	# --movie_format gif
	${@:2}
)

python source/newtonianvae/reconstruct.py ${opts[@]}
