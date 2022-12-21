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

	# --goal_img    environment/$env/observation_imgs/obs_red.npy
	# --goal_img    environment/$env/observation_imgs/obs_green.npy
	--goal_img    environment/$env/observation_imgs/obs_yellow.npy

	--episodes 10
	--fix_xmap_size 2
	--env_domain $env
	
	--alpha 0.3
	--steps 200
	
	# --movie_format gif
	${@:2}
)

python source/simulation/control.py ${opts[@]}
