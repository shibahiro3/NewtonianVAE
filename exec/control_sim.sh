#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, pointmass, pointmass_big, ... 
env=$1


cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"


if [ "$env" == "reacher2d" ]; then
	domain="reacher"
else
	domain=$env
fi

override=$workspaceFolder/environment/$env/override

python source/simulation/override.py $domain $override


opts=(
	--cf-eval     exec/config/${env}_eval.json5
	--cf-simenv   exec/config/${env}_env.json5
	--path-model  environment/$env/saves
	--path-result environment/$env/results
	--goal-img    environment/$env/observation_imgs/obs_green.npy
	--episodes 10
	--fix-xmap-size 25

	--alpha 0.1
	${@:2}
)

python source/simulation/control.py ${opts[@]}
