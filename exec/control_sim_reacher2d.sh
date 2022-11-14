#!/usr/bin/env bash

cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"

env="reacher2d"

python source/simulation/control.py \
	--cf-eval environment/$env/cf/params_eval.json5 \
	--cf-simenv environment/$env/cf/params_env.json5 \
	--path-model environment/$env/saves \
	--path-result environment/$env/results \
	--goal-img environment/$env/observation_imgs/obs_green.npy \
	--episodes 10 \
	--alpha 0.05 \
	--fix-xmap-size 12 \
	$@
