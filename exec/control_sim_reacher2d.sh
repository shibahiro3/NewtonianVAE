#!/usr/bin/env bash

cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"

python source/simulation/control.py \
	--cf-eval environment/reacher2d/cf/params_eval.json5 \
	--cf-simenv environment/reacher2d/cf/params_env.json5 \
	--path-model environment/reacher2d/saves \
	--path-result environment/$1/result \
	--goal-img environment/reacher2d/observation_imgs/obs_green.npy \
	--episodes 5 \
	--alpha 0.2 \
	--fix-xmap-size 12 \
	$@
