#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, pointmass, pointmass_big, ... 
env=$1


cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"


opts=(
	--cf-eval environment/$env/cf/params_eval.json5
	--path-model environment/$env/saves
	--path-data environment/$env/data
	--path-result environment/$env/results
	--episodes 50
	${@:2}
)

python source/newtonianvae/correlation.py ${opts[@]}
