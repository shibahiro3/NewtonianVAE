#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# $1 = reacher2d, pointmass, pointmass_big, ...

cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"

python source/nvae/reconstruct.py \
	--cf-eval environment/$1/cf/params_eval.json5 \
	--path-model environment/$1/saves \
	--path-data environment/$1/data \
	--path-result environment/$1/result \
	--episodes 10 \
	${@:2}
	# --path-model reacher2d/saves_trained \
