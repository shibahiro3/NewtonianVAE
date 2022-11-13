#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# $1 = reacher2d, pointmass, pointmass_big, ...

cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"

python source/nvae/show_loss.py \
	--path-model environment/$1/saves \
	${@:2}
