#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# $1 = reacher2d, pointmass, pointmass_big, ...

cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"

python source/view/show_data.py \
	--path-data environment/$1/data \
	${@:2}
