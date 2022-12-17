#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, point_mass, ...
env=$1


cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"


# Paper:
# To train the models, we generate 1000 random se-
# quences with 100 time-steps for the point mass and
# reacher-2D systems, and 30 time-steps for the fetch-3D
# system.

opts=(
	--cf exec/config/$env.json5
	--episodes 1400  # for train: 1000, for validation: 200, for test: 200
	# --save_anim
	${@:2}
)

python source/simulation/collect_data.py ${opts[@]}
