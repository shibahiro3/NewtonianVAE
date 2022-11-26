#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# reacher2d, point_mass, ...
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


# Paper:
# To train the models, we generate 1000 random se-
# quences with 100 time-steps for the point mass and
# reacher-2D systems, and 30 time-steps for the fetch-3D
# system.

path_data=environment/$env/data
opts=(
	--cf-simenv   exec/config/${env}_env.json5
	--path-data   $path_data
	--episodes 1050 # for train: 1000, for eval: 50
	# --save_anim
	${@:2}
)

python source/simulation/collect_data.py ${opts[@]}

cp -fr $override $path_data
chmod 444 $path_data/override/*
