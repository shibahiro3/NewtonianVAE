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

# path_data=environment/$env/data
path_data=environment/$env/data_center

mkdir -p $path_data
cp -fr $override $path_data
chmod 444 $path_data/override/*


opts=(
	--cf-simenv   exec/config/${env}_env.json5
	--path-data   $path_data
	--episodes 1600 # for train: 1000, for validation: 300, for test: 300
	# --save_anim
	${@:2}
)

python source/simulation/collect_data.py ${opts[@]}
