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
elif [ "$env" == "pointmass" ] || [ "$env" == "pointmass_big" ]; then
	domain="point_mass"
else
	echo "Unknown environment"
	exit 0
fi

python source/simulation/override.py $domain $workspaceFolder/environment/$env/override


# Paper:
# To train the models, we generate 1000 random se-
# quences with 100 time-steps for the point mass and
# reacher-2D systems, and 30 time-steps for the fetch-3D
# system.


opts=(
	--cf-simenv environment/$env/cf/params_env.json5
	# --path-data environment/$env/data
	--path-data environment/$env/data_handmade2
	--path-result environment/$env/results
	--episodes 1050 # for train: 1000, for eval: 50
	--position-size 3 # position_wrap: null
	# --position-size 0.35 # position_wrap: "endeffector"
	${@:2}
)

python source/simulation/collect_data.py ${opts[@]}
