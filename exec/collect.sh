#!/usr/bin/env bash

# This file is just a sample.
# You can change the arguments according to your content.

# $1 = reacher2d, pointmass, pointmass_big, ...

cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"

if [ "$1" == "reacher2d" ]; then
	domain="reacher"
elif [ "$1" == "pointmass" ] || [ "$1" == "pointmass_big" ]; then
	domain="point_mass"
else
	echo "Unknown environment"
	exit 0
fi

python source/simulation/override.py $domain $workspaceFolder/environment/$1/override


# Paper:
# To train the models, we generate 1000 random se-
# quences with 100 time-steps for the point mass and
# reacher-2D systems, and 30 time-steps for the fetch-3D
# system.

python source/simulation/collect_data.py \
	--cf environment/$1/cf/params.json5 \
	--cf-simenv environment/$1/cf/params_env.json5 \
	--path-data environment/$1/data \
	--path-result environment/$1/result \
	--episodes 1050 \
	${@:2}
