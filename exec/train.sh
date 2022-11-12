# This file is just a sample.
# You can change the arguments according to your content.

# $1 = reacher2d, pointmass, pointmass_big, ...

cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"

python source/nvae/train.py \
	--cf environment/$1/cf/params.json5 \
	--path-save environment/$1/saves \
	--path-data environment/$1/data \
	${@:2}
