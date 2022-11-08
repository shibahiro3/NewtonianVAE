cd ../pointmass/override
python override.py
cd ../

# Paper:
# To train the models, we generate 1000 random se-
# quences with 100 time-steps for the point mass and
# reacher-2D systems, and 30 time-steps for the fetch-3D
# system.

cd ../
python simulation/collect_data.py \
	--cf pointmass/cf/params.json5 \
	--cf-simenv pointmass/cf/params_env.json5 \
	--episodes 1050 \
	$@
