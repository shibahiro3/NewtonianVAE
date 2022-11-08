cd ../pointmass_big/override
python override.py
cd ../

# Paper:
# To train the models, we generate 1000 random se-
# quences with 100 time-steps for the point mass and
# reacher-2D systems, and 30 time-steps for the fetch-3D
# system.

cd ../
python simulation/collect_data.py \
	--cf pointmass_big/cf/params.json5 \
	--cf-simenv pointmass_big/cf/params_env.json5 \
	--episodes 120 \
	$@
