cd ../
workspaceFolder=$(pwd)
export PYTHONPATH="$workspaceFolder/source"

python source/simulation/control.py \
	--cf-eval environment/pointmass_big/cf/params_eval.json5 \
	--cf-simenv environment/pointmass_big/cf/params_env.json5 \
	--path-model environment/pointmass_big/saves \
	--goal-img environment/pointmass_big/observation_imgs/obs_315.npy \
	--episodes 5 \
	--alpha 0.5 \
	--fix-xmap-size 300 \
	$@
