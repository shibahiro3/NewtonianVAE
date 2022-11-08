cd ../
python simulation/control.py \
	--cf-eval pointmass_big/cf/params_eval.json5 \
	--cf-simenv pointmass_big/cf/params_env.json5 \
	--path-model pointmass_big/saves \
	--goal-img pointmass_big/observation_imgs/obs_315.npy \
	--episodes 5 \
	--alpha 0.5 \
	--fix-xmap-size 300 \
	$@
