cd ../
python simulation/control.py \
	--cf-eval reacher2d/cf/params_eval.json5 \
	--cf-simenv reacher2d/cf/params_env.json5 \
	--path-model reacher2d/saves \
	--goal-img reacher2d/observation_imgs/obs_green.npy \
	--episodes 5 \
	--alpha 0.2 \
	--fix-xmap-size 12 \
	$@
