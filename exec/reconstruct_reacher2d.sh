cd ../
python nvae/reconstruct.py \
	--cf-eval reacher2d/cf/params_eval.json5 \
	--path-model reacher2d/saves \
	--episodes 10 \
	$@
	# --path-model reacher2d/saves_trained \
