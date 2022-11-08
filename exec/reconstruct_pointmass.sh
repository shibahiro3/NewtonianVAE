cd ../
python nvae/reconstruct.py \
	--cf-eval pointmass/cf/params_eval.json5 \
	--path-model pointmass/saves \
	--episodes 10 \
	$@
