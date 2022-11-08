cd ../
python nvae/reconstruct.py \
	--cf-eval pointmass_big/cf/params_eval.json5 \
	--path-model pointmass_big/saves \
	--episodes 10 \
	$@
