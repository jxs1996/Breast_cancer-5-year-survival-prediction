cp ../best_param_model_data.data ./old.best_param_model_data.data
cp ../Final.model.data ./old.Final.model.data
cp ../data_normalize.data ./old.new_feature_build_data.data
cat ../config.ini >>best_fs_num_config.ini
head -n 23 best_fs_num_config.ini > tmp; cat ../config.ini >> tmp; mv tmp best_fs_num_config.ini;
