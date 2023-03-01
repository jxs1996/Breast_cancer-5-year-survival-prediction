import configparser
import logging
import json
from resample import ReSampleMethod
import hyperopt
from hyperopt import hp
from sklearn.metrics import f1_score
from scipy.misc import derivative
from hyperopt import hp, tpe, fmin, Trials
from sklearn.model_selection import KFold
import numpy as np

class BestParamCV(object):
	def __init__(self):
		pass

	'''
	设置日志格式
	'''
	def logging_config(self,msg,log_type):
		LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
		logging.basicConfig(filename='duty.log', level=logging.INFO, format=LOG_FORMAT)
		if(log_type == 'error'):
			logging.error(msg)
		elif(log_type == 'info'):
			logging.info(msg)

	
	def BGI_XGBClassifier(self,configFile,data):
		import numpy as np
		import pandas as pd
		from sklearn.model_selection import KFold
		import xgboost as xgb 
		from xgboost.sklearn import XGBClassifier
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import cross_validate
		cf = configparser.ConfigParser()
		cf.read(configFile)
		
		# 初始化参数
		learning_rate = 		float(cf.get("model_param_init","xgb_learning_rate"))
		n_estimators = 			int(cf.get("model_param_init","xgb_n_estimators"))
		max_depth = 			int(cf.get("model_param_init","xgb_max_depth"))
		min_child_weight = 		float(cf.get("model_param_init","xgb_min_child_weight"))
		gamma = 				float(cf.get("model_param_init","xgb_gamma"))
		subsample = 			float(cf.get("model_param_init","xgb_subsample"))
		colsample_bytree = 		float(cf.get("model_param_init","xgb_colsample_bytree"))
		objective = 			cf.get("model_param_init","xgb_objective")
		reg_alpha = 			float(cf.get("model_param_init","xgb_reg_alpha"))
		seed = 					int(cf.get("model_param_init","xgb_seed"))
		early_stopping_rounds = int(cf.get("model_param_init","xgb_early_stopping_rounds"))
		n_jobs = 				int(cf.get("model_param_init","xgb_n_jobs"))
		n_folds = 				int(cf.get("model_param_init","xgb_n_folds"))

		self.logging_config("[Param] learning_rate:{}".format(learning_rate),'info')
		self.logging_config("[Param] n_estimators:{}".format(n_estimators),'info')
		self.logging_config("[Param] max_depth:{}".format(max_depth),'info')
		self.logging_config("[Param] min_child_weight:{}".format(min_child_weight),'info')
		self.logging_config("[Param] gamma:{}".format(gamma),'info')
		self.logging_config("[Param] subsample:{}".format(subsample),'info')
		self.logging_config("[Param] colsample_bytree:{}".format(colsample_bytree),'info')
		self.logging_config("[Param] objective:{}".format(objective),'info')
		self.logging_config("[Param] reg_alpha:{}".format(reg_alpha),'info')
		self.logging_config("[Param] seed:{}".format(seed),'info')
		self.logging_config("[Param] early_stopping_rounds:{}".format(early_stopping_rounds),'info')
		self.logging_config("[Param] n_jobs:{}".format(n_jobs),'info')

		model = XGBClassifier(learning_rate = learning_rate, # this is analogue to eta, Typical: 0.01-0.2
						 n_estimators = n_estimators, # the number of iterations/trees.  
						 max_depth = max_depth, # Typical values: 3-10 
						 min_child_weight = min_child_weight,
						 gamma = gamma, # min_split_loss
						 subsample = subsample, # the fraction of observations to be randomly samples for each tree. 0.5-1
						 colsample_bytree = colsample_bytree, # the fraction of columns to be randomly samples for each tree. 0.5-1
						 objective= objective, 
						 reg_alpha = reg_alpha,
						 seed = seed, early_stopping_rounds = early_stopping_rounds, n_jobs=n_jobs)
		
		# 判断是否需要做超参
		if cf.get("model_param_init","model_pre_params") == 'True':
			# 寻找最优超参
			best_est= self.BGI_train_cv(x_train=data['x'], y_train=data['y'], init_estimator=model, n_folds=n_folds, method='xgb', scoring='f1_weighted')
			best_params = best_est.get_params()
			best_params['num_boost_round'] = best_params['n_estimators']
			self.logging_config("[Best Param For Pre CV] {}".format(json.dumps(best_params)),'info')
			# 获取最优超参
			model = xgb.XGBClassifier(**best_params)

		return model




	# RFclassifier
	def BGI_RFClassifier(self,configFile,data):
		import numpy as np
		import pandas as pd 
		from sklearn.model_selection import KFold
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import cross_validate

		
		cf = configparser.ConfigParser()
		cf.read(configFile)
		n_estimators =			 int(cf.get("model_param_init","rdf_n_estimators"))
		criterion =				cf.get("model_param_init","rdf_criterion")
		max_depth =				int(cf.get("model_param_init","rdf_max_depth"))
		min_samples_split =		int(cf.get("model_param_init","rdf_min_samples_split"))
		min_samples_leaf =		 int(cf.get("model_param_init","rdf_min_samples_leaf"))
		min_weight_fraction_leaf = float(cf.get("model_param_init","rdf_min_weight_fraction_leaf"))
		# max_leaf_nodes =		   int(cf.get("model_param_init","rdf_max_leaf_nodes"))
		min_impurity_decrease =	float(cf.get("model_param_init","rdf_min_impurity_decrease"))
		#min_impurity_split =	   float(cf.get("model_param_init","rdf_min_impurity_split"))
		# bootstrap =				bool(cf.get("model_param_init","rdf_bootstrap"))
		# oob_score =				bool(cf.get("model_param_init","rdf_oob_score"))
		n_jobs =				   int(cf.get("model_param_init","rdf_n_jobs"))
		random_state =			 int(cf.get("model_param_init","rdf_random_state"))
		verbose =				  int(cf.get("model_param_init","rdf_verbose"))
		# warm_start =			   bool(cf.get("model_param_init","rdf_warm_start"))
		ccp_alpha =				float(cf.get("model_param_init","rdf_ccp_alpha"))
		#max_samples =			  float(cf.get("model_param_init","rdf_max_samples"))
		n_folds =				  int(cf.get("model_param_init","rdf_n_folds"))
	  
		self.logging_config("[Param] criterion:{}".format(criterion),'info')
		self.logging_config("[Param] max_depth:{}".format(max_depth),'info')
		self.logging_config("[Param] n_estimators:{}".format(n_estimators),'info')
		self.logging_config("[Param] min_samples_split:{}".format(min_samples_split),'info')
		self.logging_config("[Param] min_samples_leaf:{}".format(min_samples_leaf),'info')
		self.logging_config("[Param] min_weight_fraction_leaf:{}".format(min_weight_fraction_leaf),'info')
		# self.logging_config("[Param] max_features:{}".format(max_features),'info')
		# self.logging_config("[Param] max_leaf_nodes:{}".format(max_leaf_nodes),'info')
		self.logging_config("[Param] min_impurity_decrease:{}".format(min_impurity_decrease),'info')
		#self.logging_config("[Param] min_impurity_split:{}".format(min_impurity_split),'info')
		# self.logging_config("[Param] bootstrap:{}".format(bootstrap),'info')
		# self.logging_config("[Param] oob_score:{}".format(oob_score),'info')
		self.logging_config("[Param] n_jobs:{}".format(n_jobs),'info')
		self.logging_config("[Param] random_state:{}".format(random_state),'info')
		self.logging_config("[Param] verbose:{}".format(verbose),'info')
		# self.logging_config("[Param] warm_start:{}".format(warm_start),'info')
		self.logging_config("[Param] ccp_alpha:{}".format(ccp_alpha),'info')
		#self.logging_config("[Param] max_samples:{}".format(max_samples),'info')
		self.logging_config("[Param] n_folds:{}".format(n_folds),'info')
		
		model = RandomForestClassifier(n_estimators = n_estimators,
								   criterion = criterion,
								   min_samples_split = min_samples_split,
								   min_samples_leaf = min_samples_leaf,
								   min_weight_fraction_leaf = min_weight_fraction_leaf,
								#    max_features = max_features,
								#    max_leaf_nodes = max_leaf_nodes,
								   min_impurity_decrease = min_impurity_decrease,
								#    min_impurity_split = None,
								   bootstrap = True,
								   oob_score = True,
								   n_jobs = n_jobs,
								   random_state = random_state,
								   verbose = verbose,
								#    warm_start = False,
								   ccp_alpha = ccp_alpha,
								#    max_samples = None,
								   max_depth = max_depth)
		
		# 判断是否需要做超参
		if cf.get("model_param_init","model_pre_params") == 'True':
			# 寻找最优超参
			best_est = self.BGI_train_cv(data['x'], data['y'], init_estimator=model, n_folds=n_folds, method='rdf', scoring='f1_weighted')
			best_params = best_est.get_params()
			self.logging_config("[Best Param For Pre CV] {}".format(json.dumps(best_params)),'info')
			# 获取最优超参
			model = RandomForestClassifier(**best_params)
		
		return model

	'''
	lightGBM
	'''
	def BGI_lgbClassifier(self,configFile,data):
		import numpy as np
		import pandas as pd
		from sklearn.model_selection import KFold
		from lightgbm import LGBMClassifier
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import cross_validate
		cf = configparser.ConfigParser()
		cf.read(configFile)
		
		# 初始化参数
		learning_rate = 		float(cf.get("model_param_init","lgb_learning_rate"))
		num_boost_round = 			int(cf.get("model_param_init","lgb_num_boost_round"))
		num_leaves = 			int(cf.get("model_param_init","lgb_num_leaves"))
		min_child_weight = 		float(cf.get("model_param_init","lgb_min_child_weight"))
		colsample_bytree = 				float(cf.get("model_param_init","lgb_colsample_bytree"))
		subsample = 			float(cf.get("model_param_init","lgb_subsample"))
		reg_alpha = 		float(cf.get("model_param_init","lgb_reg_alpha"))
		reg_lambda = 			float(cf.get("model_param_init","lgb_reg_lambda"))
		random_state =			 int(cf.get("model_param_init","lgb_random_state"))
		n_folds = 				int(cf.get("model_param_init","lgb_n_folds"))

		self.logging_config("[Param] learning_rate:{}".format(learning_rate),'info')
		self.logging_config("[Param] num_boost_round:{}".format(num_boost_round),'info')
		self.logging_config("[Param] num_leaves:{}".format(num_leaves),'info')
		self.logging_config("[Param] min_child_weight:{}".format(min_child_weight),'info')
		self.logging_config("[Param] subsample:{}".format(subsample),'info')
		self.logging_config("[Param] colsample_bytree:{}".format(colsample_bytree),'info')
		self.logging_config("[Param] reg_alpha:{}".format(reg_alpha),'info')
		self.logging_config("[Param] reg_lambda:{}".format(reg_lambda),'info')

		model = LGBMClassifier(learning_rate = learning_rate, # this is analogue to eta, Typical: 0.01-0.2
						 num_boost_round = num_boost_round, 
						 num_leaves = num_leaves, # Typical values: 3-10 
						 min_child_weight = min_child_weight,
						 subsample = subsample, # the fraction of observations to be randomly samples for each tree. 0.5-1
						 colsample_bytree = colsample_bytree, # the fraction of columns to be randomly samples for each tree. 0.5-1
						 reg_alpha = reg_alpha,
						 reg_lambda = reg_lambda,
						 verbose = -1,
						 seed = random_state
						 )
		# 判断是否需要做超参
		if cf.get("model_param_init","model_pre_params") == 'True':
			# 寻找最优超参
			best_est= self.BGI_train_cv(x_train=data['x'], y_train=data['y'], init_estimator=model, n_folds=n_folds, method='lgb', scoring='f1_weighted')
			best_params = best_est.get_params()
			best_params['num_boost_round'] = best_params['n_estimators']
			self.logging_config("[Best Param For Pre CV] {}".format(json.dumps(best_params)),'info')
			# 获取最优超参
			model = LGBMClassifier(**best_params)

		return model



	'''
	CatBoostClassifier
	'''
	def BGI_CatBoostClassifier(self,configFile,data):
		import numpy as np
		import pandas as pd
		from sklearn.model_selection import KFold
		from catboost import CatBoostClassifier
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import cross_validate
		cf = configparser.ConfigParser()
		cf.read(configFile)
		
		# 初始化参数
		learning_rate = 			float(cf.get("model_param_init","cbc_learning_rate"))
		max_depth = 			int(cf.get("model_param_init","cbc_max_depth"))
		reg_lambda = 		float(cf.get("model_param_init","cbc_reg_lambda"))
		loss_function = 				cf.get("model_param_init","cbc_loss_function")
		max_ctr_complexity = 			int(cf.get("model_param_init","cbc_max_ctr_complexity"))
		n_estimators = 		int(cf.get("model_param_init","cbc_n_estimators"))
		random_state =			 int(cf.get("model_param_init","cbc_random_state"))
		min_child_samples = 				int(cf.get("model_param_init","cbc_min_child_samples"))
		max_leaves = 				int(cf.get("model_param_init","cbc_max_leaves"))
		n_folds = 				int(cf.get("model_param_init","cbc_n_folds"))

		self.logging_config("[Param] learning_rate:{}".format(learning_rate),'info')
		self.logging_config("[Param] max_depth:{}".format(max_depth),'info')
		self.logging_config("[Param] reg_lambda:{}".format(reg_lambda),'info')
		self.logging_config("[Param] loss_function:{}".format(loss_function),'info')
		self.logging_config("[Param] max_ctr_complexity:{}".format(max_ctr_complexity),'info')
		self.logging_config("[Param] n_estimators:{}".format(n_estimators),'info')
		self.logging_config("[Param] random_state:{}".format(random_state),'info')
		self.logging_config("[Param] min_child_samples:{}".format(min_child_samples),'info')
		self.logging_config("[Param] max_leaves:{}".format(max_leaves),'info')
		self.logging_config("[Param] n_folds:{}".format(n_folds),'info')

		model = CatBoostClassifier(learning_rate = learning_rate,
						 max_depth = max_depth, 
						 loss_function = loss_function,
						 eval_metric = 'Logloss',
						 max_ctr_complexity = max_ctr_complexity,
						 n_estimators = n_estimators,
						 reg_lambda = reg_lambda,
						 min_child_samples = min_child_samples,
						 random_state = random_state,
						 max_leaves = max_leaves,
						 grow_policy = 'Lossguide',
						 verbose = 0,
						 )
		# 判断是否需要做超参
		if cf.get("model_param_init","model_pre_params") == 'True':
			# 寻找最优超参
			best_est= self.BGI_train_cv(x_train=data['x'], y_train=data['y'], init_estimator=model, n_folds=n_folds, method='cbc', scoring='f1_weighted')
			best_params = best_est.get_params()
			self.logging_config("[Best Param For Pre CV] {}".format(json.dumps(best_params)),'info')
			# 获取最优超参
			model = CatBoostClassifier(**best_params)

		return model

	'''
	SupportVectorClassifier
	'''
	def BGI_SVClassifier(self,configFile,data):
		import numpy as np
		import pandas as pd
		from sklearn.model_selection import KFold
		from sklearn.svm import SVC
		from sklearn.model_selection import GridSearchCV
		from sklearn.model_selection import cross_validate
		cf = configparser.ConfigParser()
		cf.read(configFile)
		
		# 初始化参数
		C = 			float(cf.get("model_param_init","svc_C"))
		kernel = 			cf.get("model_param_init","svc_kernel")
		gamma = 		float(cf.get("model_param_init","svc_gamma"))
		n_folds = 				int(cf.get("model_param_init","svc_n_folds"))
		probability = True
		class_weight =  None
		
		self.logging_config("[Param] C:{}".format(C),'info')
		self.logging_config("[Param] kernel:{}".format(kernel),'info')
		self.logging_config("[Param] gamma{}".format(gamma),'info')
		self.logging_config("[Param] n_folds{}".format(n_folds),'info')
		
		model = SVC(C = C,
						 gamma = gamma, 
						 kernel = kernel,
						 probability = probability,
						 class_weight= class_weight
						 )
		# 判断是否需要做超参
		if cf.get("model_param_init","model_pre_params") == 'True':
			# 寻找最优超参
			best_est= self.BGI_train_cv(x_train=data['x'], y_train=data['y'], init_estimator=model, n_folds=n_folds, method='svc', scoring='f1_weighted')
			best_params = best_est.get_params()
			self.logging_config("[Best Param For Pre CV] {}".format(json.dumps(best_params)),'info')
			# 获取最优超参
			model = SVC(**best_params)

		return model

	'''
	BGI_train_cv
	'''
	def BGI_train_cv(self, x_train, y_train, init_estimator, n_folds, method, scoring):	
		import warnings
		from tqdm import tqdm_notebook
		from sklearn.model_selection import GridSearchCV
		warnings.filterwarnings("ignore")

		# 模型参数空间
		param_space = {
			# xgb参数空间
			#----------------------------------------------------------------------------------
			'xgb': {
				'1':{ # 1. Tune max_depth and min_child_weight
					'max_depth': range(4,6),
					'min_child_weight':range(5,10)
					},
				'2':{ # 2. Tune gamma
					'gamma':[i/10.0 for i in range(4,8)]
					},
				'3':{ # 3. Tune subsample and colsample_bytree
					'subsample':[i/10.0 for i in range(5,8)],
					'colsample_bytree':[i/10.0 for i in range(5,8)]
					},
				'4':{ # 4. Tuning Regularization Parameters
					'reg_alpha':[0, 0.1, 0.01, 0.005, 0.001 ]
					},
				'5': { # 5. Tuning learning rate and the number of estimators
					'learning_rate':[ .1, .05],
					'n_estimators': [int(x) for x in range(50,150,50)],
					}
			},
			# lgb参数空间
			#----------------------------------------------------------------------------------
			'lgb': {
				'1':{ 
					'num_leaves':[int(x) for x in range(50,300,10)],
					'min_child_weight': [0.001,0.0001,0.01,0.1,0.004,0.0004,0.04,0.4]
					},
				'2':{
					'colsample_bytree':[i/10.0 for i in range(5,10)],
					'subsample':[i/10.0 for i in range(5,10)]
					},
				'3':{ 
					'reg_alpha':[i/100.0 for i in range(1,10)]
					},
				'4': { 
					'num_boost_round': [int(x) for x in range(50,500,50)],
					'learning_rate': [0.1,0.2,0.3,0.4,0.01,0.001,0.0001,0.04,0.004,0.0004],
					},
				'5':{
					'reg_lambda':[i/100.0 for i in range(1,10)]
					},
			},
			# rdf参数空间
			#----------------------------------------------------------------------------------
			'rdf': {
				'1':{ # 1. Tune n_estimator
					'n_estimators': [int(x) for x in range(50,200,20)],
					},
				'2':{ # 2. max_depth, min_impurity_split
					'max_depth':[int(x) for x in range(3,10,1)],
					'min_samples_split': [int(x) for x in range(3,60,2)]
					},
				'3':{ # 3. min_samples_split, min_samples_leaf
					'min_samples_leaf':[int(x) for x in range(3,20, 2)]
					},
				'4':{
					'max_leaf_nodes': [int(x) for x in range(11,60, 5)]
				}
			},
			# cbc参数空间
			#----------------------------------------------------------------------------------
			'cbc': {
				'1':{ # 1. Tune n_estimator
					'n_estimators': [int(x) for x in range(50,200,20)],
					},
				'2':{ # 2. max_depth, min_impurity_split
					'max_depth':[int(x) for x in range(3,10,1)],
					},
				'3':{ # 3. min_samples_split, min_samples_leaf
					'max_leaves': [int(x) for x in range(20,200, 30)],
					},
				'4':{
					'learning_rate': [i/100.0 for i in range(1,100,10)],
					},
				'5':{
					'max_ctr_complexity': [int(x) for x in range(3,8, 1)],
				},
				'6':{
					'reg_lambda':  [i/10.0 for i in range(1,100,10)],
				}
			},
			# svc参数空间
			#----------------------------------------------------------------------------------
			'svc': {
				'1':{ # 1. C
					'C': [i/100.0 for i in range(100,1000,100)],
					},
				'2':{ # 2. gamma
					'gamma':[i/100 for i in range(1,50,10)],
					}
				# 	},
				# '3':{ # 3. kernel
				# 	# 'kernel': ['rbf','poly','sigmoid','linear'],
				# 	'kernel': ['poly','sigmoid','linear'],
				# 	}
			}
		}

		best_est = init_estimator
		for k, params in tqdm_notebook(param_space[method].items()):
			self.logging_config("[param_groups] {}, [params] {}".format(k,params),'info')

			model_cv = GridSearchCV(estimator = best_est, param_grid = params, verbose = 0,
								  scoring=scoring,n_jobs=-1,cv=n_folds,return_train_score=True)
			model_cv.fit(x_train,y_train)
			
			best_est = model_cv.best_estimator_
			if method == 'xgb':
				best_est.get_booster().set_param(model_cv.best_params_)
			elif method == 'rdf' or method== 'lgb' or method =='svc':
				tmp_param = model_cv.best_params_
				best_est.set_params(**tmp_param)
			print("CV_model {}".format(model_cv.best_params_))
			print("best_est_model {}".format(best_est.get_params()))
		return best_est

	'''
	BGI_Downsample_way
	'''
	def BGI_Downsample_way(self,configFile,data,method):
		cf = configparser.ConfigParser()
		cf.read(configFile)
		random = int(cf.get("data","random"))
		trials = hyperopt.Trials()
		self.method = method
		self.data = data
		self.configFile = configFile
		self.random = random
		space = hyperopt.fmin(fn=self.hyperopt_objective,space=self.param_space_select(method=method),algo=hyperopt.tpe.suggest,max_evals=200,trials=trials,rstate=np.random.RandomState(random))
		if self.method=='lgb': 
			from lightgbm import LGBMClassifier
			space['num_leaves'] = int(round((space['num_leaves'])))
			model = LGBMClassifier(**space) 
			
		elif self.method=='xgb':
			from xgboost import XGBClassifier
			space['max_depth'] = int(round((space['max_depth'])))
			model = XGBClassifier(**space)
		
		elif self.method=='cbc':
			from catboost import CatBoostClassifier
			space['n_estimators'] = int(space['n_estimators'])
			space['max_leaves'] = int(space['max_leaves'])
			space['max_depth'] = int(space['max_depth'])
			space['max_ctr_complexity'] = int(space['max_ctr_complexity'])
			space['grow_policy'] = 'Lossguide'
			space['verbose'] = 0
			model = CatBoostClassifier(**space) 

		elif self.method=='rdf':
			from sklearn.ensemble import RandomForestClassifier
			space['max_depth'] = int(round((space['max_depth'])))
			space['n_estimators'] = int(round((space['n_estimators'])))
			space['min_samples_leaf'] = int(round((space['min_samples_leaf'])))
			space['min_samples_split'] = int(round((space['min_samples_split'])))
			model = RandomForestClassifier(**space)

		self.logging_config("[BGI_Downsample_way Method] {}".format(method),'info')
		self.logging_config("[BGI_Downsample_way Best Param] {}".format(json.dumps(space)),'info')
		print(type(space['n_estimators']))
		return model

	'''
	param_space_select
	'''
	def param_space_select(self,method):
		param_space = {
			'lgb':{
				'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
				'num_leaves': hp.quniform('num_leaves', 31, 255, 4),
				'min_child_weight': hp.uniform('min_child_weight', 0.0001, 0.1),
				'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
				'num_boost_round': hp.uniform('num_boost_round', 50, 500),
				'subsample': hp.uniform('subsample', 0.5, 1.),
				'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
				'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
			},
			'xgb':{
				'max_depth': hp.uniform('max_depth',3, 20),
				'min_child_weight': hp.uniform('min_child_weight',1.0, 5.0), 
				'gamma': hp.uniform('gamma',0.0001, 0.05), 
				'subsample': hp.uniform('subsample',0.6, 1.0), 
				'colsample_bytree': hp.uniform('colsample_bytree',0.6, 1.0),
				'reg_alpha': hp.uniform('reg_alpha',0.0001, 0.05),
				'learning_rate': hp.uniform('learning_rate',0.0001, 0.2),
			},
			'rdf':{
				'n_estimators': hp.uniform('n_estimators',40, 250),
				'min_samples_leaf' : hp.uniform('min_samples_leaf',2, 20),
				'min_samples_split':hp.uniform('min_samples_split',2, 20),
				'max_depth' : hp.uniform('max_depth',2,10), 
				'ccp_alpha': hp.uniform('ccp_alpha',0,1),
			},
			'cbc':{
				'n_estimators': hp.uniform('n_estimators',50, 200),
				'max_leaves' : hp.uniform('max_leaves',20, 200),
				'max_depth' : hp.uniform('max_depth',3,10), 
				'max_ctr_complexity': hp.uniform('max_ctr_complexity',3,8),
				'reg_lambda': hp.uniform('reg_lambda',0.1,10),
				'learning_rate': hp.uniform('learning_rate',0.01,1),
			}
		}
		return param_space[method]
	
	
	
	'''
	hyperopt_objective
	'''	
	def hyperopt_objective(self,space):
		
		model_run = '' 

		if self.method=='lgb': 
			from lightgbm import LGBMClassifier
			#space['num_boost_round'] = int(space['num_boost_round'])
			space['num_leaves'] = int(space['num_leaves'])
			space['num_boost_round'] = int(space['num_boost_round'])
			space['verbose'] = -1
			space['seed'] = self.random 
			model_run = LGBMClassifier(**space)

		if self.method=='cbc': 
			from catboost import CatBoostClassifier
			space['n_estimators'] = int(space['n_estimators'])
			space['max_leaves'] = int(space['max_leaves'])
			space['max_depth'] = int(space['max_depth'])
			space['max_ctr_complexity'] = int(space['max_ctr_complexity'])
			space['random_state'] = self.random
			space['grow_policy'] = 'Lossguide'
			space['verbose'] = 0
			model_run = CatBoostClassifier(**space) 
			
		elif self.method=='xgb':
			from xgboost import XGBClassifier
			space['max_depth'] = int(space['max_depth'])
			model_run = XGBClassifier(**space)
		
		elif self.method=='rdf':
			from sklearn.ensemble import RandomForestClassifier
			space['n_estimators'] = int(space['n_estimators'])
			space['min_samples_leaf'] = int(space['min_samples_leaf'])
			space['min_samples_split'] = int(space['min_samples_split'])
			space['max_depth'] = int(space['max_depth'])
			space['bootstrap'] = True
			space['oob_score'] = True
			model_run = RandomForestClassifier(**space)
		
		f1score = []
		from sklearn.model_selection import StratifiedKFold
		kf = StratifiedKFold(n_splits=5)
		# 5折CV
		for X_tr,X_val in kf.split(self.data['x'], self.data['y']):
			train_data = {}
			val_data = {}
			train_data['x'] = self.data['x'].iloc[X_tr, :]
			train_data['y'] = self.data['y'].iloc[X_tr]
			val_data['x'] = self.data['x'].iloc[X_val, :]
			val_data['y'] = self.data['y'].iloc[X_val]

			
			all_proba = []
			resample_method = ReSampleMethod()
			for i in range(0,100):
				downsample_data = resample_method.DownSample(configFile=self.configFile,data=train_data,random_state=i)
				model_run.fit(downsample_data['x'], downsample_data['y']) 
				all_proba.append(model_run.predict_proba(val_data['x']))
			
			pred_100 = (np.mean(all_proba, axis = 0) > 0.5).astype('int')[:,1]
			from sklearn.metrics import confusion_matrix
			tn, fp, fn, tp = confusion_matrix(val_data['y'], pred_100).ravel()
			from sklearn.metrics import f1_score
		
			f1score.append(f1_score(val_data['y'], pred_100))
		
		score = np.mean(f1score)
		self.logging_config("[Param For Pre CV] {}".format(json.dumps(space)),'info')
		self.logging_config("f1score: {}".format(score),'info')
		return -score
