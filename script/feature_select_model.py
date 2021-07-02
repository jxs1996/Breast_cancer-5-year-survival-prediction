import configparser
import logging
import pandas as pd
import numpy as np

class FeatureSelect(object):
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

	'''
	Classifier_get_class_weight
	'''
	def Classifier_get_class_weight(self,y_train):
		from sklearn.utils.class_weight import compute_class_weight
		import numpy as np
		clsx = np.unique(y_train)
		cls_weight = compute_class_weight('balanced', clsx, y_train)
		class_weight_dict = dict(zip(clsx, cls_weight))
		return class_weight_dict


	'''
	# 单变量特征选择
	# 单变量的特征选择是通过基于单变量的统计测试来选择最好的特征。它可以当做是评估器的预处理步骤。
	# Scikit-learn 将特征选择的内容作为实现了 transform 方法的对象：
	# SelectKBest 移除那些除了评分最高的 K 个特征之外的所有特征
	# SelectPercentile 移除除了用户指定的最高得分百分比之外的所有特征
	# 对每个特征应用常见的单变量统计测试: 
	# 假阳性率（false positive rate） SelectFpr,
	# 伪发现率（false discovery rate） SelectFdr , 
	# 族系误差（family wise error） SelectFwe
	# GenericUnivariateSelect 允许使用可配置方法来进行单变量特征选择。它允许超参数搜索评估器来选择最好的单变量特征。
	# 例如下面的实例，我们可以使用卡方检验样本集来选择最好的30%个特征：
	[feature_select]
	method = SelectPercentile
	score_func = YOUR_DATA
	percentile = YOUR_DATA
	'''
	def SelectPercentile(self,configFile,data):
		# 获取参数
		percentile = 10
		score_func = 'f_classif'

		import sklearn.feature_selection as sfs
		score_func = "sfs.{}".format(score_func)
		model = sfs.SelectPercentile(eval(score_func), percentile = percentile)

		data['y'].index = [int(x) for x in range(0,len(data['y']))]
		data['x'].index = [int(x) for x in range(0,len(data['x']))]
		model.fit(data['x'],data['y'])
		feature_weight = dict(zip([int(i) for i in range(0,len(model.scores_))],model.scores_))
		
		self.logging_config("[Param] score_func:{}".format(score_func),'info')
		self.logging_config("[Param] percentile:{}".format(str(percentile)),'info')
		
		return (model,feature_weight)


	'''
	# 基于 L1 的特征选取
	# Linear models 使用 L1 正则化的线性模型会得到稀疏解：
	# 他们的许多系数为 0。 当目标是降低使用另一个分类器的数据集的维度， 
	# 它们可以与 feature_selection.SelectFromModel 一起使用来选择非零系数。
	# 特别的，可以用于此目的的稀疏评估器有用于回归的 linear_model.Lasso , 
	# 以及用于分类的 linear_model.LogisticRegression 和 svm.LinearSVC
	[feature_select]
	method = LinearSVC
	C = YOUR_DATA
	'''
	def LinearSVC(self,configFile,data):
		# 获取参数
		C = 1.0
		penalty = 'l1'

		from sklearn.svm import LinearSVC
		from sklearn.feature_selection import SelectFromModel

		data['y'].index = [int(x) for x in range(0,len(data['y']))]
		data['x'].index = [int(x) for x in range(0,len(data['x']))]
		lsvc = LinearSVC(C=C, penalty=penalty, dual=False, random_state=1996).fit(data['x'],data['y'])
		model = SelectFromModel(lsvc, prefit=True)
		feature_weight = dict(zip([int(i) for i in range(0,len(lsvc.coef_[0]))],np.maximum(lsvc.coef_[0], -lsvc.coef_[0])))

		self.logging_config("[Param] C:{}".format(str(C)),'info')
		self.logging_config("[Param] penalty:{}".format(str(penalty)),'info')
		return (model,feature_weight)


	'''
	# 基于 Tree（树）的特征选取
	# 基于树的 estimators （查阅 sklearn.tree 模块和树的森林 在 sklearn.ensemble 模块） 可以用来计算特征的重要性，
	# 然后可以消除不相关的特征（当与 sklearn.feature_selection.SelectFromModel 等元转换器一同使用时）:
	[feature_select]
	method = ExtraTreesClassifier
	'''
	def ExtraTreesClassifier(self,configFile,data):
		from sklearn.ensemble import ExtraTreesClassifier
		from sklearn.feature_selection import SelectFromModel
		data['y'].index = [int(x) for x in range(0,len(data['y']))]
		data['x'].index = [int(x) for x in range(0,len(data['x']))]
		tree = ExtraTreesClassifier(random_state=1996).fit(data['x'],data['y'])
		model = SelectFromModel(tree, prefit=True)
		feature_weight = dict(zip([int(i) for i in range(0,len(tree.feature_importances_))],tree.feature_importances_))
		return (model,feature_weight)

	'''
	互信息
	'''
	def mutual_info_classif(self,configFile,data):
		from sklearn.feature_selection import mutual_info_classif
		from functools import partial
		from sklearn.feature_selection import SelectKBest
		
		mi_clf = partial(mutual_info_classif,discrete_features=False,random_state=123)
		data['y'].index = [int(x) for x in range(0,len(data['y']))]
		data['x'].index = [int(x) for x in range(0,len(data['x']))]
		model = SelectKBest(mi_clf,k='all').fit(data['x'], data['y'])
		feature_weight = dict(zip([int(i) for i in range(0,len(model.scores_))],model.scores_))

		return (model, feature_weight)
	
	'''
	Boruta selection
	'''
	def boruta_selection(self,configFile,data):
		from boruta import BorutaPy
		from sklearn.ensemble import RandomForestClassifier
		data['y'].index = [int(x) for x in range(0,len(data['y']))]
		data['x'].index = [int(x) for x in range(0,len(data['x']))]
		rf = RandomForestClassifier(n_jobs=-1, 
									class_weight=self.Classifier_get_class_weight(data['y']), 
									max_depth=3,
									n_estimators=200,
									random_state=1996)
		model = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1).fit(data['x'].values, data['y'].values)
		feature_weight = dict(zip([int(i) for i in range(0,len(model.ranking_))],model.ranking_))
		return (model, feature_weight)

	'''
	RFE
	'''
	def RFE_selection(self,configFile,data):
		from sklearn.feature_selection import RFE
		from sklearn.ensemble import ExtraTreesClassifier
		data['y'].index = [int(x) for x in range(0,len(data['y']))]
		data['x'].index = [int(x) for x in range(0,len(data['x']))]
		et = ExtraTreesClassifier(n_jobs=-1, 
								class_weight=self.Classifier_get_class_weight(data['y']), 
								max_depth=3,
								n_estimators=200,
								random_state=123,bootstrap=True)
		model = RFE(et).fit(data['x'], data['y'])
		feature_weight = dict(zip([int(i) for i in range(0,len(model.ranking_))],model.ranking_))
		return (model, feature_weight)

	'''
	特征整合分析
	feat_list: data list
	max_k_ratio: max number of features will be selected
	min_hit: Minimum number of votes 
	rank_method: rank method
	'''
	def feats_selection(self, feat_list, rank_method, max_k_ratio=0.3, min_hit=2):
		import math
		feat_sel_methods= feat_list.keys()
		sel_feat_dict = dict()
		total_n_feats = 0
		
		tmp_feat_ls=[]
		tmp_feat_set = set()
		feat_sel = dict()
		for method in feat_sel_methods:
			sel_name = method 
			i_feat_d = feat_list[sel_name]
			
			top_k = math.ceil(len(i_feat_d) * max_k_ratio)
			sel_vals = np.array(list(i_feat_d.values()))
			if len(sel_vals[sel_vals==1]) > top_k and method in rank_method:
				top_k = len(sel_vals[sel_vals==1])
		
			sel_feats = list(i_feat_d.keys())[:top_k]
			sel_vals_lst = sel_vals.tolist()[:top_k]
				
			feat_sel[sel_name]=(sel_feats, sel_vals_lst)
			tmp_feat_ls.extend(sel_feats)

		tmp_feat_set |= set(tmp_feat_ls)
		tmp_feat_ls = list(tmp_feat_set)

		out_feats = dict()
		out_feats['feats'] = tmp_feat_ls
		for method in feat_sel_methods: 
			sel_name = method
			i_feat_d = feat_sel[sel_name]
			
			t_feats_name = i_feat_d[0]
			t_feats_val = i_feat_d[1]

			if method in rank_method:
				out_feats['{}_sel'.format(method)]=[ 1 if (i in t_feats_name and t_feats_val[t_feats_name.index(i)] == 1) 
													else 0 for i in tmp_feat_ls ]
			else:
				out_feats['{}_sel'.format(method)]=[ 1 if i in t_feats_name else 0 for i in tmp_feat_ls ]

			out_feats['{}_score'.format(method)] = [ t_feats_val[t_feats_name.index(t_f)] if t_f in t_feats_name else -1
											for t_f in tmp_feat_ls]
														
		out_feats_df = pd.DataFrame(out_feats) 
		out_feats_df['n_sel'] = out_feats_df[['{}_sel'.format(md) for md in feat_sel_methods]].sum(axis=1)
		out_feats_df.sort_values(by='n_sel', inplace=True, ascending=False)
		
		final_sel_feats = out_feats_df.loc[out_feats_df['n_sel']>=min_hit,'feats'].values.tolist()
		sel_feat_dict['num']= final_sel_feats
		total_n_feats += len(final_sel_feats)
		
		return sel_feat_dict, total_n_feats, out_feats_df


