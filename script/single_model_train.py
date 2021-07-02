import configparser
import logging
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import pickle5 as pickle
from feature_select_model import FeatureSelect
from best_param_cv_model import BestParamCV
from data_normalize import NormalizeMethod
from resample import ReSampleMethod
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import re
from sklearn.metrics import roc_curve, auc, confusion_matrix
import sys
import json

'''
设置日志格式
'''
def logging_config(msg,log_type):
	LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
	logging.basicConfig(filename='duty.log', level=logging.INFO, format=LOG_FORMAT)
	if(log_type == 'error'):
		logging.error(msg)
	elif(log_type == 'info'):
		logging.info(msg)

'''
保存变量
'''
def save_variable(v,filename):
	f=open(filename,'wb')
	pickle.dump(v,f)
	f.close()
	return filename

'''
读取变量
'''
def load_variable(filename):
	f=open(filename,'rb')
	r=pickle.load(f)
	f.close()
	return r

'''
读取表格训练数据
训练集和外部测试集
训练集将会拆分成测试机和内部训练集
'''
def load_table_data(configFile):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	data_dict = {}
	# 读取数据部分高度定制化，可以自己定制
	if cf.get("data","inside_data_x"):
		inside_data_x = pd.read_csv(cf.get("data","inside_data_x"), sep="\t", index_col=0)
		data_dict["inside_data_x"] = inside_data_x
		logging_config(u"reading inside_data_x...","info")
	
	if cf.get("data","inside_data_y"):
		inside_data_y = pd.read_csv(cf.get("data","inside_data_y"), sep="\t")['Type']
		data_dict["inside_data_y"] = inside_data_y
		logging_config(u"reading inside_data_y...","info")

	if cf.get("data","outside_data_x"):
		outside_data_x  = pd.read_csv(cf.get("data","outside_data_x"), sep="\t", index_col=0)
		data_dict["outside_data_x"] = outside_data_x
		logging_config(u"reading outside_data_x...","info")

	if cf.get("data","outside_data_y"):
		outside_data_y  = pd.read_csv(cf.get("data","outside_data_y"), sep="\t")['Type']
		data_dict["outside_data_y"] = outside_data_y
		logging_config(u"reading outside_data_y...","info")

	return data_dict


'''
convert to list
'''
def feature_select_convert_list(data):
	tmp = {}
	for i in data:
		tmp[i[0]] = i[1]
	return tmp

'''
特征工程
'''
def feature_select(configFile,data):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	fs = FeatureSelect()
	method = cf.get("feature_select","method")
	method = json.loads(method)
	feature_dict = {}

	for i in method:
		if i == 'SelectPercentile':
			logging_config("Feature Select Method: SelectPercentile",'info')
			model,feature_score = fs.SelectPercentile(configFile=configFile,data=data)
			plot_feature(pd.DataFrame(data['x']), [value for key,value in feature_score.items()], 20, 'SelectPercentile')
			feature_score = sorted(feature_score.items(),key = lambda kv:(kv[1],kv[0]),reverse=True)
			feature_dict['SelectPercentile'] = feature_select_convert_list(feature_score)
			logging_config("feature list length: {}".format(len(feature_dict['SelectPercentile'])),'info')

		elif i == 'LinearSVC':
			logging_config("Feature Select Method: LinearSVC",'info')
			model,feature_score = fs.LinearSVC(configFile=configFile,data=data)
			plot_feature(pd.DataFrame(data['x']), [value for key,value in feature_score.items()], 20, 'LinearSVC')
			feature_score = sorted(feature_score.items(),key = lambda kv:(kv[1],kv[0]),reverse=True)
			feature_dict['LinearSVC'] = feature_select_convert_list(feature_score)
			logging_config("feature list length: {}".format(len(feature_dict['LinearSVC'])),'info')
			
		elif i == 'ExtraTreesClassifier':
			logging_config("Feature Select Method: ExtraTreesClassifier",'info')
			model,feature_score = fs.ExtraTreesClassifier(configFile=configFile,data=data)
			plot_feature(pd.DataFrame(data['x']), [value for key,value in feature_score.items()], 20, 'ExtraTreesClassifier')
			feature_score = sorted(feature_score.items(),key = lambda kv:(kv[1],kv[0]),reverse=True)
			feature_dict['ExtraTreesClassifier'] = feature_select_convert_list(feature_score)
			logging_config("feature list length: {}".format(len(feature_dict['ExtraTreesClassifier'])),'info')
			
		elif i == 'mutual_info_classif':
			logging_config("Feature Select Method: mutual_info_classif",'info')
			model,feature_score = fs.mutual_info_classif(configFile=configFile,data=data)
			plot_feature(pd.DataFrame(data['x']), [value for key,value in feature_score.items()], 20, 'mutual_info_classif')
			feature_score = sorted(feature_score.items(),key = lambda kv:(kv[1],kv[0]),reverse=True)
			feature_dict['mutual_info_classif'] = feature_select_convert_list(feature_score)
			logging_config("feature list length: {}".format(len(feature_dict['mutual_info_classif'])),'info')
		
		elif i == 'boruta_selection':
			logging_config("Feature Select Method: boruta_selection",'info')
			model,feature_score = fs.boruta_selection(configFile=configFile,data=data)
			plot_feature(pd.DataFrame(data['x']), len(feature_score) - np.array([value for key,value in feature_score.items()]), 20, 'boruta_selection',ascending=False)
			feature_score = sorted(feature_score.items(),key = lambda kv:(kv[1],kv[0]))
			feature_dict['boruta_selection'] = feature_select_convert_list(feature_score)
			logging_config("feature list length: {}".format(len(feature_dict['boruta_selection'])),'info')

		elif i == 'RFE_selection':
			logging_config("Feature Select Method: RFE_selection",'info')
			model,feature_score = fs.RFE_selection(configFile=configFile,data=data)
			plot_feature(pd.DataFrame(data['x']), len(feature_score) - np.array([value for key,value in feature_score.items()]), 20, 'RFE_selection',ascending=False)
			feature_score = sorted(feature_score.items(),key = lambda kv:(kv[1],kv[0]))
			feature_dict['RFE_selection'] = feature_select_convert_list(feature_score)
			logging_config("feature list length: {}".format(len(feature_dict['RFE_selection'])),'info')
	
	if len(method) > 1:
		max_k_ratio = float(cf.get("feature_select","max_k_ratio"))
		min_hit = int(cf.get("feature_select","min_hit"))
		# feature select by voting
		sel_feat_dict, total_n_feats, out_feats_df = fs.feats_selection(feat_list=feature_dict, rank_method=['boruta_selection','RFE_selection'], max_k_ratio=max_k_ratio, min_hit=min_hit)
		logging_config("Feature Select By Voting: {} feature select".format(total_n_feats),'info')
		logging_config("Feature Select By Voting Save To feature_select.byVote.data",'info')
		data = {'sel_feat_dict':sel_feat_dict,'total_n_feats':total_n_feats,'out_feats_df':out_feats_df}
		save_variable(data,"feature_select.model.data")
	
	else:
		# merge时采用的数据
		logging_config("Feature Select Method feature_score Save To feature_select.feature_score.data",'info')
		save_variable(feature_score,"feature_select.feature_score.data")
		logging_config("Feature Select Method Save To feature_select.model.data",'info')
		save_variable(model,"feature_select.model.data")
	
	return 0

'''
特征重构
'''
def feature_build(configFile,model,data):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	method = cf.get("feature_select","method")
	method = json.loads(method)
	if len(method) == 1:
		# 内部训练
		data_dict = {'train':{},'test':{}}
		
		# RFE特征选择
		new_col_name = []
		for i in range(0,len(data['train']['x'].columns)):
			if model.support_[i]:
				new_col_name.append(data['train']['x'].columns[i])
		data_dict['train']['x'] = pd.DataFrame(model.transform(data['train']['x']),columns=new_col_name)
		data_dict['train']['y'] = data['train']['y']
		logging_config("feature build inside_data_train_x:{}".format(data_dict['train']['x'].shape),'info')
		logging_config("feature build inside_data_train_y:{}".format(data_dict['train']['y'].shape),'info')
		
		# 内部测试
		data_dict['test']['x'] = pd.DataFrame(model.transform(data['test']['x']),columns=new_col_name)
		data_dict['test']['y'] = data['test']['y']
		logging_config("feature build inside_data_test_x:{}".format(data_dict['test']['x'].shape),'info')
		logging_config("feature build inside_data_test_y:{}".format(data_dict['test']['y'].shape),'info')

		# 外部测试
		if 'test_out' in data:
			data_dict['test_out'] = {}
			data_dict['test_out']['x'] = pd.DataFrame(model.transform(data['test_out']['x']),columns=new_col_name)
			data_dict['test_out']['y'] = data['test_out']['y']
			logging_config("feature build inside_data_test_out_x:{}".format(data_dict['test_out']['x'].shape),'info')
			logging_config("feature build inside_data_test_out_y:{}".format(data_dict['test_out']['y'].shape),'info')
	# by vote	
	else:
		# 内部训练
		data_dict = {'train':{},'test':{}}
		index = []
		names_dict = dict(zip([int(i) for i in range(0,len(data['train']['x'].columns))], data['train']['x'].columns))
		data['train']['x'].columns = [int(x) for x in range(0, len(data['train']['x'].columns))]
		data_dict['train']['x'] = pd.DataFrame(pd.DataFrame(data['train']['x'])[model['sel_feat_dict']['num']])
		data_dict['train']['x'].columns = [names_dict[int(x)] for x in data_dict['train']['x'].columns]
		data_dict['train']['y'] = data['train']['y']
		logging_config("feature build inside_data_train_x:{}".format(data_dict['train']['x'].shape),'info')
		logging_config("feature build inside_data_train_y:{}".format(data_dict['train']['y'].shape),'info')
		
		# 内部测试
		data['test']['x'].columns = [int(x) for x in range(0, len(data['test']['x'].columns))] 
		data_dict['test']['x'] = pd.DataFrame(pd.DataFrame(data['test']['x'])[model['sel_feat_dict']['num']])
		data_dict['test']['x'].columns = data_dict['train']['x'].columns
		data_dict['test']['y'] = data['test']['y']
		logging_config("feature build inside_data_test_x:{}".format(data_dict['test']['x'].shape),'info')
		logging_config("feature build inside_data_test_y:{}".format(data_dict['test']['y'].shape),'info')

		# 外部测试
		if 'test_out' in data:
			data_dict['test_out'] = {}
			data['test_out']['x'].columns = [int(x) for x in range(0, len(data['test_out']['x'].columns))] 
			data_dict['test_out']['x'] = pd.DataFrame(pd.DataFrame(data['test_out']['x'])[model['sel_feat_dict']['num']])
			data_dict['test_out']['x'].columns = data_dict['train']['x'].columns
			data_dict['test_out']['y'] = data['test_out']['y']
			logging_config("feature build inside_data_test_out_x:{}".format(data_dict['test_out']['x'].shape),'info')
			logging_config("feature build inside_data_test_out_y:{}".format(data_dict['test_out']['y'].shape),'info')
		
	logging_config("New Feature Build Data Save To new_feature_build_data.data",'info')	
	save_variable(data_dict,"new_feature_build_data.data")


'''
数据标准化
'''
def data_normalize(configFile,data):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	nl = NormalizeMethod()
	if cf.get("standardization","isUse") == 'True':
		data_dict = {}
		if cf.get("standardization","method") == 'StandardScaler':
			data_dict = nl.StandardScalerMethod(configFile,data)
		elif cf.get("standardization","method") == 'LeaveOneOutEncoder':
			data_dict = nl.LeaveOneOutEncoderMethod(configFile,data)
		elif cf.get("standardization","method") == 'MinMaxScaler':
			data_dict = nl.MinMaxScalerMethod(configFile,data)
		elif cf.get("standardization","method") == 'logMinMaxScaler':
			data_dict = nl.logMinMaxScalerMethod(configFile,data)
		logging_config("Standardized Method {}".format(cf.get("standardization","method")),'info')
		logging_config("Standardized Data Save To data_normalize.data",'info')	
		save_variable(data_dict,"data_normalize.data")
	else:
		logging_config("Not Standardized Data Save To data_normalize.data",'info')
		save_variable(data,"data_normalize.data")


'''
数据拆分
'''
def data_split(configFile,data):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	test_rate = float(cf.get("data","test_rate"))
	logging_config("[Data Split] test_rate {}".format(test_rate),'info')
	X_train, X_test, y_train, y_test = train_test_split(data['inside_data_x'], data['inside_data_y'], test_size=test_rate, stratify=data['inside_data_y'])
	data_dict = {'test':{},'train':{}}
	data_dict['train']['x'] = X_train
	data_dict['train']['y'] = y_train
	data_dict['test']['x'] = X_test
	data_dict['test']['y'] = y_test
	if 'outside_data_x' in data:
		data_dict['test_out'] = {}
		data_dict['test_out']['x'] = data['outside_data_x']
		data_dict['test_out']['y'] = data['outside_data_y']
	logging_config("Inside Data Split Result Save To inside_data_split_data.data",'info')	
	save_variable(data_dict,"inside_data_split_data.data")

'''
采样方法
'''
def resample_data(configFile,data):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	if cf.get("model_param_init","model_downsample_way") == 'True':
		logging_config("[Resample] jump because model_downsample_way is True",'info')
		return data
	else:
		resampleMethod = ReSampleMethod()
		data_dict = resampleMethod.main(configFile=configFile,data=data)
		return data_dict

'''
CV最优参数
'''
def best_param_cv(configFile,data):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	bpm = BestParamCV()

	method_list = ['xgb','rdf','lgb','cbc','svc']
	method = ''
	# 判断启用哪个模型进行训练
	for item in method_list:
		if cf.get("model_param_init","{}_method".format(item)) == 'True':
			method = item
			break
	
	if cf.get("model_param_init","model_downsample_way") == 'True':
			model = bpm.BGI_Downsample_way(configFile=configFile, data=data['train'], method=method)
	else:
		if method == 'xgb':	
			logging_config("Best Param CV Model: XGBClassifier",'info')
			model = bpm.BGI_XGBClassifier(configFile=configFile, data=data['train'])
		elif method == 'lgb':
			logging_config("Best Param CV Model: lgbClassifier",'info')
			model = bpm.BGI_lgbClassifier(configFile=configFile, data=data['train'])
		elif method == 'rdf':
			logging_config("Best Param CV Model: RandomForestClassifierr",'info')
			model = bpm.BGI_RFClassifier(configFile=configFile, data=data['train'])
		elif method == 'cbc':
			logging_config("Best Param CV Model: CatBoostClassifier",'info')
			model = bpm.BGI_CatBoostClassifier(configFile=configFile, data=data['train'])
		elif method == 'svc':
			logging_config("Best Param CV Model: SupportVectorClassifier",'info')
			model = bpm.BGI_SVClassifier(configFile=configFile, data=data['train'])

	# 判断是否是超参训练还是手动调参
	if cf.get("model_param_init","model_pre_params") == 'True':
		logging_config("Best Param CV Train Model Save To best_param_model_data.data",'info')	
		save_variable(model,"best_param_model_data.data")
	
	else:
		logging_config("Manual Best Param Train Model Save To manual_best_param_model_data.data",'info')	
		save_variable(model,"best_param_model_data.data")

'''
Classifier_get_class_weight
'''
def Classifier_get_class_weight(y_train):
	from sklearn.utils.class_weight import compute_class_weight
	import numpy as np
	clsx = np.unique(y_train)
	cls_weight = compute_class_weight('balanced', clsx, y_train)
	class_weight_dict = dict(zip(clsx, cls_weight))
	return class_weight_dict

'''
绘制ROC曲线，寻找最优阈值
'''
def plot_ROC(y, y_pred, name):
	#计算fpr(假阳性率), tpr(真阳性率), thresholds(阈值)[绘制ROC曲线要用到这几个值]
	fpr,tpr,thresholds=roc_curve(y,y_pred)
	#interp:插值 把结果添加到tprs列表中
	roc_auc=auc(fpr,tpr)
	distance = np.array(fpr)*np.array(fpr) + (np.array(tpr)-1)*(np.array(tpr)-1)
	tmp = distance.tolist()
	best_cutoff_index = tmp.index(min(tmp))
	best_cutoff = thresholds[best_cutoff_index]
	#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
	a = '%.2f' % roc_auc
	b = '%.2f' % best_cutoff
	plt.rcParams["figure.figsize"] = 12,8
	plt.plot(fpr,tpr,lw=1,label='AUC={} best_cutoff={}'.format(a,b))
	plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
	plt.xlim([-0.05,1.05])
	plt.ylim([-0.05,1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc='lower right')
	plt.savefig('{}_ROC.pdf'.format(name), format='pdf')
	plt.close()
	return best_cutoff

'''
得分分布图
'''
def plot_density(y, y_pred, name):
	import seaborn as sns
	import matplotlib.pyplot as plt
	plt.rcParams["figure.figsize"] = 12,8
	sns.set(style="whitegrid")
	df = pd.DataFrame({'proba':y_pred,'class':y})
	sns.kdeplot(data=df,x='proba',hue='class',cut=0,fill=True,common_norm=False,alpha=0.4)
	plt.title("{} Density Plot".format(name))
	plt.savefig("{}.pdf".format(name), format='pdf')
	plt.close()

'''
特征权重图
'''
def plot_feature(trn_data,import_fea,max_num,filename,ascending = True):
	import pandas as pd
	import numpy as np
	from matplotlib import pyplot as plt
	columns = pd.DataFrame(trn_data.columns)
	columns.columns = ['indicators']
	columns.insert(1,'sort val',import_fea)
	if ascending:
		columns = columns.sort_values(by=['sort val'],ascending = False)
	else:
		columns = columns.sort_values(by=['sort val'],ascending = True)
	columns.insert(2,'sum val',np.cumsum(columns['sort val']))
	plt.rcParams["figure.figsize"] = 12,8
	plt.plot(list(columns['sum val']),marker='o',color='#01939A')
	plt.ylabel("Accumulated value")
	plt.xlabel('features counts')
	plt.title('Cumulative curve of each indicator')
	plt.show()
	plt.rcParams['figure.figsize'] = (12.0, 8.0) # 单位是inches
	plt.savefig("{}.feature_add.pdf".format(filename), format='pdf')
	plt.close()

	#数据
	new_importance=columns.iloc[0:max_num,:]
	new_importance = new_importance.sort_values(by=['sort val'],ascending = True)
	name=new_importance['indicators']
	colleges=new_importance['sort val']

	#图像绘制
	fig,ax=plt.subplots()
	b=ax.barh(range(len(name)),colleges,color='#01939A')

	#添加数据标签
	for rect in b:
		w=rect.get_width()
		ax.text(w,rect.get_y()+rect.get_height()/2,'%0.4f'%(w),ha='left',va='center')
	#设置Y轴刻度线标签
	ax.set_yticks(range(len(name)))
	ax.set_yticklabels(name)
	plt.ylabel("Feature index")
	plt.title('Feature top {} most importances'.format(max_num))
	plt.savefig("{}.feature_sort.pdf".format(filename), format='pdf')
	plt.close()

	####画图之后，根据曲线图获取最后筛选的特征
	final_features = columns.iloc[0:20,0]  ###临界值，可根据自己的曲线图修改
	return final_features,columns


'''
模型训练 + 验证
'''
def model_train_test(configFile,model,data):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	if cf.get("model_param_init","model_downsample_way") == 'True':
		resample_method = ReSampleMethod()
		all_proba = []
		feature_importances_list = []
		model_list = []
		for i in range(0,100):
			downsample_data = resample_method.DownSample(configFile=configFile,data=data['train'],random_state=i)
			weight_class_dict = Classifier_get_class_weight(downsample_data['y'])
			ins_weight_lst = [weight_class_dict[i] for i in downsample_data['y']]
			if cf.get("model_param_init","xgb_method") == 'True':
				model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst)
			elif cf.get("model_param_init","rdf_method") == 'True':
				model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst) 
			elif  cf.get("model_param_init","lgb_method") == 'True':
				model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst)
			elif  cf.get("model_param_init","cbc_method") == 'True':
					model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst)
			elif  cf.get("model_param_init","svc_method") == 'True':
					model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst)  
			model_list.append(model)
			all_proba.append(model.predict_proba(data['train']['x']))
		yt_pred = np.mean(all_proba, axis = 0)
		logging_config('model_downsample_way_model Save to model_downsample_way_model.data','info')
		save_variable(model_list,"model_downsample_way_model.data")
	else:
		weight_class_dict = Classifier_get_class_weight(data['train']['y'])
		ins_weight_lst = [weight_class_dict[i] for i in data['train']['y']]
		if cf.get("model_param_init","xgb_method") == 'True':
			model.fit(data['train']['x'],data['train']['y'], sample_weight=ins_weight_lst)
		elif cf.get("model_param_init","rdf_method") == 'True':
			model.fit(data['train']['x'],data['train']['y'], sample_weight=ins_weight_lst)
		elif  cf.get("model_param_init","lgb_method") == 'True':
			model.fit(data['train']['x'],data['train']['y'], sample_weight=ins_weight_lst)
		elif  cf.get("model_param_init","cbc_method") == 'True':
			model.fit(data['train']['x'],data['train']['y'], sample_weight=ins_weight_lst)
		elif  cf.get("model_param_init","svc_method") == 'True':
			model.fit(data['train']['x'],data['train']['y'], sample_weight=ins_weight_lst)
		# 模型特征查看
		# plot_feature(pd.DataFrame(data['train']['x']), model.feature_importances_, 20, 'Train')
		logging_config('Final model Save to Final.model.data','info')
		logging_config("Final model Feature Num: {}".format(str(len(data['train']['x'].columns))),'info')
		save_variable(model,"Final.model.data")
		yt_pred = model.predict_proba(data['train']['x'])
	
	from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
	from sklearn.metrics import roc_curve, auc, confusion_matrix
	yt_pred = yt_pred[:,1]
	# ROC曲线
	model_result = {}
	model_result['train'] = [list(data['train']['y']),list(yt_pred)]
	best_cutoff = plot_ROC(list(data['train']['y']), list(yt_pred), 'Train')
	# 是否按默认default_cut_off区分
	if cf.get("data","default_cut_off") == 'True':
		best_cutoff = 0.5
	# 得分分布图
	plot_density(list(data['train']['y']), list(yt_pred), "Train_Density")
	# 混淆矩阵
	yt_pred = list(yt_pred)
	for i in range(0,len(yt_pred)):
		if yt_pred[i] >= best_cutoff:
			yt_pred[i] = 1
		else:
			yt_pred[i] = 0
	class_names = ['0', '1']
	cnf_matrix = confusion_matrix(list(data['train']['y']), yt_pred,labels=list(range(len(class_names))))
	plot_confusion_matrix(cnf_matrix,
						classes=class_names,
						output_name="Inside_Train_confusion_matrix",
						normalize=False, 
						title='Unnormalized confusion matrix - train')
	#计算准确率
	logging_config('f1 score:%2.f%%'%(f1_score(data['train']['y'],yt_pred)*100),'info')
	logging_config('Train accuracy:%2.f%%'%(accuracy_score(data['train']['y'],yt_pred)*100),'info')
	logging_config('Train recall:%2.f%%'%(recall_score(data['train']['y'],yt_pred)*100),'info')
	logging_config('Train precision_score:%2.f%%'%(precision_score(data['train']['y'],yt_pred)*100),'info')
	logging_config ('**************************','info')
	

	if cf.get("model_param_init","model_downsample_way") == 'True':
		all_proba = []
		for model in model_list:
			all_proba.append(model.predict_proba(data['test']['x']))
		y_pred = np.mean(all_proba, axis = 0)
	else:
		y_pred = model.predict_proba(data['test']['x'])
	
	y_pred = y_pred[:,1]
	y_pred = list(y_pred)
	# ROC曲线
	model_result['test'] = [list(data['test']['y']),list(y_pred)]
	plot_ROC(list(data['test']['y']), list(y_pred), 'Test')
	# 得分分布图
	plot_density(list(data['test']['y']), y_pred, "Test_Density")

	for i in range(0,len(y_pred)):
		if y_pred[i] >= best_cutoff:
			y_pred[i] = 1
		else:
			y_pred[i] = 0
	
	#混淆矩阵
	cnf_matrix = confusion_matrix(list(data['test']['y']), y_pred, labels=list(range(len(class_names))))
	plot_confusion_matrix(cnf_matrix, 
						classes=class_names,
						output_name="Inside_Test_confusion_matrix",
						normalize=False, 
						title='Unnormalized confusion matrix - test')
	#计算准确率
	logging_config('f1 score:%2.f%%'%(f1_score(data['test']['y'],y_pred)*100),'info')
	logging_config('Test accuracy:%2.f%%'%(accuracy_score(data['test']['y'],y_pred)*100),'info')
	logging_config('Test recall:%2.f%%'%(recall_score(data['test']['y'],y_pred)*100),'info')
	logging_config('Test precision_score:%2.f%%'%(precision_score(data['test']['y'],y_pred)*100),'info')
	
	
	if 'test_out' in data:
		if cf.get("model_param_init","model_downsample_way") == 'True':
			all_proba = []
			for model in model_list:
				all_proba.append(model.predict_proba(data['test_out']['x']))
			y_outside_pred = np.mean(all_proba, axis = 0)
		else:
			y_outside_pred = model.predict_proba(data['test_out']['x'])
		
		y_outside_pred = y_outside_pred[:,1]
		y_outside_pred = list(y_outside_pred)

		# ROC曲线
		model_result['test_out'] = [list(data['test_out']['y']),list(y_outside_pred)]
		plot_ROC(list(data['test_out']['y']), list(y_outside_pred), 'Outside_Test')
		
		# 得分分布图
		plot_density(list(data['test_out']['y']), y_outside_pred, "Outside_Test_Density")

		for i in range(0,len(y_outside_pred)):
			if y_outside_pred[i] >= best_cutoff:
				y_outside_pred[i] = 1
			else:
				y_outside_pred[i] = 0
		
		#混淆矩阵
		cnf_matrix = confusion_matrix(list(data['test_out']['y']), y_outside_pred,labels=list(range(len(class_names))))
		plot_confusion_matrix(cnf_matrix, 
						classes=class_names,
						output_name="Outside_Test_confusion_matrix",
						normalize=False, 
						title='Unnormalized confusion matrix - outside_test')
		#计算准确率
		logging_config('f1 score:%2.f%%'%(f1_score(data['test_out']['y'],y_outside_pred)*100),'info')
		logging_config('Outside Test accuracy:%2.f%%'%(accuracy_score(data['test_out']['y'],y_outside_pred)*100),'info')
		logging_config('Outside Test recall:%2.f%%'%(recall_score(data['test_out']['y'],y_outside_pred)*100),'info')
		logging_config('Outside Test precision_score:%2.f%%'%(precision_score(data['test_out']['y'],y_outside_pred)*100),'info')
	
	#保存预测结果
	save_variable(model_result,"model_result.data")

'''
交叉验证
'''
def model_cross_valid(configFile, model, data):
	cf = configparser.ConfigParser()
	cf.read(configFile)
	n_folds = int(cf.get("data","corss_valid"))
	KF = StratifiedKFold(n_splits = n_folds, shuffle = True)
	train_data = pd.DataFrame(data['train']['x'])
	train_label = data['train']['y']
	tprs=[]
	aucs=[]
	mean_fpr=np.linspace(0,1,100)
	flod_times=0
	cv_result = {}
	cv_result['predict'] = {}
	flod_time = 0
	for train_index,test_index in KF.split(train_data,train_label):
		X_train,X_test = train_data.iloc[train_index], train_data.iloc[test_index]
		Y_train,Y_test = train_label.iloc[train_index], train_label.iloc[test_index]
		
		flod_time += 1
		if cf.get("model_param_init","model_downsample_way") == 'True':
			resample_method = ReSampleMethod()
			all_proba = []
			for i in range(0,100):
				tmp_data = {"x":X_train,"y":Y_train}
				downsample_data = resample_method.DownSample(configFile=configFile,data=tmp_data,random_state=i)
				weight_class_dict = Classifier_get_class_weight(downsample_data['y'])
				ins_weight_lst = [weight_class_dict[i] for i in downsample_data['y']]
				if cf.get("model_param_init","xgb_method") == 'True':
					model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst)
				elif cf.get("model_param_init","rdf_method") == 'True':
					model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst) 
				elif  cf.get("model_param_init","lgb_method") == 'True':
					model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst)
				elif  cf.get("model_param_init","cbc_method") == 'True':
					model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst)
				elif  cf.get("model_param_init","svc_method") == 'True':
					model.fit(downsample_data['x'], downsample_data['y'], sample_weight=ins_weight_lst)
				all_proba.append(model.predict_proba(X_test))
				y_pred = np.mean(all_proba, axis = 0)
		else:
			weight_class_dict = Classifier_get_class_weight(Y_train)
			ins_weight_lst = [weight_class_dict[i] for i in Y_train]
			if cf.get("model_param_init","xgb_method") == 'True':
				model.fit(X_train, Y_train, sample_weight=ins_weight_lst)
			elif cf.get("model_param_init","rdf_method") == 'True':
				model.fit(X_train, Y_train, sample_weight=ins_weight_lst) 
			elif  cf.get("model_param_init","lgb_method") == 'True':
				model.fit(X_train, Y_train, sample_weight=ins_weight_lst)
			elif  cf.get("model_param_init","cbc_method") == 'True':
				model.fit(X_train, Y_train, sample_weight=ins_weight_lst)
			elif  cf.get("model_param_init","svc_method") == 'True':
				model.fit(X_train, Y_train, sample_weight=ins_weight_lst)
			y_pred = model.predict_proba(X_test)
		y_pred = y_pred[:,1]
		#计算fpr(假阳性率), tpr(真阳性率), thresholds(阈值)[绘制ROC曲线要用到这几个值]
		fpr,tpr,thresholds=roc_curve(Y_test,y_pred)
		cv_result['predict'][flod_time] = [list(Y_test),list(y_pred)]
		#interp:插值 把结果添加到tprs列表中
		tprs.append(interp(mean_fpr,fpr,tpr))
		tprs[-1][0]=0.0
		roc_auc=auc(fpr,tpr)
		aucs.append(roc_auc)
		#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
		plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (flod_times,roc_auc))
		flod_times +=1
	plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
	mean_tpr=np.mean(tprs,axis=0)
	mean_tpr[-1]=1.0
	mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
	std_auc=np.std(aucs)
	plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2,alpha=.8)
	cv_result['cv'] = [mean_fpr,mean_tpr,mean_auc,std_auc]
	#保存预测结果
	save_variable(cv_result,"cv_result.data")

	std_tpr=np.std(tprs,axis=0)
	tprs_upper=np.minimum(mean_tpr+std_tpr,1)
	tprs_lower=np.maximum(mean_tpr-std_tpr,0)
	plt.fill_between(mean_fpr,tprs_lower,tprs_upper,color='gray',alpha=.2, label=r'$\pm$ 1 std. dev.')
	plt.xlim([-0.05,1.05])
	plt.ylim([-0.05,1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc='lower right')
	plt.savefig('CV_ROC.pdf', format='pdf')
	plt.close()


'''
混淆矩阵
'''
def plot_confusion_matrix(cm, classes, output_name, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		logging_config(u"Normalized confusion matrix...","info")
	else:
		logging_config(u"Confusion matrix, without normalization...","info")
	
	import matplotlib.pyplot as plt
	import numpy as np
	import  matplotlib
	import seaborn as sns
	rc={'font.size': 12, 'axes.labelsize': 12, 'figure.dpi':300,
	'axes.titlesize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
	'lines.linewidth':1,
	'font.family': 'Arial',
	'font.size':12
	}

	plt.rcParams["figure.figsize"] = 12,8
	sns.set(style='ticks',rc=rc)
	plt.imshow(cm, cmap=plt.cm.Blues)
	
	indices = range(len(cm))
	plt.xticks(indices, ['Control','Case'])
	plt.yticks(indices, ['Control','Case'])

	plt.title(title)
	plt.colorbar()
	
	tick_marks = np.arange(len(classes))

	fmt = '.2f' if normalize else 'd'
	print(cm.max())
	thresh = cm.max() / 2.
	print(thresh)
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig("{}.pdf".format(output_name), format='pdf')
	plt.close()


if __name__ == "__main__":
	#  读取数据
	configFile = sys.argv[1]
	data_dict = load_table_data(configFile)
	
	#数据拆分 inside_data_split_data.data
	data_split(configFile=configFile,data=data_dict)
	data_dict = {}
	
	#  标准化 data_normalize.data
	data_dict = load_variable("inside_data_split_data.data")
	data_normalize(configFile=configFile,data=data_dict)
	data_dict={}

	# #  特征工程 feature_select.model.data
	data_dict = load_variable("data_normalize.data")
	cf = configparser.ConfigParser()
	cf.read(configFile)
	method = cf.get("feature_select","method")
	method = json.loads(method)
	if len(method) >= 1:
		feature_select(configFile=configFile,data=data_dict['train'])
		# # # # 特征构建 new_feature_build_data.data
		feature_select_model = load_variable("feature_select.model.data")
		feature_build(configFile=configFile,model=feature_select_model,data=data_dict)

	# # # # 内部训练集重采样 可跳过
	if len(method) >= 1:
		data_dict = load_variable("new_feature_build_data.data")
	data_dict['test']['x'] = data_dict['test']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
	data_dict['train']['x'] = data_dict['train']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
	data_dict['test_out']['x'] = data_dict['test_out']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
	data_dict['train'] = resample_data(configFile=configFile,data=data_dict['train'])

	
	# # # 最优化模型参数 best_param_model_data.data
	best_param_cv(configFile=configFile, data=data_dict)
	data_dict = {}
	
	# 交叉验证
	model = load_variable("best_param_model_data.data")

	if len(method) >= 1:
		data_dict = load_variable("new_feature_build_data.data")
	else:
		data_dict = load_variable("data_normalize.data")
	data_dict['test']['x'] = data_dict['test']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
	data_dict['train']['x'] = data_dict['train']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
	data_dict['test_out']['x'] = data_dict['test_out']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
	model_cross_valid(configFile=configFile,model=model,data=data_dict)

	# # 模型训练+测试结果
	model_train_test(configFile=configFile,model=model,data=data_dict)


