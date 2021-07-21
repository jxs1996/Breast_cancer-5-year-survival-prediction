from single_model_train import *


'''
merge数据标准化
'''
def merge_data_normalize(configFile,data,fs_class_dict):
    cf = configparser.ConfigParser()
    cf.read(configFile)
    fs_label = ["SNV","CNA","CNGM","pheno"]
    # StandardScaler / LeaveOneOutEncoder / MinMaxScaler
    fs_standardization = ["","StandardScaler","logMinMaxScaler",""]

    fs_data_train = []
    fs_data_test = []
    if cf.get("data","outside_data_y"):
        fs_data_test_out = []
   
    for j in range(0,len(fs_label)):
        # 获取特征切片数据
        data_tmp_dict = {'train':{},'test':{}}
        data_tmp_dict['train']['x'] = data['train']['x'][fs_class_dict[fs_label[j]]]
        data_tmp_dict['train']['y'] = data['train']['y']
        data_tmp_dict['test']['x'] = data['test']['x'][fs_class_dict[fs_label[j]]]
        data_tmp_dict['test']['y'] = data['test']['y']
        if cf.get("data","outside_data_y"):
            data_tmp_dict['test_out'] = {}
            data_tmp_dict['test_out']['x'] = data['test_out']['x'][fs_class_dict[fs_label[j]]]
            data_tmp_dict['test_out']['y'] = data['test_out']['y']
        print(data_tmp_dict['train']['x'])
        # 特征切片数据标准化
        nl = NormalizeMethod()
        if fs_standardization[j] == 'StandardScaler':
            data_tmp_dict = nl.StandardScalerMethod(configFile,data_tmp_dict)
        elif fs_standardization[j] == 'LeaveOneOutEncoder':
            data_tmp_dict = nl.LeaveOneOutEncoderMethod(configFile,data_tmp_dict)
        elif fs_standardization[j] == 'MinMaxScaler':
            data_tmp_dict = nl.MinMaxScalerMethod(configFile,data_tmp_dict)
        elif fs_standardization[j] == 'logMinMaxScaler':
            data_tmp_dict = nl.logMinMaxScalerMethod(configFile,data_tmp_dict)
        elif fs_standardization[j] == '':
            pass
        print(data_tmp_dict['train']['x'])

        # 标准化数据装箱
        logging_config("Feature {} Use Standardized Method {}".format(fs_label[j],fs_standardization[j]),'info')
        fs_data_train.append(data_tmp_dict['train']['x'])
        fs_data_test.append(data_tmp_dict['test']['x'])
        if cf.get("data","outside_data_y"):
            fs_data_test_out.append(data_tmp_dict['test_out']['x'])
            
    data_dict = {'train':{},'test':{}}
    import re
    data_dict['train']['x'] = pd.DataFrame(pd.concat(fs_data_train,axis=1))
    data_dict['train']['x'] = data_dict['train']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
    data_dict['train']['y'] = data['train']['y']
    data_dict['test']['x'] = pd.DataFrame(pd.concat(fs_data_test,axis=1))
    data_dict['test']['x'] = data_dict['test']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
    data_dict['test']['y'] = data['test']['y']
    if cf.get("data","outside_data_y"):
        data_dict['test_out'] = {}
        data_dict['test_out']['x'] = pd.DataFrame(pd.concat(fs_data_test_out,axis=1))
        data_dict['test_out']['x'] = data_dict['test_out']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
        data_dict['test_out']['y'] = data['test_out']['y']

    save_variable(data_dict,"data_normalize.data")

if __name__ == "__main__":
    data_dict={}
    data_dict["inside_data_y"] = pd.read_csv("../../data/5_year_death.dataset1.txt", sep="\t")
    data_dict["inside_data_x_cna"] = pd.read_csv("../../data/CNA.5_year_death.dataset1.txt", sep="\t", index_col=0)
    data_dict["inside_data_x_snp"] = pd.read_csv("../../data/SNV.5_year_death.dataset1.txt", sep="\t", index_col=0)
    data_dict["inside_data_x_mut"] = pd.read_csv("../../data/CNGM.5_year_death.dataset1.txt", sep="\t", index_col=0)
    data_dict["inside_data_x"] = pd.DataFrame(pd.concat([data_dict["inside_data_x_cna"],data_dict["inside_data_x_snp"],data_dict["inside_data_x_mut"]],axis=1))
    data_dict['inside_data_x'] = data_dict['inside_data_x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))

    data_dict["outside_data_y"] = pd.read_csv("../../data/5_year_death.dataset2.txt", sep="\t")
    data_dict["outside_data_x_cna"] = pd.read_csv("../../data/CNA.5_year_death.dataset2.txt", sep="\t", index_col=0)
    data_dict["outside_data_x_snp"] = pd.read_csv("../../data/SNP.5_year_death.dataset2.txt", sep="\t", index_col=0)
    data_dict["outside_data_x_mut"] = pd.read_csv("../../data/CNGM.5_year_death.dataset2.txt", sep="\t", index_col=0)
    data_dict["outside_data_x"] = pd.DataFrame(pd.concat([data_dict["outside_data_x_cna"],data_dict["outside_data_x_snp"],data_dict["outside_data_x_mut"]],axis=1))
    data_dict['outside_data_x'] = data_dict['outside_data_x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))

    x = load_variable("fusion_fs_featurebuild.data")
    new_data_dict = {}
    y = data_dict["inside_data_y"]
    pheno_x = y[["AGE_AT_DIAGNOSIS","LYMPH_NODES_EXAMINED_POSITIVE","INFERRED_MENOPAUSAL_STATE"]]
    pheno_x.index = data_dict["inside_data_x"].index
    new_data_dict["inside_data_x"] =  pd.DataFrame(pd.concat([data_dict["inside_data_x"][x['train']['x'].columns],pheno_x],axis=1))
    new_data_dict["inside_data_y"] =  pd.read_csv("../../data/5_year_death.dataset1.txt", sep="\t")['Type']


    y = data_dict["outside_data_y"]
    pheno_x = y[["AGE_AT_DIAGNOSIS","LYMPH_NODES_EXAMINED_POSITIVE","INFERRED_MENOPAUSAL_STATE"]]
    pheno_x.index = data_dict["outside_data_x"].index
    new_data_dict["outside_data_x"] =  pd.DataFrame(pd.concat([data_dict["outside_data_x"][x['train']['x'].columns],pheno_x],axis=1))
    new_data_dict["outside_data_y"] =  pd.read_csv("../../data/5_year_death.dataset2.txt", sep="\t")['Type']

    fs_key_list = { 'CNA':[], 'SNV':[], 'CNGM':[], 'pheno':[]}
    for i in  new_data_dict["outside_data_x"].columns:
        if re.match( r'^\d+$', i):
            fs_key_list['CNA'].append(i)
        elif re.match( r'.*-.*', i):
            fs_key_list['SNV'].append(i)
        elif i in ["AGE_AT_DIAGNOSIS","LYMPH_NODES_EXAMINED_POSITIVE","INFERRED_MENOPAUSAL_STATE"]:
            fs_key_list['pheno'].append(i)
            print("pheno",i)
        else:
            fs_key_list['CNGM'].append(i)

    

    configFile = sys.argv[1]

    #数据拆分 inside_data_split_data.data
    data_split(configFile=configFile,data=new_data_dict)
    new_data_dict = {}
    data_dict = {}

    #  标准化 data_normalize.data
    data_dict = load_variable("inside_data_split_data.data")
    merge_data_normalize(configFile=configFile,data=data_dict,fs_class_dict=fs_key_list)
    data_dict={}

    #  特征工程 feature_select.model.data
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
    data_dict['train'] = resample_data(configFile=configFile,data=data_dict['train'])
    
    # # # 最优化模型参数 best_param_model_data.data
    best_param_cv(configFile=configFile, data=data_dict)
    data_dict = {}

    # # 交叉验证
    model = load_variable("best_param_model_data.data")
    if len(method) >= 1:
        data_dict = load_variable("new_feature_build_data.data")
    else:
        data_dict = load_variable("data_normalize.data")
    model_cross_valid(configFile=configFile,model=model,data=data_dict)

    # # 模型训练+测试结果
    model_train_test(configFile=configFile,model=model,data=data_dict)
