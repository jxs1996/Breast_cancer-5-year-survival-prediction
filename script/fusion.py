from single_model_train import *

'''
merge读取标签数据 
'''
def load_merge_y_data(configFile):
    cf = configparser.ConfigParser()
    cf.read(configFile)
    data_dict = {}
    # 读取数据部分高度定制化，可以自己定制
    if cf.get("data","inside_data_y"):
        inside_data_y = pd.read_csv(cf.get("data","inside_data_y"), sep="\t")['label']
        data_dict["inside_data_y"] = inside_data_y
        logging_config(u"reading inside_data_y...","info")

    if cf.get("data","outside_data_y"):
        outside_data_y  = pd.read_csv(cf.get("data","outside_data_y"), sep="\t")['label']
        data_dict["outside_data_y"] = outside_data_y
        logging_config(u"reading outside_data_y...","info")

    return data_dict


def merge_data(configFile,data):
    cf = configparser.ConfigParser()
    cf.read(configFile)
    fs_label = json.loads(cf.get("FS_model","fs_label"))
    fs_model_path = json.loads(cf.get("FS_model","fs_model_path"))
    fs_data_path = json.loads(cf.get("FS_model","fs_data_path"))
    fs_outside_data_path = json.loads(cf.get("FS_model","fs_outside_data_path"))
    tmp_data = []
    tmp_data2 = []
    fs_class_dict = {}
    for j in range(0,len(fs_label)):
        fs_model = load_variable(fs_model_path[j])
        fs_model['train']['x'] = fs_model['train']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+','-',x))
        fs_list = fs_model['train']['x'].columns
        
        # 读取数据并且按列名取值
        fs_data = pd.read_csv(fs_data_path[j], sep="\t", index_col=0)
        fs_data = fs_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
        tmp_data.append(pd.DataFrame(pd.DataFrame(fs_data)[fs_list]))
        if len(fs_outside_data_path) > 0:
            fs_data = pd.read_csv(fs_outside_data_path[j], sep="\t", index_col=0)
            fs_data = fs_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
            tmp_data2.append(pd.DataFrame(pd.DataFrame(fs_data)[fs_list]))
        fs_class_dict[fs_label[j]] = fs_list

    data_dict = {}
    data_dict["inside_data_x"] = pd.DataFrame(pd.concat(tmp_data,axis=1))
    data_dict["inside_data_y"] = data["inside_data_y"]
    if len(fs_outside_data_path) > 0:
        data_dict["outside_data_x"] = pd.DataFrame(pd.concat(tmp_data2,axis=1))
        data_dict["outside_data_y"] = data["outside_data_y"]
    data = {}
    return(data_dict,fs_class_dict)


'''
merge数据标准化
'''
def merge_data_normalize(configFile,data,fs_class_dict):
    cf = configparser.ConfigParser()
    cf.read(configFile)
    fs_label = json.loads(cf.get("FS_model","fs_label"))
    fs_standardization = json.loads(cf.get("FS_model","fs_standardization"))
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
    #  读取merge_config.ini配置文件
    configFile = sys.argv[1]
    data_dict = load_merge_y_data(configFile)
    #  合并特征
    data_dict,fs_class_dict = merge_data(configFile,data_dict)
    
    #  数据拆分 inside_data_split_data.data
    data_split(configFile=configFile,data=data_dict)
    data_dict = {}
    
    #  标准化 data_normalize.data
    data_dict = load_variable("inside_data_split_data.data")
    merge_data_normalize(configFile=configFile,data=data_dict,fs_class_dict=fs_class_dict)
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
