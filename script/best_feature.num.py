from single_model_train import *


'''
读取数据整理结果 
'''
def load_data(configFile):
    cf = configparser.ConfigParser()
    cf.read(configFile)
    # 读取训练结果
    model_result = load_variable(cf.get("best_fs_data","model_result"))
    # 读取训练数据 来自特征构建或者直接标准化的数据
    data = load_variable(cf.get("best_fs_data","fs_data"))

    data_dict = {}
    best_data_dict = {}
    fs_list = ()
    fs = []
    x = []
    y = []
    fs_num = 0
    max_auc = 0
    # 如果训练数据为普通数据
    if(cf.get("best_fs_data","model_type") == 'single'):
        # 如果模型特征权重为feature_importances_
        if(cf.get("best_fs_data","fs_type") == 'feature_importances_'):
            fs_list = dict(zip([data['train']['x'].columns[int(i)] for i in range(0,len(model_result.feature_importances_))], model_result.feature_importances_))
            fs_list = sorted(fs_list.items(),key = lambda kv:(kv[1],kv[0]),reverse=True)
        # 如果模型特征权重为
        if(cf.get("best_fs_data","fs_type") == 'coef_'):
            fs_list = dict(zip([data['train']['x'].columns[int(i)] for i in range(0,len(model_result.coef_[0]))], np.maximum(model_result.coef_[0], -model_result.coef_[0])))
            fs_list = sorted(fs_list.items(),key = lambda kv:(kv[1],kv[0]),reverse=True)

    for i in fs_list:
        fs_num += 1
        fs.append(i[0])
        data_dict['train'] = {}
        data_dict['train']['x'] = pd.DataFrame(data['train']['x'],columns=fs)
        data_dict['train']['y'] = data['train']['y']
        data_dict['test'] = {}
        data_dict['test']['x'] = pd.DataFrame(data['test']['x'],columns=fs)
        data_dict['test']['y'] = data['test']['y']
        if(cf.get("best_fs_data","test_out") == 'yes'):
            data_dict['test_out'] = {}
            data_dict['test_out']['x'] = pd.DataFrame(data['test_out']['x'],columns=fs)
            data_dict['test_out']['y'] = data['test_out']['y']
        data_dict['test']['x'] = data_dict['test']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
        data_dict['train']['x'] = data_dict['train']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
        data_dict['test_out']['x'] = data_dict['test_out']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
        weight_class_dict = Classifier_get_class_weight(data_dict['train']['y'])
        ins_weight_lst = [weight_class_dict[i] for i in data_dict['train']['y']]
        # 读取超参model
        model = load_variable(cf.get("best_fs_data","best_param_model_data"))
        modelnew = model
        modelnew.fit(data_dict['train']['x'],data_dict['train']['y'], sample_weight=ins_weight_lst)
        y_pred = modelnew.predict_proba(data_dict['test']['x'])
        y_pred = y_pred[:,1]
        fpr,tpr,thresholds=roc_curve(list(data_dict['test']['y']),list(y_pred))
        roc_auc=auc(fpr,tpr)
        x.append(fs_num)
        y.append(roc_auc)
        if(roc_auc > max_auc):
            max_auc = roc_auc
            best_fs_num = fs_num
            best_data_dict['train'] = {}
            best_data_dict['train']['x'] = data_dict['train']['x']
            best_data_dict['train']['y'] = data_dict['train']['y']
            best_data_dict['test'] = {}
            best_data_dict['test']['x'] = data_dict['test']['x']
            best_data_dict['test']['y'] = data_dict['test']['y']
            if(cf.get("best_fs_data","test_out") == 'yes'):
                best_data_dict['test_out'] = {}
                best_data_dict['test_out']['x'] = data_dict['test_out']['x']
                best_data_dict['test_out']['y'] = data_dict['test_out']['y']
    
    a = '%.2f' % max_auc
    b = '%.2f' % best_fs_num
    best_fs_select = [x,y]
    save_variable(best_fs_select,"best_fs_select_result.data")
    plt.plot(x,y,lw=2,alpha=1,label='MAX_AUC={} feature={}'.format(a,b))
    plt.xlabel('feature number')
    plt.ylabel('AUC')
    plt.legend(loc='upper right')
    plt.savefig("best_auc_fs_num.pdf", format='pdf')
    plt.close()
    save_variable(best_data_dict,"new_feature_build_data")
    # 读取超参model
    model = load_variable(cf.get("best_fs_data","best_param_model_data"))
    return(best_data_dict,model)


# 需要超参的model，model训练结果，数据标准化的结果
if __name__ == "__main__":
    configFile = sys.argv[1]
    data_dict,model = load_data(configFile)
    cf = configparser.ConfigParser()
    cf.read(configFile)
    model = load_variable(cf.get("best_fs_data","best_param_model_data"))
    data_dict['test']['x'] = data_dict['test']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
    data_dict['train']['x'] = data_dict['train']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
    data_dict['test_out']['x'] = data_dict['test_out']['x'].rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '-', x))
    model_cross_valid(configFile=configFile,model=model,data=data_dict)

    # # 模型训练+测试结果
    model_train_test(configFile=configFile,model=model,data=data_dict)

