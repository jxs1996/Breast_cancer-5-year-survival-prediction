import configparser
import logging
import json
import re

class ReSampleMethod(object):
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
    主入口
    '''
    def main(self,configFile,data):
        cf = configparser.ConfigParser()
        cf.read(configFile)
        isUse = cf.get("resample","isUse")
        if isUse == 'True':
            if cf.get("resample","SMOTEENN") == 'True':
                data_dict = self.SMOTEENN(configFile,data)
            elif cf.get("resample","SMOTETomek") == 'True':
                data_dict = self.SMOTETomek(configFile,data)
            elif cf.get("resample","RandomOverSampler") == 'True':
                data_dict = self.RandomOverSampler(configFile,data)
            elif cf.get("resample","DownSample") == 'True':
                data_dict = self.DownSample(configFile,data)
            elif cf.get("resample","ADASYN") == 'True':
                data_dict = self.ADASYN(configFile,data)
            return data_dict
        else:
            return data

    '''
    SMOTEENN
    '''
    def SMOTEENN(self,configFile,data):
        from imblearn.combine import SMOTEENN
        cf = configparser.ConfigParser()
        cf.read(configFile)
        sampling_strategy = str(cf.get("resample","sampling_strategy"))
        random = int(cf.get("data","random"))
        matchObj = re.match( r'.*[^0-9\.].*', sampling_strategy, re.I)
        if not matchObj:
            sampling_strategy = float(sampling_strategy)

        model = SMOTEENN(sampling_strategy=sampling_strategy,random_state=random)
        data['y'].index = [int(x) for x in range(0,len(data['y']))]
        data['x'].index = [int(x) for x in range(0,len(data['x']))]
        X_resampled, y_resampled = model.fit_resample(data['x'], data['y'])
        self.logging_config(u"resample class \n {}".format(y_resampled.value_counts()),"info")
        data_dict = {'x':X_resampled,'y':y_resampled}
        return data_dict

    '''
    SMOTEENN
    '''
    def SMOTETomek(self,configFile,data):
        from imblearn.combine import SMOTETomek
        cf = configparser.ConfigParser()
        cf.read(configFile)
        sampling_strategy = str(cf.get("resample","sampling_strategy"))
        random = int(cf.get("data","random"))
        matchObj = re.match( r'.*[^0-9\.].*', sampling_strategy, re.I)
        if not matchObj:
            sampling_strategy = float(sampling_strategy)

        model = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random)
        data['y'].index = [int(x) for x in range(0,len(data['y']))]
        data['x'].index = [int(x) for x in range(0,len(data['x']))]
        X_resampled, y_resampled = model.fit_resample(data['x'], data['y'])
        self.logging_config(u"resample class \n {}".format(y_resampled.value_counts()),"info")
        data_dict = {'x':X_resampled,'y':y_resampled}
        return data_dict


    '''
    RandomOverSampler
    '''
    def RandomOverSampler(self,configFile,data):
        from imblearn.over_sampling import RandomOverSampler
        cf = configparser.ConfigParser()
        cf.read(configFile)
        sampling_strategy = str(cf.get("resample","sampling_strategy"))
        random = int(cf.get("data","random"))
        matchObj = re.match( r'.*[^0-9\.].*', sampling_strategy, re.I)
        if not matchObj:
            sampling_strategy = float(sampling_strategy)

        model = RandomOverSampler(sampling_strategy=sampling_strategy,random_state=random)
        data['y'].index = [int(x) for x in range(0,len(data['y']))]
        data['x'].index = [int(x) for x in range(0,len(data['x']))]
        X_resampled, y_resampled = model.fit_resample(data['x'], data['y'])
        self.logging_config(u"resample class \n {}".format(y_resampled.value_counts()),"info")
        data_dict = {'x':X_resampled,'y':y_resampled}
        return data_dict

    '''
    ADASYN
    '''
    def ADASYN(self,configFile,data):
        from imblearn.over_sampling import ADASYN
        cf = configparser.ConfigParser()
        cf.read(configFile)
        sampling_strategy = str(cf.get("resample","sampling_strategy"))
        random = int(cf.get("data","random"))
        matchObj = re.match( r'.*[^0-9\.].*', sampling_strategy, re.I)
        if not matchObj:
            sampling_strategy = float(sampling_strategy)

        model = ADASYN(sampling_strategy=sampling_strategy,random_state=random)
        data['y'].index = [int(x) for x in range(0,len(data['y']))]
        data['x'].index = [int(x) for x in range(0,len(data['x']))]
        X_resampled, y_resampled = model.fit_resample(data['x'], data['y'])
        self.logging_config(u"resample class \n {}".format(y_resampled.value_counts()),"info")
        data_dict = {'x':X_resampled,'y':y_resampled}
        return data_dict

    '''
    DownSample
    '''
    def DownSample(self,configFile,data,replace = 1,random_state = -9):
        from sklearn.utils import resample
        cf = configparser.ConfigParser()
        cf.read(configFile)
        random = int(cf.get("data","random"))
        if random_state != -9:
            random = random_state
        index_dict = {}
        data['y'].index = [int(x) for x in range(0,len(data['y']))]
        data['x'].index = [int(x) for x in range(0,len(data['x']))]
        for i in data['y'].index:
            if data['y'][i] not in index_dict:
                index_dict[data['y'][i]] = []
                index_dict[data['y'][i]].append(i)
            else:
                index_dict[data['y'][i]].append(i)

        min_type = ''
        min_type_sample_num = len(data['y'])
        max_type = ''
        max_type_sample_num = 0
        for i in index_dict.keys():
            if len(index_dict[i]) < min_type_sample_num:
                min_type_sample_num = len(index_dict[i])
                min_type = i
            if len(index_dict[i]) > max_type_sample_num:
                max_type_sample_num = len(index_dict[i])
                max_type = i
            if random_state == -9:
                self.logging_config(u"Train Data Type {}: {}".format(i,len(index_dict[i])),"info")
        if random_state == -9:
            self.logging_config(u"Rsample max Type {}: {} -> {}".format(max_type, max_type_sample_num, min_type_sample_num),"info")
        
        max_index = resample(index_dict[max_type] , n_samples=min_type_sample_num, replace=replace, random_state = random)
        sample_index = index_dict[min_type] + max_index
        data_dict = {"x":data['x'].iloc[sample_index],"y":data['y'].iloc[sample_index]}
        return data_dict