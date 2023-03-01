import configparser
import pandas as pd
import logging
import numpy as np

class NormalizeMethod(object):
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
    StandardScalerMethod
    '''
    def StandardScalerMethod(self,configFile,data):
        from sklearn.preprocessing import StandardScaler
        data_dict = {'train':{},'test':{}}
        scaler = StandardScaler()
        scaler.fit(X=data['train']['x'])
        data_dict['train']['x'] = pd.DataFrame(scaler.transform(data['train']['x']),columns=data['train']['x'].columns,index=data['train']['x'].index)
        data_dict['train']['y'] = data['train']['y']
        x = data_dict['train']['y']
        data_dict['test']['x'] = pd.DataFrame(scaler.transform(data['test']['x']),columns=data['train']['x'].columns,index=data['test']['x'].index)
        data_dict['test']['y'] = data['test']['y']
        if 'test_out' in data:
            data_dict['test_out'] = {}
            data_dict['test_out']['x'] = pd.DataFrame(scaler.transform(data['test_out']['x']),columns=data['train']['x'].columns,index=data['test_out']['x'].index)
            data_dict['test_out']['y'] = data['test_out']['y']
        return data_dict

    '''
    LeaveOneOutEncoderMethod
    '''
    def LeaveOneOutEncoderMethod(self,configFile,data):
        import category_encoders as ce
        data_dict = {'train':{},'test':{}}
        scaler = ce.LeaveOneOutEncoder(cols=data['train']['x'].columns)
        scaler.fit(data['train']['x'],data['train']['y'])
        data_dict['train']['x'] = pd.DataFrame(scaler.transform(data['train']['x']))
        data_dict['train']['y'] = data['train']['y']
        data_dict['test']['x'] = pd.DataFrame(scaler.transform(data['test']['x']))
        data_dict['test']['y'] = data['test']['y']
        if 'test_out' in data:
            data_dict['test_out'] = {}
            data_dict['test_out']['x'] = pd.DataFrame(scaler.transform(data['test_out']['x']))
            data_dict['test_out']['y'] = data['test_out']['y']
        return data_dict
        
    '''
    MinMaxScaler
    '''
    def MinMaxScalerMethod(self,configFile,data):
        from sklearn.preprocessing import MinMaxScaler
        data_dict = {'train':{},'test':{}}
        scaler = MinMaxScaler()
        scaler.fit(X=data['train']['x'])
        data_dict['train']['x'] = pd.DataFrame(scaler.transform(data['train']['x']))
        data_dict['train']['y'] = data['train']['y']
        data_dict['train']['x'].columns = data['train']['x'].columns
        data_dict['train']['x'].index = data['train']['x'].index 
        x = data_dict['train']['y']
        data_dict['test']['x'] = pd.DataFrame(scaler.transform(data['test']['x']))
        data_dict['test']['y'] = data['test']['y']
        data_dict['test']['x'].columns = data['test']['x'].columns
        data_dict['test']['x'].index = data['test']['x'].index
        if 'test_out' in data:
            data_dict['test_out'] = {}
            data_dict['test_out']['x'] = pd.DataFrame(scaler.transform(data['test_out']['x']))
            data_dict['test_out']['y'] = data['test_out']['y']
            data_dict['test_out']['x'].columns = data['test_out']['x'].columns
            data_dict['test_out']['x'].index = data['test_out']['x'].index
        return data_dict
    
    '''
    logMinMaxScaler
    '''
    def logMinMaxScalerMethod(self,configFile,data):
        from sklearn.preprocessing import MinMaxScaler
        data_dict = {'train':{},'test':{}}
        scaler = MinMaxScaler()
        data['train']['x'] = np.log(data['train']['x']+1)
        data['test']['x'] = np.log(data['test']['x']+1)
        scaler.fit(X=data['train']['x'])
        data_dict['train']['x'] = pd.DataFrame(scaler.transform(data['train']['x']))
        data_dict['train']['y'] = data['train']['y']
        data_dict['train']['x'].columns = data['train']['x'].columns
        data_dict['train']['x'].index = data['train']['x'].index 
        x = data_dict['train']['y']
        data_dict['test']['x'] = pd.DataFrame(scaler.transform(data['test']['x']))
        data_dict['test']['y'] = data['test']['y']
        data_dict['test']['x'].columns = data['test']['x'].columns
        data_dict['test']['x'].index = data['test']['x'].index
        if 'test_out' in data:
            data_dict['test_out'] = {}
            data['test_out']['x'] = np.log(data['test_out']['x']+1)
            data_dict['test_out']['x'] = pd.DataFrame(scaler.transform(data['test_out']['x']))
            data_dict['test_out']['y'] = data['test_out']['y']
            data_dict['test_out']['x'].columns = data['test_out']['x'].columns
            data_dict['test_out']['x'].index = data['test_out']['x'].index
        return data_dict