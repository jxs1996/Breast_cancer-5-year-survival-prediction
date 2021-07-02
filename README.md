# Breast-cancer-5-year-survival-prediction
A configurable machine learning framework for analysis and prediction of five-year breast cancer survival 

# Requirements
`python 3.8`
`scikit-learn 0.24.2`
`lightgbm 3.2.1`
`matplotlib 3.4.2`

If there is a problem with the package during operation, just install it 

# Usage
##### single model training:
```
python {PATH_TO_SCRIPT_DIR}/single_model_train.py config.ini
```
##### fusion model training:
```
python {PATH_TO_SCRIPT_DIR}/fusion.py config.ini
```
##### Model optimal feature selection:
Extract model features and sort them by weight. Add the features from high to low one by one to retrain the model, calculate the AUC, and select the model with the largest AUC as the final result. 
```
python {PATH_TO_SCRIPT_DIR}/best_feature.num.py config.ini
```
