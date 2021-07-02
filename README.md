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
python {PATH_TO_SCRIPT_DIR}/fusion.py fusion.config.ini
```
##### fusion_add_pheno model training:
Phenotypic data (age, gender, number of positive lymph nodes, and menopausal status) are fused with these characteristics to see if phenotypic data can improve the performance of the model. 
```
python {PATH_TO_SCRIPT_DIR}/fusion_add_pheno.py best_fs_num_config.ini
```
##### Model optimal feature selection:
Extract model features and sort them by weight. Add the features from high to low one by one to retrain the model, calculate the AUC, and select the model with the largest AUC as the final result. 
```
python {PATH_TO_SCRIPT_DIR}/best_feature.num.py best_fs_num_config.ini
```

# Description 
1. The data of the BRCA project is stored in the data folder. After pulling it, it needs to be decompressed before it can be used normally.
2. The cbc, rdf, and lgb folders correspond to three different algorithm models, and the folders are the training results of different characteristics (cna, snp, mut, fusion, fusion_add_pheno).
3. The results of the article can be viewed in duty.log, and other intermediate data has been deleted. If you need to try to run it yourself, the results of duty.log may be inconsistent, the intermediate data will be saved in the *.data file, and the image will be saved in *pdf.
4. The best result of each model is in the fs_record folder.
5. The order of execution, cna, mut, snp> fusion> fusion_add_pheno. If the file contains a *sh file, run the *.sh file first, and then run the python command.
6. The configuration file can configure various parameters of the model. 

![image](https://github.com/jxs1996/Breast_cancer-5-year-survival-prediction/blob/main/pipeline.png)
