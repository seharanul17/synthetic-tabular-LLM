import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import accuracy_score ,balanced_accuracy_score,  f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

def compute_metric(label, pred, pred_proba=None, n_class=-1, regression=False):
    metric={}
    if regression:
        metric['MSE'] =  mean_squared_error(label, pred, squared=True)
        metric['MAE'] = mean_absolute_error(label, pred)
    else:
        if n_class > 2:
            metric['ACC'] = accuracy_score(y_true=label, y_pred=pred)   
            metric['BalancedACC'] = balanced_accuracy_score(y_true=label, y_pred=pred)   
            metric['Sensitivity'] = recall_score(y_true=label, y_pred=pred, average='weighted')
            metric['Precision'] = precision_score(y_true=label, y_pred=pred, average='weighted')
            res = []
            for l in np.unique(label):
                prec,recall,_,_ = precision_recall_fscore_support(np.array(label)==l,
                                                              np.array(pred)==l,
                                                              pos_label=True,average=None)
                res.append([l, (np.array(label)==l).sum()/len(label) ,recall[0],recall[1]])
            df = pd.DataFrame(res, columns = ['class', 'n','specificity','sensitivity'])
            metric['Specificity'] = (df['specificity']*df['n']).sum()
            assert np.isclose(metric['Sensitivity'], (df['sensitivity']*df['n']).sum())
            metric['AUC'] = roc_auc_score(y_true=label, y_score=pred_proba, average='weighted', multi_class='ovr')
            metric['F1'] = f1_score(y_true=label, y_pred=pred, average='weighted')    
            metric['KAPPA'] = cohen_kappa_score(label, pred)
            metric['ConfusionMatrix'] = confusion_matrix(y_true=label, y_pred=pred)
    
        elif n_class ==2:
            metric['ACC'] = accuracy_score(y_true=label, y_pred=pred)   
            metric['BalancedACC'] = balanced_accuracy_score(y_true=label, y_pred=pred)   
            metric['Sensitivity'] = recall_score(y_true=label, y_pred=pred, average='binary')
            metric['Precision'] = precision_score(y_true=label, y_pred=pred, average='binary')    
            tn, fp, fn, tp = confusion_matrix(y_true=label, y_pred=pred).ravel()
            metric['Specificity'] = tn / (tn+fp)
            assert metric['Sensitivity'] == tp / (tp+fn)
            metric['AUC'] = roc_auc_score(y_true=label, y_score=pred_proba[:, 1])
            metric['F1'] = f1_score(y_true=label, y_pred=pred, average='binary')
            metric['KAPPA'] = cohen_kappa_score(label, pred)
            metric['ConfusionMatrix'] = confusion_matrix(y_true=label, y_pred=pred)
        else:
            raise

    metric = {key:[value] for key, value in metric.items()}
    return pd.DataFrame.from_dict(metric)

from imblearn.over_sampling import SMOTE, SMOTENC
def fuc_sampling(mode, X_train, y_train, random_state, cate):
    if mode=='SMOTE':
        sampler = SMOTE(random_state=random_state)
        return sampler.fit_resample(X_train, y_train)
    elif mode=='SMOTENC':
        sampler = SMOTENC(random_state=random_state, categorical_features=cate)
        return sampler.fit_resample(X_train, y_train)
    elif mode== 'SMOTEN':
        sampler = SMOTEN(random_state=random_state)
        return sampler.fit_resample(X_train, y_train)
    elif mode == 'ClusterCentroids':
        sampler = ClusterCentroids(random_state=random_state )
        return sampler.fit_resample(X_train, y_train)
    elif mode == 'SMOTETomek':
        sampler = SMOTETomek(random_state=random_state)
        return sampler.fit_resample(X_train, y_train)
    elif mode == 'BorderlineSMOTE':
        sampler = BorderlineSMOTE(random_state=random_state)
        return sampler.fit_resample(X_train, y_train)
    elif mode == 'ADASYN':
        sampler = ADASYN(random_state=random_state)
        return sampler.fit_resample(X_train, y_train)
    elif mode == 'RandomOverSampler':
        sampler = RandomOverSampler(random_state=random_state)
        return sampler.fit_resample(X_train, y_train)
    elif mode == 'OneSidedSelection':
        sampler = OneSidedSelection(random_state=random_state)
        return sampler.fit_resample(X_train, y_train)
    elif mode is None or mode=='None':
        return X_train, y_train
    else:
        raise
        
from sklearn.preprocessing import LabelEncoder
def categorical_variable_encode(configs, X_train, y_train, X_test, y_test, real_data_save_dir):
    org_X_train = pd.read_csv(os.path.join(real_data_save_dir, f'X_train.csv'), index_col='index').values
    org_y_train = pd.read_csv(os.path.join(real_data_save_dir, f'y_train.csv'), index_col='index').values
    org_X_test = pd.read_csv(os.path.join(real_data_save_dir, f'X_test.csv'), index_col='index').values
    org_y_test = pd.read_csv(os.path.join(real_data_save_dir, f'y_test.csv'), index_col='index').values
        
    
    org_X = np.concatenate([org_X_train, org_X_test, X_train],axis=0)
    org_y = np.concatenate([org_y_train, org_y_test, y_train],axis=0)

    X_columns = X_train.columns
    
    if configs['data'] == 'income':
        target_encode = True
        cat_idx= [1,3,5,6,7,8,9,13] 
    elif configs['data'] == 'HELOC':
        target_encode = True # ['Bad', 'Good']
        cat_idx = [] # https://github.com/kathrinse/TabSurvey/blob/main/config/heloc.yml
    elif configs['data'] == 'Diabetes':
        target_encode = True # ['NO', '<30', '>30']
        cat_idx = [2, 3, 4, 5, 10, 11, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    elif configs['data'] == 'Sick':
        target_encode = True # ['negative','sick']
        cat_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26]
    elif configs['data'] == 'Travel':
        target_encode = False # [0, 1]
        cat_idx = [1, 2, 4, 5]
    else:
        raise NotImplementedError
    if target_encode:
        le = LabelEncoder()
        le.fit(org_y)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        
        num_classes = len(le.classes_)

    # Preprocess data
    X_train = X_train.values
    X_test = X_test.values
    
    for i in range(X_train.shape[1]):
        if i in cat_idx:
            le = LabelEncoder()
            le.fit(org_X[:, i].astype(str))
            X_train[:, i] = le.transform(X_train[:, i])
            X_test[:, i] = le.transform(X_test[:, i])

    X_train = pd.DataFrame(X_train, columns=X_columns)
    X_test = pd.DataFrame(X_test, columns=X_columns)
    
    return X_train, y_train, X_test, y_test

def get_data(configs, syn_data_save_dir, real_data_save_dir):
    
    X_test = pd.read_csv(os.path.join(real_data_save_dir, f'X_test.csv'), index_col='index') 
    y_test = pd.read_csv(os.path.join(real_data_save_dir, f'y_test.csv'), index_col='index') 
    n_org_test = X_test.shape[0]
    
    if configs['synModel'] == 'None':
        X_train = pd.read_csv(os.path.join(real_data_save_dir, f'X_train.csv'), index_col='index') 
        y_train = pd.read_csv(os.path.join(real_data_save_dir, f'y_train.csv'), index_col='index') 
            
        n_org_train = X_train.shape[0]
        n_syn = 0
        X_train, y_train, X_test, y_test = categorical_variable_encode(configs, X_train, y_train, X_test, y_test,real_data_save_dir)

    elif configs['synModel'] in ['SMOTE', 'SMOTENC', 'SMOTENorg', 'SMOTENCNorg']:
        X_train = pd.read_csv(os.path.join(real_data_save_dir, f'X_train.csv'), index_col='index') 
        y_train = pd.read_csv(os.path.join(real_data_save_dir, f'y_train.csv'), index_col='index') 
        n_org_train = X_train.shape[0]
        X_train, y_train, X_test, y_test = categorical_variable_encode(configs, X_train, y_train, X_test, y_test,real_data_save_dir)

        X_train, y_train = fuc_sampling(configs['synModel'].replace('Norg',''), X_train, y_train, configs['random_state'], configs['cat_idx'])
        n_syn = X_train.shape[0] - n_org_train

        if configs['synModel'].endswith('Norg'):
            pass
        else:
            X_train = X_train.iloc[n_org_train:]
            y_train = y_train[n_org_train:]
            n_org_train = 0
            assert n_syn == X_train.shape[0]
        
    elif configs['synModel'].endswith('Norg'):
        syn_data_save_dir = syn_data_save_dir.replace(configs['synModel'], configs['synModel'].split('Norg')[0])
        X_train = pd.read_csv(os.path.join(real_data_save_dir, f'X_train.csv'), index_col='index') 
        y_train = pd.read_csv(os.path.join(real_data_save_dir, f'y_train.csv'), index_col='index') 

        n_org_train = X_train.shape[0]
        
        n = configs['synSamples']
        synSamplingIndex=configs['synSamplingIndex']
        if configs['data'] == 'Diabetes':
            n_samples_syn_index = 10000
        elif configs['data'] == 'income':
            n_samples_syn_index = 20000
        else:
            n_samples_syn_index = 1000

        samples = pd.read_csv(os.path.join(syn_data_save_dir, f"{configs['data']}_samples.csv"),index_col='synindex')
        y_train_sample = pd.DataFrame(samples[configs['target']]).iloc[synSamplingIndex*n_samples_syn_index:synSamplingIndex*n_samples_syn_index+n]
        X_train_sample = samples.drop(columns=[configs['target']]).iloc[synSamplingIndex*n_samples_syn_index:synSamplingIndex*n_samples_syn_index+n]

        #Diabetes
        if configs['data'] == 'Diabetes':
            X_train_sample = X_train_sample.fillna('?')
            for c in ['examide', 'citoglipton']:
                if c in X_train_sample.columns:
                    X_train_sample = X_train_sample.drop(columns=[c])
        
        n_syn = X_train_sample.shape[0]
        if n_syn<n:
            raise
        X_train = pd.concat([X_train, X_train_sample])
        y_train = pd.concat([y_train, y_train_sample])
        X_train, y_train, X_test, y_test = categorical_variable_encode(configs, X_train, y_train, X_test, y_test,real_data_save_dir)
        
    else:
        n = configs['synSamples']
        synSamplingIndex=configs['synSamplingIndex']
        if configs['data'] == 'Diabetes':
            n_samples_syn_index = 10000
        elif configs['data'] == 'income':
            n_samples_syn_index = 20000
        else:
            n_samples_syn_index = 1000

        samples = pd.read_csv(os.path.join(syn_data_save_dir, f"{configs['data']}_samples.csv"),index_col='synindex')
        y_train = pd.DataFrame(samples[configs['target']]).iloc[synSamplingIndex*n_samples_syn_index:synSamplingIndex*n_samples_syn_index+n]
        X_train = samples.drop(columns=[configs['target']]).iloc[synSamplingIndex*n_samples_syn_index:synSamplingIndex*n_samples_syn_index+n]

        if configs['data'] == 'Diabetes':
            X_train = X_train.fillna('?')
        
        n_syn = X_train.shape[0]
        n_org_train = 0
        if n_syn<n:
            raise
        X_train, y_train, X_test, y_test = categorical_variable_encode(configs, X_train, y_train, X_test, y_test,real_data_save_dir)

        
    return X_train, y_train, X_test, y_test, n_syn, n_org_train, n_org_test

def init_models(args, random_state):

    models = {

        'XGBoostClassifier_grid':XGBClassifier(learning_rate=args['xg_lr'],max_depth=args['xg_max_depth'],
                                              random_state=random_state,device='cuda'), 
        'CatBoostClassifier_grid':CatBoostClassifier(learning_rate=args['cat_lr'], max_depth=args['cat_max_depth'],
                                          random_state=random_state,verbose=False),
        'LGBMClassifier_grid':lgb.LGBMClassifier(learning_rate=args['lgbm_lr'], max_depth=args['lgbm_max_depth'],
                                          random_state=random_state,verbose_eval=-1,verbose=-1),
        'GradientBoostingClassifier':GradientBoostingClassifier(random_state=random_state),
        }
    return models

DATA2TARGET = {
    'income':'income',
    'Diabetes':'readmitted',
    'HELOC':'RiskPerformance',
    'Sick':'Class',
    'Travel':'Target'
}

DATA2NCLASS = {
    'income':2,
    'Diabetes':3,
    'HELOC':2,
    'Sick':2,
    'Travel':2
}

ML_PARAMS = {
    'Travel':{
        'lr_max_iter':100,
        'dt_max_depth':6,
        'rf_max_depth':12,
        'rf_n_estimators':75,
    },
    'Sick':{
        'lr_max_iter':200,
        'dt_max_depth':10,
        'rf_max_depth':12,
        'rf_n_estimators':90,
    },
    'HELOC':{
        'lr_max_iter':500,
        'dt_max_depth':6,
        'rf_max_depth':12,
        'rf_n_estimators':78,
    },
    'income':{
        'lr_max_iter':1000,
        'dt_max_depth':8,
        'rf_max_depth':12,
        'rf_n_estimators':85,
    },
    'Diabetes':{
        'lr_max_iter':500,
        'dt_max_depth':10,
        'rf_max_depth':20,
        'rf_n_estimators':120,
    }
}

df_all_result = pd.DataFrame()
DATA_NAME = 'Sick'
synSamplingIndex = 0
n = 1000
for sM in ['Sick_STPromptNorg']:
    configs={
        'data': DATA_NAME,
        'target':DATA2TARGET[DATA_NAME],
        'n_class':DATA2NCLASS[DATA_NAME],
        'is_regression':False,

        # hyperparams
        'lr_max_iter':ML_PARAMS[DATA_NAME]['lr_max_iter'],
        'dt_max_depth':ML_PARAMS[DATA_NAME]['dt_max_depth'],
        'rf_max_depth':ML_PARAMS[DATA_NAME]['rf_max_depth'],
        'rf_n_estimators':ML_PARAMS[DATA_NAME]['rf_n_estimators'],

        # xgboost
        'xg_max_depth':4,
        'xg_lr':0.03,

        # catboost
        'cat_max_depth':6,
        'cat_lr':0.003,

        # lightGBM
        'lgbm_max_depth':3,
        'lgbm_lr':0.1,

        # synthetic data
        'synModel':sM,
        'synSamples':n,

        'synSamplingIndex':synSamplingIndex,
    }
    models = init_models(configs, 42)
    syn_data_save_dir=f"../../data/syndata/{configs['synModel']}"
    real_data_save_dir=f"../../data/realdata/{configs['data']}"

    for k in tqdm(models.keys()):
        configs['model']=k    
        for random_state in range(5):
            configs['random_state']=random_state
            X_train, y_train, X_test, y_test, n_syn, n_org_train, n_org_test = get_data(configs, syn_data_save_dir, 
                                                                                        real_data_save_dir)
            df_save = pd.DataFrame([configs])
            df_save['n_syn'] = n_syn
            df_save['n_org_train']=n_org_train
            df_save['n_org_test']=n_org_test                

            model = init_models(configs, random_state)[k]

            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train_np = scaler.transform(X_train)
            X_test_np = scaler.transform(X_test)

            model.fit(X_train_np, y_train)
            pred_test = model.predict(X_test_np)
            pred_test_proba = model.predict_proba(X_test_np)

            df_metric = compute_metric(y_test, pred_test, pred_test_proba, configs['n_class'], regression=configs['is_regression'])

            df_save = pd.concat([df_save, df_metric], axis=1)
            df_all_result= pd.concat([df_all_result, df_save.copy()])
            
            
df = df_all_result[['data','model','synModel','F1','BalancedACC']]
df = (df.groupby(['data','model','synModel']).mean()*100).reset_index()

print(tabulate(df, headers='keys', tablefmt='psql'))
