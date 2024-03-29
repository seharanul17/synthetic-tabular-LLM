import os
from sklearn import datasets
import openai
import os
import IPython
from langchain.llms import OpenAI
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from io import StringIO


def get_prompt_conclass(inital_prompt, numbering, n_samples_per_class,nclass,nset, name_cols):
    prompt=""
    for i in range(nset):
        prompt+=name_cols
        for j in range(nclass):
            prompt+=f'{numbering[j]}.\n'
            for k in range(n_samples_per_class):
                prompt +='{'+f'v{i*(n_samples_per_class*nclass)+j*n_samples_per_class+k}'+'}'
            prompt += f'\n'
        prompt += f'\n'  
    prompt+=name_cols
    
    prompt = inital_prompt+prompt
    return prompt
    
def filtering_categorical(result_df, categorical_features, unique_features):
    org_df = result_df.copy()
    shape_before = org_df.shape
    
    for column in categorical_features:
        if column=='Target':
            result_df = result_df[result_df[column].map(lambda x: int(x) in unique_features[column])]
        else:
            result_df = result_df[result_df[column].map(lambda x: x in unique_features[column])]
        
    if shape_before!=result_df.shape:
        for column in categorical_features:
            filtered = org_df[org_df[column].map(lambda x: x not in unique_features[column])]
    return result_df
    
def parse_prompt2df(one_prompt, split, inital_prompt, col_name):
    one_prompt = one_prompt.replace(inital_prompt, '')
    input_prompt_data = one_prompt.split(split)
    input_prompt_data = [x for x in input_prompt_data if x]
    input_prompt_data = '\n'.join(input_prompt_data)
    input_df = pd.read_csv(StringIO(input_prompt_data), sep=",", header=None, names=col_name)
    input_df = input_df.dropna()
    return input_df


def parse_result(one_prompt, name_cols, col_name, categorical_features, unique_features, filter_flag=True):
    one_prompt = one_prompt.replace(name_cols, '')
    result_df = pd.read_csv(StringIO(one_prompt), sep=",", header=None, names=col_name)
    result_df = result_df.dropna()
    if filter_flag:
        result_df = filtering_categorical(result_df, categorical_features, unique_features)
    return result_df
    

def get_unique_features(data, categorical_features):
    unique_features={}
    for column in categorical_features:
        try:
            unique_features[column] = sorted(data[column].unique())
        except:
            unique_features[column] = data[column].unique()
    return unique_features

def get_sampleidx_from_data(unique_features, target, n_samples_total, n_batch, n_samples_per_class, nset, name_cols, data):
    # input sampling
    unique_classes = unique_features[target]
    random_idx_batch_list=[]
    target_df_list=[]
    for c in unique_classes:
        target_df=data[data[target]==c]
        if len(target_df) < n_samples_total:
            replace_flag=True
        else:
            replace_flag=False
        random_idx_batch = np.random.choice(len(target_df), n_samples_total, replace=replace_flag)
        random_idx_batch = random_idx_batch.reshape(n_batch,nset,1,n_samples_per_class)
        random_idx_batch_list.append(random_idx_batch)
        target_df_list.append(target_df)
    random_idx_batch_list = np.concatenate(random_idx_batch_list, axis=2)
    return random_idx_batch_list, target_df_list

def get_input_from_idx(target_df_list, random_idx_batch_list, data, n_batch, n_samples_per_class, nset, nclass ):
    fv_cols = ('{},'*len(data.columns))[:-1] + '\n' 
    # input selection 
    inputs_batch = []
    for batch_idx in range(n_batch):
        inputs = {}
        for i in range(nset): #5
            for j in range(nclass): #2
                target_df = target_df_list[j]
                for k in range(n_samples_per_class): #3
                    idx = random_idx_batch_list[batch_idx, i,j,k]
                    inputs[f'v{i*(n_samples_per_class*nclass)+j*n_samples_per_class+k}']=fv_cols.format(
                        *target_df.iloc[idx].values
                    )
        inputs_batch.append(inputs)
    return inputs_batch
    
def make_final_prompt(unique_categorical_features, TARGET, data, template1_prompt,
                      N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS):
    
    random_idx_batch_list, target_df_list = get_sampleidx_from_data(unique_categorical_features, TARGET, 
                                                                    N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, data)
    inputs_batch = get_input_from_idx(target_df_list, random_idx_batch_list, data, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, N_CLASS)
    final_prompt = template1_prompt.batch(inputs_batch)
    return final_prompt, inputs_batch

def useThis(one_prompt):
    char = one_prompt[0]
    if char.isdigit() and int(char) in [0,1,2,3,4]:
        return True, int(char)
    else:
        return False, None