import openai
import os
from dotenv import load_dotenv
import pandas as pd
import string
import random
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from util import get_prompt_conclass, parse_prompt2df, parse_result, get_unique_features, make_final_prompt

openai_key = "Your-OpenAI-Key"

params = {
    "openai_key":openai_key,
    "model":"gpt-3.5-turbo-16k-0613",
    "DATA_NAME":"Sick",
    "TARGET":"Class",
    "N_CLASS":2,
    "N_SAMPLES_PER_CLASS":15,
    "N_SET":4,
    "USE_RANDOM_WORD":True,
    "N_BATCH":20,
    "MODEL_NAME":"Sick_STPrompt_New",
    "N_TARGET_SAMPLES":1000,
}

params.update({
    "DATA_DIR":f"../../data/realdata/{params['DATA_NAME']}",
    "SAVE_DIR":f"../../data/syndata/{params['MODEL_NAME']}"
})


# init API
load_dotenv()
openai.api_key = params['openai_key']
os.environ["OPENAI_API_KEY"] = params['openai_key']

llm = ChatOpenAI(model=params["model"])
output_parser = StrOutputParser()

# init params
DATA_NAME=params['DATA_NAME']
TARGET=params['TARGET']
REAL_DATA_SAVE_DIR=params['DATA_DIR']
symModel=params['MODEL_NAME']
SYN_DATA_SAVE_DIR=params['SAVE_DIR']
os.makedirs(SYN_DATA_SAVE_DIR, exist_ok=True)

# read real data
X_train = pd.read_csv(os.path.join(REAL_DATA_SAVE_DIR, f'X_train.csv'), index_col='index')
y_train = pd.read_csv(os.path.join(REAL_DATA_SAVE_DIR, f'y_train.csv'), index_col='index')
data = pd.concat((y_train, X_train),axis=1)

# Sick dataset
CATEGORICAL_FEATURES = ['sex', 'on_thyroxine', 'query_on_thyroxine', 
                        'on_antithyroid_medication', 'sick', 'pregnant', 
                        'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
                        'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 
                        'hypopituitary', 'psych', 'TSH_measured', 'T3_measured', 
                        'TT4_measured', 'T4U_measured', 'FTI_measured', 
                        'referral_source', 'Class']
NAME_COLS = ','.join(data.columns) + '\n'    
unique_categorical_features=get_unique_features(data, CATEGORICAL_FEATURES)
unique_categorical_features['Class'] =['sick', 'negative']
cat_idx = []
for i,c in enumerate(X_train.columns):
    if c in CATEGORICAL_FEATURES:
        cat_idx.append(i)
        
N_CLASS = params['N_CLASS']
N_SAMPLES_PER_CLASS = params['N_SAMPLES_PER_CLASS']
N_SET= params['N_SET']
N_BATCH = params['N_BATCH']
N_SAMPLES_TOTAL = N_SAMPLES_PER_CLASS*N_SET*N_BATCH

# apply random word stretagy
if params['USE_RANDOM_WORD']:
    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        first = ''.join(random.choice(string.ascii_uppercase) for _ in range(1))
        left = ''.join(random.choice(chars) for _ in range(size-1))
        return first+left
    
    def make_random_categorical_values(unique_categorical_features):
        mapper = {}
        mapper_r = {}
        new_unique_categorical_features = {}
        for c in unique_categorical_features:
            mapper[c] ={}
            mapper_r[c]={}
            new_unique_categorical_features[c] = []
    
            for v in unique_categorical_features[c]:
                a = id_generator(3)
                new_unique_categorical_features[c].append(a)
    
                mapper[c][v] = a
                mapper_r[c][a] = v
        return mapper, mapper_r, new_unique_categorical_features
    
    mapper, mapper_r, unique_categorical_features = make_random_categorical_values(unique_categorical_features)
        
    for c in mapper:
        data[c] = data[c].map(lambda x: mapper[c][x])
        
        
        
# make prompt template
initial_prompt="""Class: hypothyroidism is a condition in which the thyroid gland is underperforming or producing too little thyroid hormone,
age: the age of an patient,
sex: the biological sex of an patient,
TSH: thyroid stimulating hormone,
T3: triiodothyronine hormone,
TT4: total levothyroxine hormone,
T4U: levothyroxine hormone uptake,
FTI: free levothyroxine hormone index,
referral_source: institution that supplied the thyroid disease record.\n\n
"""

numbering=['A','B','C','D']

prompt=get_prompt_conclass(initial_prompt, numbering, N_SAMPLES_PER_CLASS,N_CLASS,N_SET, NAME_COLS)

# init chain
template1 = prompt
template1_prompt = PromptTemplate.from_template(template1)

llm1 = (
    template1_prompt
    | llm
    | output_parser
)
# print example inputs for LLM
final_prompt, _ = make_final_prompt(unique_categorical_features, TARGET, data, template1_prompt,
                           N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS)


input_df_all=pd.DataFrame()
synthetic_df_all=pd.DataFrame()
text_results = []

columns1=data.columns
columns2=list(data.columns)

err=[]

# generate synthetic dataset
while len(synthetic_df_all) < params['N_TARGET_SAMPLES']:
    final_prompt, inputs_batch = make_final_prompt(unique_categorical_features, TARGET, data, template1_prompt,
                                                   N_SAMPLES_TOTAL, N_BATCH, N_SAMPLES_PER_CLASS, N_SET, NAME_COLS, N_CLASS)
    
    inter_text = llm1.batch(inputs_batch)
    
    for i in range(len(inter_text)):
        try:
            text_results.append(final_prompt[i].text+inter_text[i])
            input_df = parse_prompt2df(final_prompt[i].text, split=NAME_COLS, inital_prompt=initial_prompt, col_name=columns1)
            result_df = parse_result(inter_text[i], NAME_COLS, columns2, CATEGORICAL_FEATURES, unique_categorical_features)
            
            input_df_all = pd.concat([input_df_all, input_df], axis=0)
            synthetic_df_all = pd.concat([synthetic_df_all, result_df], axis=0)
        except Exception as e:
            err.append(inter_text[i])
    print('Number of Generated Samples:', len(synthetic_df_all),'/',params['N_TARGET_SAMPLES'])
    
# random words to original values
synthetic_df_all_r = synthetic_df_all.copy()

if params['USE_RANDOM_WORD']:
    for c in mapper_r:
        input_df_all[c] = input_df_all[c].map(lambda x: mapper_r[c][x] if x in mapper_r[c] else x)
    for c in mapper_r:
        synthetic_df_all_r[c] = synthetic_df_all_r[c].map(lambda x: mapper_r[c][x] if x in mapper_r[c] else x)
        
# save
file_name=os.path.join(SYN_DATA_SAVE_DIR, f'{DATA_NAME}_samples.csv')

# save prompt template
with open(file_name.replace('.csv','.txt'),'w') as f:
    f.write(template1+'\n===\n'+final_prompt[0].text)

# save synthetic tabular data
synthetic_df_all_r.to_csv(file_name, index_label='synindex')
print('Saved:', file_name)