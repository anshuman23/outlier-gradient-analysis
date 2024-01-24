import pickle
from src.lora_model import LORAEngineGeneration
from src.influence import IFEngineGeneration

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.environ["WANDB_MODE"] = 'disabled'

DATASET = 'math_with_reason' #Choose from math_with_reason, math_without_reason, grammars

#Finetune LLM
# !python /YOUR-DATAINF-PATH/DataInf/src/sft_trainer.py \
#     --model_name /YOUR-LLAMA-PATH/llama/models_hf/llama-2-13b-chat \
#     --dataset_name /YOUR-DATAINF-PATH/DataInf/datasets/math_with_reason_train.hf \
#     --output_dir /YOUR-DATAINF-PATH/DataInf/models/math_with_reason_13bf \
#     --dataset_text_field text \
#     --load_in_8bit \
#     --use_peft

#Set paths: # Please change the following objects to  "YOUR-LLAMA-PATH" and "YOUR-DATAINF-PATH" respectively
base_path = ""
project_path ="" 
lora_engine = LORAEngineGeneration(base_path=base_path, 
                                   project_path=project_path,
                                   dataset_name=DATASET)

#Test model prediction works
prompt = """
Emily scored 20 points in the first game, 50 points in the second, 10 in the third, and 30 in the fourth game. What is her total points? Output only the answer.
"""
inputs = lora_engine.tokenizer(prompt, return_tensors="pt").to("cuda")

generate_ids = lora_engine.model.generate(input_ids=inputs.input_ids, 
                                          max_length=128,
                                          pad_token_id=lora_engine.tokenizer.eos_token_id)
output = lora_engine.tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]

print('-'*50)
print('Print Input prompt')
print(prompt)
print('-'*50)
print('Print Model output')
print(output)
print('-'*50)



#Obtain gradients
tokenized_datasets, collate_fn = lora_engine.create_tokenized_datasets()
tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)

with open('saved_grads/{}/train.pkl'.format(DATASET), 'wb') as fp:
    pickle.dump(tr_grad_dict, fp)
with open('saved_grads/{}/val.pkl'.format(DATASET), 'wb') as fp:
    pickle.dump(val_grad_dict, fp)


#Run Outlier Gradient Analysis and Influence-Based Baselines
influence_engine = IFEngineGeneration()
influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict)
#influence_engine.preprocess_gradients(tr_grad_dict, tr_grad_dict)
influence_engine.compute_hvps()
influence_engine.compute_IF()


identity_df=influence_engine.IF_dict['identity']
proposed_df=influence_engine.IF_dict['proposed']

print(identity_df, proposed_df)

n_train, n_val = 900, 100
n_sample_per_class = 90 
n_class = 10

identity_auc_list, proposed_auc_list=[], []
for i in range(n_val):
    gt_array=np.zeros(n_train)
    gt_array[(i//n_class)*n_sample_per_class:((i//n_class)+1)*n_sample_per_class]=1
    
    identity_auc_list.append(roc_auc_score(gt_array, (identity_df.iloc[i,:].to_numpy())))
    proposed_auc_list.append(roc_auc_score(gt_array, (proposed_df.iloc[i,:].to_numpy())))
    
print(f'identity AUC: {np.mean(identity_auc_list):.3f}/{np.std(identity_auc_list):.3f}')
print(f'proposed AUC: {np.mean(proposed_auc_list):.3f}/{np.std(proposed_auc_list):.3f}')


identity_recall_list, proposed_recall_list=[], []
for i in range(n_val):
    correct_label = i // 10
    sorted_labels = np.argsort(np.abs(identity_df.iloc[i].values))[::-1] // 90
    recall_identity = np.count_nonzero(sorted_labels[0:90] == correct_label) / 90.0
    identity_recall_list.append(recall_identity)
    
    sorted_labels = np.argsort(np.abs(proposed_df.iloc[i].values))[::-1] // 90
    recall_proposed = np.count_nonzero(sorted_labels[0:90] == correct_label) / 90.0
    proposed_recall_list.append(recall_proposed)
    
print(f'identity Recall: {np.mean(identity_recall_list):.3f}/{np.std(identity_recall_list):.3f}')
print(f'proposed Recall: {np.mean(proposed_recall_list):.3f}/{np.std(proposed_recall_list):.3f}')
