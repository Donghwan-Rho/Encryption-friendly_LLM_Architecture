"""Script to evaluate a pretrained model."""

import os  

import torch
import torch.nn as nn
import hydra
import random


import time
import datetime
import logging
from collections import defaultdict
import transformers
from omegaconf import OmegaConf, open_dict

import cramming
import evaluate
from termcolor import colored
from safetensors.torch import load_file, save_file
from cramming.data.downstream_task_preparation import prepare_task_dataloaders_modified
import json
from cramming.architectures.crammed_bert_modified import crammedBertConfig
from cramming.architectures.architectures_grad import ScriptableLMForPreTraining_grad
import argparse

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################################
### After getting pre-trained weight, we need to go another generation, ###
### since our input for HE model is embedded data.                      ###
### This process is excuted the below `main_downstream_process_modifie` ###
###########################################################################

def main_downstream_process_modified(cfg, setup):
    
    cfg.impl.microbatch_size = 1
    cfg.eval.batch_size = 1
    
    print(f'User name: {cfg.eval.user_name}')
    local_checkpoint_folder = os.path.join(cfg.base_dir, cfg.eval.user_name, "checkpoints")
    all_checkpoints = [f for f in os.listdir(local_checkpoint_folder)]
    checkpoint_paths = [os.path.join(local_checkpoint_folder, c) for c in all_checkpoints]
    checkpoint_name = checkpoint_paths[cfg.eval.ckpt_num]
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name)
    with open(os.path.join(checkpoint_name, "model_config.json"), "r") as file:
        cfg_arch = OmegaConf.create(json.load(file))  # Could have done pure hydra here, but wanted interop
    cfg_arch.architectures = ["ScriptableCrammedBERT-grad"]
    cfg_arch.poolinglora = True
    
    tasks, train_dataset, eval_datasets = prepare_task_dataloaders_modified(tokenizer, cfg.eval, cfg.impl)
    
    # Change the base directory for loading weights and saving fine-tuning data.
    os.chdir("../../../../../")

    ################################
    ## Train ##      |  ## Eval ## # 
    # RTE: 2490      |     277     #
    # MRPC: 3668     |     408     #  
    # COLA: 8551     |    1043     #
    # STS-B: 5749    |    1500     # 
    # SST-2: 67349   |     872     #     
    # QNLI: 104743   |    5463     #
    ################################
                        
    for task_name, task in tasks.items():
        os.makedirs(f'./fine-tuning_data', exist_ok=True)
        
        cfg_arch.task_name = task_name
        
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        
        ## Put the pre-trained weights path ##
        model_state_dict_loaded = load_file(f"./pre-trained_weights/{cfg.eval.user_name}/model.safetensors")
        model.load_state_dict(model_state_dict_loaded, strict=False)
        
        model = model.to(device)
        model.eval()
        
        ########################################################
        ### We require train/eval labels to use in HE model. ###
        ### Here, we save in the first index.                ### 
        ### For example, for mrpc eval dataset labels,       ###
        ### we save this with `labels_mrpc_eval.pth`         ###
        ########################################################     
        
        #################################################
        ## 1. Set a target dataset in GLUE_sane.yaml   ##
        ## 2. If we want to generate eval dataset,     ##
        ##    then, call eval_datasets[f'{task_name}'] ##
        ##    Otherwise, we call train_datset.         ##
        ## 3. If we want to generate train dataset,    ##
        ##    then, set eval_dataset = train_dataset   ##
        #################################################
        
        os.makedirs(os.path.join(f'./fine-tuning_data', f'{task_name}_train_inputs'), exist_ok=True)
        os.makedirs(os.path.join(f'./fine-tuning_data', f'{task_name}_train_masks'), exist_ok=True)
        
        if cfg.eval.save_train_data:
            saving_dataset = train_dataset
            print(f'Save train data.')
        else:
            saving_dataset = eval_datasets[f'{task_name}']
            print(f'Save evaluation data.')
            
        data_set_len = len(saving_dataset)
        print(f'The length of the dataset: {data_set_len}')
        
        labels_ = saving_dataset[:]['labels'].cpu()
        
        if cfg.eval.save_train_data:
            torch.save(labels_, f"./fine-tuning_data/labels_{task_name}_train.pth")
        else:
            torch.save(labels_, f"./fine-tuning_data/labels_{task_name}_eval.pth")

        for i in range(data_set_len):
            idx = i
            labels = saving_dataset[idx]["labels"].unsqueeze(0).to('cuda')
            decoded_sentence = tokenizer.decode(saving_dataset[idx]["input_ids"])
            encoded_inputs = tokenizer([decoded_sentence], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            data_modified = {'input_ids': encoded_inputs["input_ids"].to('cuda'), 'attention_mask': encoded_inputs["attention_mask"].to('cuda'), 'labels': labels}
            
            outputs, emb_outputs, attentions_outputs, tf_outputs = model(**data_modified)
        
        # model.encoder.save = 'train'
        # model.encoder.number = 0
        # os.makedirs(os.path.join(f'./fine-tuning_data', task_name, 'train_labels'), exist_ok=True)
        # os.makedirs(os.path.join(f'./fine-tuning_data', task_name, 'train_inputs'), exist_ok=True)
        # os.makedirs(os.path.join(f'./fine-tuning_data', task_name, 'train_masks'), exist_ok=True)
        # data_set_len = len(train_dataset)
        # print(f'The length of train dataset: {data_set_len}')
        
        # labels_ = train_dataset[:]['labels'].cpu()
        # torch.save(labels_, f"./fine-tuning_data/{task_name}/train_labels/labels.pth")

        # for j in range(data_set_len):
        #     idx = j
        #     labels = train_dataset[idx]["labels"].unsqueeze(0).to('cuda')
        #     decoded_sentence = tokenizer.decode(train_dataset[idx]["input_ids"])
        #     encoded_inputs = tokenizer([decoded_sentence], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        #     data_modified = {'input_ids': encoded_inputs["input_ids"].to('cuda'), 'attention_mask': encoded_inputs["attention_mask"].to('cuda'), 'labels': labels}
            
        #     outputs, emb_outputs, attentions_outputs, tf_outputs = model(**data_modified)


@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher_modified(cfg, main_downstream_process_modified, job_name="downstream finetuning")

if __name__ == "__main__":
    launch()

