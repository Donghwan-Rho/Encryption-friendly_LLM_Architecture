"""Script to evaluate a pretrained model."""

import os  

import torch
import torch.nn as nn
import hydra


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

log = logging.getLogger(__name__)

def convert_to_serializable(outputs):
    return {
        'logits': outputs['logits'].detach().cpu().tolist(),
        'loss': outputs['loss'].item()  # loss 값은 스칼라이므로 item()을 사용해 Python 숫자로 변환
    }
    
def main_downstream_process(cfg, setup):
    # torch.set_default_dtype(torch.float64)
    # log.info(f'torch.set_default_dtype(torch.float64)')
    # print(f'cfg: {cfg}')
    
    """This function controls the central routine."""
    local_time = time.time()
    # cfg.name = "layers2_12hrs_layernorm_relu"
    # cfg.name = "10lys_ln_relu_matX_var100_ratioX_249900_250000_penaltycoerr100"
    local_checkpoint_folder = os.path.join(cfg.base_dir, cfg.name, "checkpoints")
    all_checkpoints = [f for f in os.listdir(local_checkpoint_folder)]
    checkpoint_paths = [os.path.join(local_checkpoint_folder, c) for c in all_checkpoints]
    checkpoint_name = checkpoint_paths[cfg.eval.ckpt_num]
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name)
    with open(os.path.join(checkpoint_name, "model_config.json"), "r") as file:
        cfg_arch = OmegaConf.create(json.load(file))  # Could have done pure hydra here, but wanted interop
        
    cfg_arch.norm = cfg_arch.eval_norm
    cfg_arch.nonlin = cfg_arch.eval_nonlin
    tasks, train_dataset, eval_datasets = prepare_task_dataloaders_modified(tokenizer, cfg.eval, cfg.impl)
    
    metrics = dict()
    stats = defaultdict(list)
    
    for task_name, task in tasks.items():
        print(colored('task_name: {}'.format(task_name), 'red'))
        print(colored('task: {}'.format(task), 'red'))
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        log.info(f"Sentense test eval with padding size 128")
        log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        
        cfg_arch.task_name = task_name
        
        # 여기서 제가 드린 safetensor 파일을 불러와주세요.
        # load_file에서 safetensors 파일을 불러올 경로를 수정해야 합니다.
        #############################
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        current_path = os.getcwd()
        # print(f'current_path: {current_path}')
        model_state_dict_loaded = load_file(os.path.join(current_path, '..', '..', '..', '..', '..', 'safetensors', f'{cfg.name}_eval_{task_name}_128padding.safetensors'))
        model.load_state_dict(model_state_dict_loaded)
        
        # for name, param in model.named_parameters():
        #     log.info(f'{name}: {param.dtype}')
        #############################
        # log.info('Task model: {}'.format(model))
        
        model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        model.eval()
        eval_dataset = eval_datasets[f'{task_name}']
        
        # # eval_dataset에서 데이터를 하나씩 뽑아서 모델의 output 값을 얻음
        # # 128 padding
        # dict_list = []
        # for i in range(len(eval_dataset)):
        #     input_ids = eval_dataset[i]["input_ids"]
        #     decoded_sentence = tokenizer.decode(input_ids)
        #     print(f'decoded_sentence: {decoded_sentence}')        
        #     encoded_inputs = tokenizer([decoded_sentence], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        #     attention_mask = eval_dataset[i]["attention_mask"].unsqueeze(0).to('cuda')
        #     labels = eval_dataset[i]["labels"].unsqueeze(0).to('cuda')
        #     data = {'input_ids': encoded_inputs["input_ids"].to('cuda'), 'attention_mask': encoded_inputs["attention_mask"].to('cuda'), 'labels': labels}
        #     outputs = model(**data)
        #     print(f'{i} outputs: {outputs}')
        #     dict_list.append(convert_to_serializable(outputs))
        # # model의 output 정보를 json 파일로 저장
        # with open (f'{cfg.name}_{task_name}_128padding_zeroindex.json', 'w') as f:
        #     json.dump(dict_list, f, indent=4)  
        
        # eval_dataset에서 원하는 index의 data를 뽑아서 model의 output을 얻음
        ######################################################
        idx = 4
        
        # # padding 없는 버전
        # input_ids = eval_dataset[idx]["input_ids"].unsqueeze(0).to('cuda')
        # attention_mask = eval_dataset[idx]["attention_mask"].unsqueeze(0).to('cuda')
        # labels = eval_dataset[idx]["labels"].unsqueeze(0).to('cuda')
        # # Model의 input data 구성
        # data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        # print(f'data: {data}')
        
        # padding 있는 버전
        # input_ids = eval_dataset[idx]["input_ids"].unsqueeze(0).to('cuda')
        # attention_mask = eval_dataset[idx]["attention_mask"].unsqueeze(0).to('cuda')
        labels = eval_dataset[idx]["labels"].unsqueeze(0).to('cuda')
        # data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        decoded_sentence = tokenizer.decode(eval_dataset[idx]["input_ids"])
        print(f'decoded_sentence: {decoded_sentence}')        
        encoded_inputs = tokenizer([decoded_sentence], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        print(colored(f'encoded_inputs["input_ids"]: {encoded_inputs["input_ids"]}', 'green'))
        print(colored(f'encoded_inputs["attention_mask"]: {encoded_inputs["attention_mask"]}', 'green'))
        data_modified = {'input_ids': encoded_inputs["input_ids"].to('cuda'), 'attention_mask': encoded_inputs["attention_mask"].to('cuda'), 'labels': labels}
        print(colored(f'data_modified: {data_modified}', 'green'))
        
        # 여러 번 output을 구해도 일정한 값이 나옴을 확인할 수 있음
        # 모델의 중간 과정의 tensor들을 출력하고 싶다면 crammed-bert_modified.py, components.py, attention_modified.py에서 print하시면 됩니다.
        # outputs = model(**data)
        # print(f'outputs: {outputs}')
        # outputs = model(**data)
        # print(f'outputs: {outputs}')
        
        if not cfg_arch.get_grad:
            outputs_modified = model(**data_modified)
            print(f'outputs_modified: {outputs_modified}')
        else:
            outputs_modified, before_zero_indexing_hidden_states, first_token_tensor = model(**data_modified)
            print(f'outputs_modified: {outputs_modified}')
            outputs_modified['loss'].backward()
            print(f'before_zero_indexing_hidden_states.grad: {before_zero_indexing_hidden_states.grad}')
            print(f'before_zero_indexing_hidden_states.grad[0][0]: {before_zero_indexing_hidden_states.grad[0][0][:10]}')
            print(f'first_token_tensor.grad: {first_token_tensor.grad[0][:10]}')    
        
        # outputs_modified = model(**data_modified)
        # print(f'outputs_modified: {outputs_modified}')
        ######################################################        

    # Save to summary:
    if cramming.utils.is_main_process():
        cramming.utils.save_summary("downstream", cfg, stats, time.time() - local_time, setup)
    return metrics  # will be dumped into yaml

@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")

if __name__ == "__main__":
    launch()
