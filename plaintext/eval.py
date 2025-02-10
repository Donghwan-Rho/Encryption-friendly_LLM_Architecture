"""Script to evaluate a pretrained model."""

import os  

import torch
import hydra


import time
import datetime
import logging
from collections import defaultdict

import cramming
import evaluate
from termcolor import colored
from safetensors.torch import load_file, save_file
from omegaconf import OmegaConf

log = logging.getLogger(__name__)
'''
python eval.py eval=GLUE_sane name=dhrho eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True impl.compile_torch=False eval.ckpt_num=8
'''
def main_downstream_process(cfg, setup):
    # torch.set_default_dtype(torch.float64)
    # log.info(f'torch.set_default_dtype(torch.float64)')
    """This function controls the central routine."""
    local_time = time.time()

    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    # cfg_arch.tasks = tasks
    cfg_arch.attention.is_train = False
    cfg_arch.embedding.is_train = False
    cfg_arch.is_train = False
    cfg_arch.norm = cfg_arch.eval_norm
    cfg_arch.nonlin = cfg_arch.eval_nonlin
    log.info(colored('cfg_arch: {}'.format(cfg_arch), 'green'))
    tasks = cramming.prepare_task_dataloaders(tokenizer, cfg.eval, cfg.impl)
    
    metrics = dict()
    stats = defaultdict(list)
    
    # Start the clocks now:
    for task_name, task in tasks.items():
        print(colored('task_name: {}'.format(task_name), 'red'))
        print(colored('task: {}'.format(task), 'red'))
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        # Prepare model for finetuning:
        
        cfg_arch.task_name = task_name
        
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
            
        OmegaConf.set_struct(cfg, False)  # cfg 구조 변경 허용
        cfg.impl.experiment_float64 = True

        for name, param in model.named_parameters():
            log.info(f'{name}: {param.dtype}')
        # # encoder 파라미터 freeze
        # for param in model.encoder.parameters():
        #     param.requires_grad = False
        # # freeze되지 않은 파라미터 확인 및 출력
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         log.info(f"Trainable parameter: {name}")  # freeze되지 않은 파라미터 출력
        #     else:
        #         log.info(f"Untrainable parameter: {name}")  # freeze된 파라미터 출력
        
        log.info('Task Model: {}'.format(model))
        model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        model_engine.load_checkpoint(cfg_arch, model_file)
        print(f'model.encoder.layers[0].norm1.weight[:10]: {model.encoder.layers[0].norm1.weight[:10]}')
        print(f'model.head.weight[:10]: {model.head.weight[0][:10]}')

        try:
            assert task_name != "record"
            # print('assert task_name != "record"')
            metric = evaluate.load(task["details"]["collection"], task_name, cache_dir=cfg.impl.path)
        except (FileNotFoundError, AssertionError):  # no specific metric downloadable from evaluate, construct directly
            print('(FileNotFoundError, AssertionError)')
            targets = [evaluate.load(metric_name, cache_dir=cfg.impl.path) for metric_name in task["details"]["target_metrics"]]
            metric = evaluate.CombinedEvaluations(targets)
        # Launch training
        model_engine.train()
        loss_vals = []
        self_count = 0
        for epoch in range(cfg.eval.epochs):
            print(colored('epoch: {}'.format(epoch), 'yellow'))
            train_time = time.time()

            for step, batch in enumerate(task["trainloader"]):
                # batch: int64
                # print(f"batch['input_ids']: {batch['input_ids'].dtype}")
                # print(f"batch['token_type_ids']: {batch['token_type_ids'].dtype}")
                # print(f"batch['attention_mask']: {batch['attention_mask'].dtype}")
                # print(f"batch['labels']: {batch['labels'].dtype}")
                self_count += 1
                # Heavy lifting is moved to engines
                device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
                # print(colored('loss = model_engine.step(device_batch)', 'yellow'))
                loss = model_engine.step(device_batch)
                loss_vals.append(loss.detach())
                if cfg.dryrun:
                    break
                # print(model.encoder.embedding.norm.weight)
                print(f'model_engine.optimizer: {model_engine.optimizer}')
                print(f'model_engine.scheduler: {model_engine.scheduler}')
                print(f"optimizer.param_groups[0]['lr']: {optimizer.param_groups[0]['lr']}")
               
            metrics[task_name] = validate(model_engine, task["validloader"], metric, setup, cfg)
            stats[f"{task_name}_epoch"] += [epoch]
            stats[f"{task_name}_loss"] += [loss.item()]

            stats[f"{task_name}_avg_loss"] += [torch.stack(loss_vals).mean().item()]  # Smoothed loss
            loss_vals = []
            current_lr = model_engine.optimizer.param_groups[0]["lr"]

            log_msg = f"Train loss {loss.item():2.4f} at step {step} with lr {current_lr:.5f}. "
            log_msg += f"[Avg: {stats[f'{task_name}_avg_loss'][-1]:2.4f}] after epoch {epoch}."

            stats[f"{task_name}_train_time"] += [(time.time() - train_time)]
            estimated_train_finish = str(datetime.timedelta(seconds=stats[f"{task_name}_train_time"][-1] * cfg.eval.epochs))
            tokens_per_second = (step + 1) * cfg.eval.max_seq_length * cfg.impl.microbatch_size / stats[f"{task_name}_train_time"][-1]
            log_msg += (
                f" Perf: {stats[f'{task_name}_train_time'][-1]/60:2.4f}min per epoch ({tokens_per_second:.0f}t/s). "
                f"Estimated Total Train: {estimated_train_finish}."
            )

            for name, metric_val in metrics[task_name].items():
                stats[f"{task_name}_{name}"] += [metric_val]
            log.info(log_msg)
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in metrics[task_name].items()])
            log.info(f"Validation metric is {msg_metrics} after epoch {epoch}.")
            cramming.utils.wandb_log(stats, cfg)

            if cfg.dryrun:
                break       
            
        # # Ended Fine-Tuning
        # model_state_dict = model.state_dict()
        # save_file(model_state_dict, 'layers2_12hrs_layernorm_relu_eval_rte.safetensors')
        # model_1 = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        # print(colored(f'model.encoder.embedding.norm.weight[:10]: {model.encoder.embedding.norm.weight[:10]}', 'yellow'))
        # print(colored(f'model_1.encoder.embedding.norm.weight[:10]: {model_1.encoder.embedding.norm.weight[:10]}', 'red'))
        # model_state_dict_loaded = load_file('layers2_12hrs_layernorm_relu_eval_rte.safetensors')
        # model_1.load_state_dict(model_state_dict_loaded)
        # print(colored(f'model_1.encoder.embedding.norm.weight[:10]: {model_1.encoder.embedding.norm.weight[:10]}', 'green'))
        # # for name, param in model.named_parameters():
        # #     print(f'{name}: {param.shape}')
      
        # Launch extra testing if extra validation set exists (as with MNLI-mismatched):
        if task["extra_validloader"] is not None:
            extra_eval_metric = validate(model_engine, task["extra_validloader"], metric, setup, cfg)
            # metrics[task_name + "extra"] = extra_eval_metric
            metrics[task_name].update({f"{k}_extra": v for k, v in extra_eval_metric.items()})
            for name, metric_val in extra_eval_metric.items():
                stats[f"{task_name}_{name}_extra"] += [metric_val]
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in extra_eval_metric.items()])
            log.info(f"Extra validation metric is {msg_metrics} after finetuning.")
            cramming.utils.wandb_log({f"{task_name}_{k}_extra": [v] for k, v in extra_eval_metric.items()}, cfg)

    # Check average metric over all tasks:
    target_metrics = []
    for task_name, task in tasks.items():
        target_metric_names = task["details"]["target_metrics"]
        for metric_name in target_metric_names:
            target_metrics.append(metrics[task_name][metric_name])
    metrics[f"{cfg.eval.name}_amean"] = torch.as_tensor(target_metrics).mean().item()
    metrics[f"{cfg.eval.name}_hmean"] = torch.as_tensor(target_metrics).pow(-1).mean().pow(-1).item()
    log.info(f"Overall average metric on evaluation {cfg.eval.name} is {metrics[f'{cfg.eval.name}_amean']:.2f}.")
    cramming.utils.wandb_log(
        {f"{cfg.eval.name}_amean": [metrics[f"{cfg.eval.name}_amean"]], f"{cfg.eval.name}_hmean": [metrics[f"{cfg.eval.name}_hmean"]]},
        cfg,
    )

    # Save to summary:
    if cramming.utils.is_main_process():
        cramming.utils.save_summary("downstream", cfg, stats, time.time() - local_time, setup)
    return metrics  # will be dumped into yaml


@torch.no_grad()
def validate(model_engine, validloader, metric, setup, cfg):
    """Evaluate on validation set."""
    model_engine.eval()
    for step, batch in enumerate(validloader):
        device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
        _, predictions = model_engine.forward_inference(**device_batch)

        if getattr(metric, "config_name", "") != "multirc":
            metric.add_batch(predictions=predictions, references=device_batch["labels"])
        else:  # uuuuuughhhhh, whhyyy multirc
            pred_indices = range(step * predictions.shape[0], (step + 1) * predictions.shape[0])
            packages = [dict(idx=validloader.index_lookup[pred_indices[i]], prediction=p) for i, p in enumerate(predictions.cpu())]
            metric.add_batch(predictions=packages, references=batch["labels"])

        if cfg.dryrun and step > 1:
            break

    try:
        eval_metric = metric.compute()
    except ValueError:  # pearson corr computation will raise errors if metric values are NaN
        log.info("Value Error in metrics computation, maybe non-finite values in prediction. Returning backup score.")
        eval_metric = metric.compute(predictions=[0, 1], references=[1, 0])  # spoof terrible result if metric computation fails
    model_engine.train()
    return {k: float(v) for k, v in eval_metric.items()}  # force float returns


@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")


if __name__ == "__main__":
    launch()
