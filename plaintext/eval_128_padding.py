"""Script to evaluate a pretrained model."""

import os  

import torch
import hydra


import time
import datetime
import logging
from collections import defaultdict
from omegaconf import OmegaConf
import numpy as np
from matplotlib.ticker import FuncFormatter
from contextlib import nullcontext

import cramming
import evaluate
from termcolor import colored
from safetensors.torch import load_file, save_file
from cramming.data.downstream_task_preparation import prepare_task_dataloaders_modified

log = logging.getLogger(__name__)

def power_formatter(y, pos):
    if y == 0:
        return "0"
    exponent = np.log2(abs(y))
    sign = '-' if y < 0 else ''
    return f"{sign}2^{int(exponent)}"

def main_downstream_process(cfg, setup):
    """This function controls the central routine."""
    local_time = time.time()
    
    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    
    cfg_arch.attention.is_train = False
    cfg_arch.embedding.is_train = False
    cfg_arch.is_train = False
    cfg_arch.norm = cfg_arch.eval_norm
    cfg_arch.nonlin = cfg_arch.eval_nonlin
    
    OmegaConf.set_struct(cfg, False)
    
    cfg.impl.experiment_float64 = cfg_arch.experiment_float64
    if cfg_arch.experiment_float64:
        torch.set_default_dtype(torch.float64)
        log.info(colored(f'Default dtype: torch.float64', 'yellow'))
        cfg_arch.attention.seq_op_in_fp32 = False
    else:
        cfg_arch.attention.seq_op_in_fp32 = True
        log.info(colored(f'Default dtype: torch.float32', 'yellow'))
    
    tasks, train_dataset, eval_datasets = prepare_task_dataloaders_modified(tokenizer, cfg.eval, cfg.impl)
    
    metrics = dict()
    stats = defaultdict(list)
    
    # Start the clocks now:
    for task_name, task in tasks.items():        
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        log.info(f"Padding size: 128")
        
        # Prepare model for finetuning:
        cfg_arch.task_name = task_name
        
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        if cfg_arch.experiment_float64:
            model = model.to(dtype=torch.float64)
            for param in model.parameters():
                param.data = param.data.to(dtype=torch.float64)
                
        model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        model_engine.load_checkpoint(cfg_arch, model_file)
        model_engine.task_name = task_name
        log.info('Model:\n{}'.format(model))
        
        # Checking whether each parameter is freezed
        trainable_param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                pass
            
        total_params = sum(p.numel() for p in model.parameters()) # Total # of params of The Model
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Total # of trainable params of The Model
        log.info(f'Total number of parameters of the model: {total_params}')
        log.info(f'Total number of trainable parameters of the model: {total_trainable_params}')
        log.info(f'Trainable parameters are {total_trainable_params/total_params*100}%.')
        
        model_engine.optimizer.noise = cfg.eval.grad_noise
        if model_engine.optimizer.noise:
            model_engine.optimizer.noise_scale = 2**(-cfg.eval.grad_noise_scale)
            log.info(colored(f'Add noise to gradient with Mean 0, Std_dev {model_engine.optimizer.noise_scale}.', 'green'))
        else:
            log.info(colored(f'Not add noise to the gradients.', 'green'))
        model_engine.optimizer.task_name = task_name

        try:
            assert task_name != "record"
            metric = evaluate.load(task["details"]["collection"], task_name, cache_dir=cfg.impl.path)
            if task_name == 'rte':
                metric = evaluate.load('glue', 'mrpc', cache_dir=cfg.impl.path)
        except (FileNotFoundError, AssertionError):  # no specific metric downloadable from evaluate, construct directly
            print('(FileNotFoundError, AssertionError)')
            targets = [evaluate.load(metric_name, cache_dir=cfg.impl.path) for metric_name in task["details"]["target_metrics"]]
            metric = evaluate.CombinedEvaluations(targets)
        # Launch training
        model_engine.train()
        
        # Whether activating dropout
        if cfg.eval.dropout_on == True:
            log.info(colored('Dropout On', 'yellow'))
        else:
            model_engine.eval()
            log.info(colored('Dropout Off', 'yellow'))
        
        loss_vals = []
        self_count = 0
        for epoch in range(cfg.eval.epochs):
            print(colored('epoch: {}'.format(epoch), 'yellow'))
            train_time = time.time()

            for step, batch in enumerate(task["trainloader"]):
                self_count += 1
                
                device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
                loss = model_engine.step(device_batch)
                
                model_engine.accumulated_samples += model_engine.effective_mbs
                context = model_engine.model.no_sync if model_engine.accumulated_samples < model_engine.current_batch_size else nullcontext
                with context():
                    loss = model_engine.forward(**device_batch)["loss"]
                    model_engine.backward(loss)
                    model_engine.steps += 1
                    if model_engine.accumulated_samples >= model_engine.current_batch_size:
                        model_engine.accumulated_samples = 0
                        if model_engine.cfg_train.gradient_clipping is not None:
                            model_engine.scaler.unscale_(model_engine.optimizer)
                            torch.nn.utils.clip_grad_norm_(model_engine.model.parameters(), model_engine.cfg_train.gradient_clipping, norm_type=2.0)
                        model_engine.scaler.step(model_engine.optimizer)
                        model_engine.scaler.update()
                        if step == 0:
                            model_engine.optimizer.graph_gradients(task_name, trainable_param_names, epoch)
                            
                        model_engine.optimizer.zero_grad()
                        model_engine.schedule_batch_size()
                    model_engine.scheduler.step()
                    
                loss_vals.append(loss.detach())
                if cfg.dryrun:
                    break
            
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
