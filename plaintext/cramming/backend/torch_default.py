"""Basic training backend engine for pytorch training with all bells and whistles.

Interface set up to be compliant with the deepspeed engine interface.


There are two versions here, the TorchEngineMinimal, which is the default, and TorchEngineFull which contains a few training variations
that were tested but ultimately discarded, so read that part only if you're interested.

"""
import torch
import torch._inductor.utils

import os
import json
from omegaconf import OmegaConf
from functools import partial
from contextlib import nullcontext
import time
import matplotlib.pyplot as plt
import numpy as np
import math

import logging

import transformers
from safetensors.torch import load_file, save_file
from transformers.utils.generic import working_or_temp_dir

# from torch.distributed.optim import ZeroRedundancyOptimizer

from .utils import group_parameters, prepare_pretraining_dataloader, update_ema, updated_latest_weight_average
from .optimizers.schedulers import get_schedule_fn
from .optimizers import Adahessian, AdamWScale, Shampoo, LARS, SAM, ProgressiveBatching, AGD, Sophia
from termcolor import colored

log = logging.getLogger(__name__)
_default_setup = dict(device=torch.device("cpu"), dtype=torch.float)

import warnings
from termcolor import colored

warnings.filterwarnings("ignore", "Detected call of ", UserWarning)  # schedulers are deliberately used differently

def chebishev(a, b, x, d): 
    y = (2*x-(a+b))/(b-a)
    l = [1, y] # order 0
    for i in range(d-1):
        l.append(2*y*l[len(l)-1]-l[len(l)-2])
    return l  

def evalcheb(coeffs, x, a, b):
    d = len(coeffs) - 1
    l = chebishev(a, b, x, d)
    return sum(coeffs[i]*l[i] for i in range(len(coeffs)))

def polyeval_torch(coeffs, x):
    return sum(coeffs[i]*(x**i) for i in range(len(coeffs)))

def Inverse_sqrt_approx(x):
    coeffs = torch.tensor([ 3.9359587065628147684037685394287109375, -5.3255082193645648658275604248046875, 4.4768915243548690341413021087646484375, -3.96794509016399388201534748077392578125, 
    3.60465136755010462366044521331787109375, -3.322337739053182303905487060546875, 3.0916074502383708022534847259521484375, -2.896631363124470226466655731201171875, 
    2.72791187408074620179831981658935546875, -2.5793033506706706248223781585693359375, 2.44660075974752544425427913665771484375, -2.32680037745376466773450374603271484375, 
    2.217681929774698801338672637939453125, -2.117557876961654983460903167724609375, 2.02511555564342415891587734222412109375, -1.939313754808608791790902614593505859375, 
    1.859312654458335600793361663818359375, -1.784424995074004982598125934600830078125, 1.714081198746498557738959789276123046875, -1.64780391713065910153090953826904296875, 
    1.58518910566999693401157855987548828125, -1.525891713941746274940669536590576171875, 1.46961470393944182433187961578369140625, -1.416100508829913451336324214935302734375,
    1.365124309084421838633716106414794921875, -1.316488680937254684977233409881591796875, 1.27001929425387061201035976409912109375, -1.225561422210375894792377948760986328125, 
    1.18297708568934467621147632598876953125, -1.142142698741736239753663539886474609375, 1.1029471132133039645850658416748046875, -1.065289984026094316504895687103271484375, 
    1.0290803940370096825063228607177734375, -0.994235690528512350283563137054443359375, 0.9606804954719336819835007190704345703125, -0.928345859307228238321840763092041015625, 
    0.897168534005686524324119091033935546875, -0.867090345798715134151279926300048828125, 0.838057651637427625246345996856689453125, -0.81002086631997372023761272430419921875, 
    0.7829340495736687444150447845458984375, -0.756754544194336631335318088531494140625, 0.7314426578695929492823779582977294921875, -0.7069613825196938705630600452423095703125, 
    0.6832761459818357252515852451324462890625, -0.66035459167324006557464599609375, 0.6381663825550276669673621654510498046875, -0.61668302624821080826222896575927734375, 
    0.59587771864244132302701473236083984375, -0.5757252037028592894785106182098388671875, 0.5562016475150812766514718532562255859375, -0.5372845248775774962268769741058349609375, 
    0.5189525169762418954633176326751708984375, -0.501185418879686039872467517852783203125, 0.48396405575249445973895490169525146484375, -0.46727020682146758190356194972991943359375, 
    0.45108653626766681554727256298065185546875, -0.43539653029665714711882174015045166015625, 0.42018443975166519521735608577728271484375, -0.4054352276943973265588283538818359375, 
    0.39113452145875271526165306568145751953125, -0.37726856872495773131959140300750732421875, 0.36382419723258863086812198162078857421875, -0.35078877777186789899133145809173583984375, 
    0.33815019015173675143159925937652587890625, -0.32589679186048670089803636074066162109375, 0.31401738917838883935473859310150146484375, -0.30250121051676615024916827678680419921875, 
    0.291337881781146279536187648773193359375, -0.28051740359296672977507114410400390625, 0.270030130193845252506434917449951171875, -0.25986674990053870715200901031494140625, 
    0.2500182669709829497151076793670654296875, -0.24047598477091014501638710498809814453125, 0.231231490130767269874922931194305419921875, -0.22227663879448300576768815517425537109375, 
    0.213603541877319003106094896793365478515625, -0.205204553248449883540160953998565673828125, 0.19707225776937775663100183010101318359375, -0.18919946031883227988146245479583740234375, 
    0.181579175545493853860534727573394775390625, -0.17420461829487976501695811748504638671875, 0.167069194655823594075627624988555908203125, -0.16016649358562062843702733516693115234375, 
    0.1534902790663181804120540618896484375, -0.1470344827584995073266327381134033203125, 0.140793197111406698240898549556732177734375, -0.134760668902572433580644428730010986328125, 
    0.128931293171262950636446475982666015625, -0.123299607521175857982598245143890380859375, 0.1178602867661311393021605908870697021484375, -0.1126081378944263633457012474536895751953125,
    0.107538095330937721882946789264678955078125, -0.10264521647468427545391023159027099609375, 9.7924677496848744340240955352783203125e-2, -9.33717693768585377256385982036590576171875e-2, 
    8.89818941658404582994990050792694091796875e-2, -8.47505614573265120270662009716033935546875e-2, 8.06733850557748155551962554454803466796875e-2, -7.6746079827216817648150026798248291015625e-2, 
    7.2964458720207403530366718769073486328125e-2, -6.932442994775556144304573535919189453125e-2, 6.5821994318184806616045534610748291015625e-2, -6.245324270503260777331888675689697265625e-2, 
    5.921435364842864146339707076549530029296875e-2, -5.610159107845902326516807079315185546875e-2, 5.311130215255843722843565046787261962890625e-2, -5.023991520005210986710153520107269287109375e-2,
    4.748393776691273160395212471485137939453125e-2, -4.48399547540248022414743900299072265625e-2, 4.230462664310152831603772938251495361328125e-2, -3.98746878046267738682217895984649658203125e-2, 
    3.7546944882478783256374299526214599609375e-2, -3.531827525063135908567346632480621337890625e-2, 3.318562553664605729863978922367095947265625e-2, -3.114601020826057720114476978778839111328125e-2,
    2.9196510218554294624482281506061553955078125e-2, -2.733427170636559822014532983303070068359375e-2, 2.5556504747754615891608409583568572998046875e-2, -2.386048215572600383893586695194244384765625e-2, 
    2.224353832474434966570697724819183349609375e-2, -2.0703068117171596895786933600902557373046875e-2, 1.923652578892642850405536592006683349609375e-2, -1.7841423951580281936912797391414642333984375e-2, 
    1.6515332568673102286993525922298431396484375e-2, -1.525587798374772319220937788486480712890625e-2, 1.40607419782412534914328716695308685302734375e-2, -8.238728574133347137831151485443115234375e-2])
    
    res = evalcheb(coeffs, x, 0.0, 1.0)
    res = 0.5*(res*(3-x*res*res))
    res = 0.5*(res*(3-x*res*res))
    res = 0.5*(res*(3-x*res*res))
    return res

def initialize_torch(model, dataset, tokenizer, cfg_train, cfg_impl, elapsed_time, setup=_default_setup):
    """initialize a torch engine."""
    if dataset is not None:
        # print(colored('dataloader = prepare_pretraining_dataloader', 'magenta'))
        dataloader = prepare_pretraining_dataloader(dataset, tokenizer, cfg_train, cfg_impl)
    else:
        dataloader = None

    # in most cases we can use a simpler Engine class:
    require_full_engine = "sequence_curriculum" in cfg_train or "weight_averaging" in cfg_train or "gradinit" in cfg_train

    if require_full_engine:
        model_engine = TorchEngineFull(model, cfg_train, cfg_impl, elapsed_time, setup=setup, seq_length=tokenizer.model_max_length)
    else:
        model_engine = TorchEngineMinimal(model, cfg_train, cfg_impl, elapsed_time, setup=setup, seq_length=tokenizer.model_max_length)
    model_engine.train()  # This is the default engine state. Pretraining scripts may change this.
    return model_engine, model_engine.optimizer, model_engine.scheduler, dataloader


class TorchEngineMinimal(torch.nn.Module):
    """This class mirrors deepspeed functionality. Not all changes are implemented in this version.

    See TorchEngineFull for more modifications.
    """

    def __init__(self, model, cfg_train, cfg_impl, already_elapsed_time=0.0, setup=_default_setup, seq_length=128):
        """Load Engine. The model will be compiled by default."""
        super().__init__()
        # 여기서 model은 아직 ScriptableLMForPreTraining
        if cfg_impl.experiment_float64:
            setup['dtype'] = torch.float64
    
        self.cfg_train = cfg_train
        self.cfg_impl = cfg_impl
        if self.cfg_impl.microbatch_size is None:
            self.cfg_impl.microbatch_size = self.cfg_train.batch_size
        if self.cfg_impl.microbatch_size > self.cfg_train.batch_size:
            raise ValueError(f"MBS is {self.cfg_impl.microbatch_size}, but BS is only {self.cfg_train.batch_size}.")
        self.current_seq_length = seq_length

        # Mixed Precision:
        enabled = self.cfg_impl.mixed_precision if setup["device"].type != "cpu" else False
        # Modules like LN are unsupported on CPU amp, so mixed precision args are disregarded on CPU
        # See https://pytorch.org/docs/stable/amp.html#cpu-op-specific-behavior and check for layer_norm
        enable_scaling = self.cfg_impl.grad_scaling and self.cfg_impl.mixed_precision and setup["device"].type != "cpu"
        # self.scaler = torch.cuda.amp.GradScaler(enabled=enable_scaling)
        self.scaler = torch.amp.GradScaler(init_scale=2.0**16, enabled=enable_scaling, device=setup["device"].type)
        amp_dtype = getattr(torch, self.cfg_impl.mixed_precision_target_dtype) if setup["device"].type != "cpu" else torch.bfloat16
        self.amp_settings = dict(device_type=setup["device"].type, enabled=enabled, dtype=amp_dtype)

        # Choose setup and move model
        self.setup = setup # device, dtype
        model.to(**self.setup)

        from ..utils import flatten

        # model = torch.compile(
        #     model,
        #     mode=self.cfg_impl.mode,
        #     dynamic=self.cfg_impl.dynamic,
        #     fullgraph=self.cfg_impl.fullgraph,
        #     backend=self.cfg_impl.backend,
        #     disable=not cfg_impl.compile_torch,
        #     # detailed options; cannot be given at the same time as mode:
        #     options=flatten(cfg_impl._inductor_vars, parent_key="", sep=".") if cfg_impl._inductor_vars is not None else None,
        # )
        # print('after compile model', model)

        if torch.distributed.is_initialized():
            # print('if torch.distributed.is_initialized()')
            self.model = self._init_distributed(model)
            self.num_machines = torch.distributed.get_world_size()
        else:
            self.model = model
            self.model.no_sync = nullcontext
            self.num_machines = 1

        # Microbatch accumulation settings and counters
        self.effective_mbs = self.cfg_impl.microbatch_size * self.num_machines  # across machines
        self.current_batch_size = self.cfg_train.batch_size if self.cfg_train.batch_size_ramp == 0 else self.effective_mbs
        self.accumulation_steps_expected = self.current_batch_size // self.effective_mbs
        self.accumulated_samples = 0  # Record the number of samples seen, reset after triggering gradient update
        self.steps = 0  # Record the number of times "step" has been triggered

        self.initial_time = time.time() - already_elapsed_time
        self.optimizer, self.scheduler = _load_optimizer(model, cfg_train, cfg_impl, self.initial_time)

    def step(self, batch: dict[str, torch.Tensor]):
        self.accumulated_samples += self.effective_mbs
        context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        with context():
            loss = self.forward(**batch)["loss"]
            self.backward(loss)
            self.optimizer_step()
        return loss.detach()

    def step_heatmap(self, batch: dict[str, torch.Tensor]):
        self.accumulated_samples += self.effective_mbs
        context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        with context():
            loss = self.forward(**batch)["loss"]
        return loss.detach()

    def to_device(self, batch: dict[str, torch.Tensor], keys: list[str] = ["input_ids", "labels"]):
        """Move batch of data into device memory."""
        device_batch = {
            k: v.to(device=self.setup["device"], dtype=torch.long if k == "input_ids" else None, non_blocking=True)
            for k, v in batch.items()
            if k in keys  # Add more keywords here if needed
        }
        return device_batch

    def forward(self, *inputs, **kwargs):
        with torch.autocast(**self.amp_settings):
            return self.model(*inputs, **kwargs)

    def backward(self, loss):
        return self.scaler.scale(loss / self.accumulation_steps_expected).backward()

    @torch.no_grad()
    def forward_inference(self, *inputs, **kwargs):
        with torch.autocast(**self.amp_settings):
            outputs = self.model(*inputs, **kwargs)["logits"]
        if outputs.shape[-1] == 1:
            predictions = outputs.squeeze(dim=-1)
        else:
            predictions = outputs.argmax(dim=-1)
        return outputs, predictions

    def optimizer_step(self):
        """Requires a scheduler that is based on iterations instead of epochs."""
        self.steps += 1
        if self.accumulated_samples >= self.current_batch_size:
            self.accumulated_samples = 0

            if self.cfg_train.gradient_clipping is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg_train.gradient_clipping, norm_type=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # print(f'optimizer_step optimizer_step')
            self.optimizer.zero_grad()
            self.schedule_batch_size()
        self.scheduler.step()  # Trigger in every step, otherwise things get annoying with grad accumulation

    def set_train_batch_size(self, batch_size):
        """Allow dynamic modifications of batch size."""
        self.current_batch_size = batch_size
        self.accumulation_steps_expected = self.current_batch_size // self.effective_mbs

    def schedule_batch_size(self):
        """Optionally implement linear batch size ramp-ups."""
        if (self.cfg_train.batch_size_ramp > 0) and (self.cfg_train.batch_size_ramp < 1):
            # interpret batch_size_ramp as percentage of total budget:
            elapsed_hours = (time.time() - self.initial_time) / 60 / 60
            fake_step = int(elapsed_hours / self.cfg_train.budget * self.cfg_train.steps)

            batch_size_step = self.cfg_train.batch_size / (self.cfg_train.steps * self.cfg_train.batch_size_ramp)
            new_batch_size = min(int(fake_step * batch_size_step // self.effective_mbs + 1) * self.effective_mbs, self.cfg_train.batch_size)
        elif self.steps < self.cfg_train.batch_size_ramp:
            # interpret batch_size_ramp as fixed number of steps for ramp:
            batch_size_step = self.cfg_train.batch_size / self.cfg_train.batch_size_ramp
            new_batch_size = int(self.steps * batch_size_step // self.effective_mbs + 1) * self.effective_mbs
        else:
            new_batch_size = self.cfg_train.batch_size
        self.set_train_batch_size(new_batch_size)

    def record_batch_size(self):
        if self.cfg_train.optim_mod.name != "progressive-batching":
            return self.current_batch_size
        else:
            return self.optimizer.last_full_step_accumulation * self.current_batch_size

    def record_tokens_per_step(self):
        """Tokens in each microbatch step."""
        return self.current_seq_length * self.effective_mbs

    def _init_distributed(self, model):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.setup["device"]] if self.setup["device"].type == "cuda" else None,
            output_device=self.setup["device"] if self.setup["device"].type == "cuda" else None,
            broadcast_buffers=self.cfg_impl.broadcast_buffers,
            bucket_cap_mb=self.cfg_impl.bucket_cap_mb,
            gradient_as_bucket_view=self.cfg_impl.gradient_as_bucket_view,
            static_graph=self.cfg_impl.static_graph,
        )
        return model

    @torch.no_grad()
    def retrieve_model_state_dict(self):
        if self.cfg_impl.compile_torch:
            if torch.distributed.is_initialized():
                state_dict = self.model.module._orig_mod.state_dict()  # ughhhh
            else:
                state_dict = self.model._orig_mod.state_dict()  # ugh
        else:
            if torch.distributed.is_initialized():
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        return state_dict

    def load_checkpoint(self, cfg_arch, file, skip_optim_state=True):
        """Load list of states from checkpoint file. Not generally compatible with any other engine?"""
        if file.startswith("hf://"):
            if file.endswith("-untrained"):
                log.info("Loading NO pretrained model as a sanity check ...")
            else:
                self.model = self.model.from_pretrained(file.split("hf://")[1], config=cfg_arch).to(**self.setup)
                # reinit optimizer:
                self.optimizer, self.scheduler = _load_optimizer(self.model, self.cfg_train, self.cfg_impl)
        else:
            model_state = load_file(file, device=str(self.setup["device"]))
            # This loader includes a few legacy options:
            if "encoder.embedding.word_embedding.weight" not in model_state:
                # Hack to save space when saving the model, more clever though would be save the right one in the first place
                model_state["encoder.embedding.word_embedding.weight"] = model_state["decoder.weight"]
            try:
                sanitized_state = {}
                for k, v in model_state.items():
                    if k.startswith("module."):
                        k = k[7:]
                    if self.cfg_impl.compile_torch:
                        k = f"_orig_mod.{k}"
                    if torch.distributed.is_initialized():
                        k = f"module.{k}"
                    sanitized_state[k] = v
                self.model.load_state_dict(sanitized_state, strict=True)
            except RuntimeError as e:
                log.info(f"State dict difference is {str(e).split('Error(s) in loading state_dict for')[1]}... Ok?")
                self.model.load_state_dict(sanitized_state, strict=False)
            self.model.to(**self.setup)

    def save_training_checkpoint(self, identifier="intermediate.pth", directory="", metadata=None):
        """Path, identifier and additional client state. This checkpoint can be used to resume training.
        The default behavior is to save this checkpoint relative to the training working directory.

        Has to be .pth because safetensors are annoying to dump a bunch of optim states, scales and schedules
        """
        file = os.path.join(directory, str(identifier))
        if directory != "":
            os.makedirs(directory, exist_ok=True)

        save_state = dict()
        save_state["optim"] = self.optimizer.state_dict()
        save_state["model"] = self.model.state_dict()  # this is the raw key containing _orig and _module flags
        save_state["scheduler"] = self.scheduler.state_dict()
        save_state["scaler"] = self.scaler.state_dict()
        save_state["metadata"] = metadata
        
        torch.save(save_state, file)

    def load_training_checkpoint(self, identifier="intermediate.pth", directory=""):
        self.optimizer.zero_grad()
        file = os.path.join(directory, str(identifier))

        save_state = torch.load(file, map_location=torch.device("cpu"))
        self.model.load_state_dict(save_state["model"])  # why does this end up on GPU?
        self.optimizer.load_state_dict(save_state["optim"])
        self.scheduler.load_state_dict(save_state["scheduler"])
        self.scaler.load_state_dict(save_state["scaler"])
        log.info(f"Sucessfully loaded state with metadata {save_state['metadata']}")
        return save_state["metadata"]

    def save_final_model(self, base_directory, identifier, tokenizer, cfg_arch, dryrun=False):
        """This checkpoint can be used for downstream tasks.
        The default behavior is to save this checkpoint to a checkpoints folder under base_directory/name/checkpoints"""
        try:
            identifier_str = f"{identifier:2.4f}"
        except ValueError:
            identifier_str = str(identifier)
        full_path = os.path.join(base_directory, "checkpoints", identifier_str)
        os.makedirs(full_path, exist_ok=True)
        # This saves tokenizer_config.json, tokenizer.json and special_tokens_map.json to this folder
        if not dryrun:
            tokenizer.save_pretrained(full_path)
            # Save model.safetensors, model_config.json
            save_file(self.retrieve_model_state_dict(), os.path.join(full_path, "model.safetensors"))
            torch.save(self.retrieve_model_state_dict(), os.path.join(full_path, "model.pth"))
            # legacy save: torch.save(self.retrieve_model_state_dict(), os.path.join(full_path, "model.pth"))
            with open(os.path.join(full_path, "model_config.json"), "w") as file:
                json.dump(OmegaConf.to_container(cfg_arch, resolve=True), file)

    def push_to_hub(self, tokenizer, cfg, dryrun=False):
        """Analogous to save_final_model, but save model to hugginface hub."""
        from huggingface_hub import HfApi
        from io import BytesIO

        api = HfApi()

        if not dryrun:
            log.info(f"Pushing model to hub repository {cfg.impl.hf_directoy_name}.")
            final_state_dict = self.retrieve_model_state_dict()
            self.model.load_state_dict(final_state_dict)

            # Push model with safetensors:
            # This is a manual modification of model.push_to_hub which doesn't support safetensors yet
            repo_id = cfg.impl.hf_directoy_name
            if os.path.isdir(repo_id):
                working_dir = repo_id
                repo_id = repo_id.split(os.path.sep)[-1]
            else:
                working_dir = repo_id.split("/")[-1]
            repo_id = self.model._create_repo(repo_id)
            use_temp_dir = not os.path.isdir(working_dir)
            with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
                files_timestamps = self.model._get_files_timestamps(work_dir)
                # Save all files.
                self.model.save_pretrained(
                    work_dir,
                    max_shard_size="10GB",
                    safe_serialization=True,
                    state_dict=self.retrieve_model_state_dict(),
                )
                self.model._upload_modified_files(
                    work_dir,
                    repo_id,
                    files_timestamps,
                    commit_message=None,
                    token=None,
                    create_pr=False,
                )
            # Push tokenizer:
            tokenizer.push_to_hub(cfg.impl.hf_directoy_name)
            # Push config files:
            for config_group, config_name in zip([cfg.arch, cfg.data, cfg.train], ["arch", "data", "train"]):
                buffer = BytesIO()
                buffer.write(json.dumps(OmegaConf.to_container(config_group, resolve=True), indent=4).encode())
                api.upload_file(
                    path_or_fileobj=buffer,
                    path_in_repo=f"{config_name}_budget_hours_{cfg.budget}.json",
                    repo_id=f"{api.whoami()['name']}/{cfg.impl.hf_directoy_name}",  # there has to be a better way to do this, but ...
                    repo_type="model",
                )
        else:
            log.info(f"Skipping huggingface upload in dryrun state. Would upload to {cfg.impl.hf_directoy_name}.")


class TorchEngineFull(TorchEngineMinimal):
    """This class mirrors deepspeed functionality. Not all changes are implemented in this version.

    See TorchEngineFull for more modifications.
    """

    def __init__(self, model, cfg_train, cfg_impl, setup=_default_setup, seq_length=128):
        """Load Engine. The model will be compiled by default."""
        super().__init__(model, cfg_train, cfg_impl, setup, seq_length)

        # Optional sequence curriculum:
        self.sequence_curriculum = "sequence_curriculum" in cfg_train
        self.data_seq_length = seq_length
        self.current_seq_length = seq_length if not self.sequence_curriculum else cfg_train.sequence_curriculum.lengths[0]
        self.sequence_unfold = None if not self.sequence_curriculum else cfg_train.sequence_curriculum.unfold

        # Optional EMA/LAWA-type weight averages
        if "weight_averaging" in cfg_train:
            self.weight_averaging_frequency = cfg_train.weight_averaging.frequency
            self.weight_averaging = cfg_train.weight_averaging
            if self.weight_averaging.type == "EMA":
                self.param_store = [p.detach().clone() for p in model.parameters()]  # keep on CPU
                self.buffer_store = [b.detach().clone() for b in model.buffers()]
            else:
                self.store = []
        else:
            self.weight_averaging_frequency = 0
        self.initial_time = time.time()

    def optimizer_step(self):
        """Requires a scheduler that is based on iterations instead of epochs."""
        super().optimizer_step()
        if self.accumulated_samples >= self.current_batch_size:
            self.schedule_curriculum()
            self.moving_average_computation()

    def to_device(self, batch: dict[str, torch.Tensor], keys: list[str] = ["input_ids", "labels"]):
        """Move batch of data into device memory."""
        device_batch = super().to_device(batch)
        self.set_sequence_curriculum_(device_batch)
        return device_batch

    def set_sequence_curriculum_(self, device_batch):
        """Assume huggingface data is B S"""
        if self.sequence_curriculum:
            for key, tensor in device_batch.items():
                if self.sequence_unfold:
                    device_batch[key] = tensor.view(-1, self.current_seq_length)
                else:
                    device_batch[key] = tensor[:, : self.current_seq_length].clone()

    def schedule_curriculum(self):
        """Optionally implement linear sequence lengths curriculum."""
        if self.sequence_curriculum:
            # Sequence curriculum should be a dict of two lists:
            # lengths (needs to be monotone ascending integers)
            # triggers (needs to be monotone ascending floats between 0 and 1)
            # and a keyword unfold = True/False
            elapsed_hours = (time.time() - self.initial_time) / 60 / 60
            fraction_elapsed = elapsed_hours / self.cfg_train.budget
            lengths = self.cfg_train.sequence_curriculum.lengths
            triggers = self.cfg_train.sequence_curriculum.triggers
            for trigger, length in zip(triggers, lengths):
                if fraction_elapsed > trigger:
                    self.current_seq_length = length

    def record_tokens_per_step(self):
        """Tokens in each microbatch step."""
        if not self.sequence_curriculum:
            return self.current_seq_length * self.cfg_impl.microbatch_size
        else:
            if self.sequence_unfold:
                # Same number of tokens in this case:
                return self.current_seq_length * (self.data_seq_length // self.current_seq_length) * self.cfg_impl.microbatch_size
            else:
                # Reduced number of tokens here:
                return self.current_seq_length * self.cfg_impl.microbatch_size

    def moving_average_computation(self):
        if self.weight_averaging_frequency > 0:
            if (self.steps % self.weight_averaging_frequency) == 0:
                params = [p.detach().cpu() for p in self.model.parameters()]
                buffers = [b.detach().cpu() for b in self.model.buffers()]
                if self.weight_averaging.type == "EMA":
                    update_ema(params, self.param_store, buffers, self.buffer_store, momentum=self.weight_averaging.momentum)
                else:  # latest weight averaging
                    self.param_store, self.buffer_store = updated_latest_weight_average(
                        params, buffers, self.store, last_k=self.weight_averaging.last_k
                    )

    @torch.no_grad()
    def retrieve_model_state_dict(self):
        if self.weight_averaging_frequency > 0:
            # Use weight averaged weights
            for param, param_ma in zip(self.model.parameters(), self.param_store):
                param.copy_(param_ma.data)
            for buffer, buffer_ma in zip(self.model.buffers(), self.buffer_store):
                buffer.copy_(buffer_ma.data)
            return self.model.state_dict()
        else:
            # Else use normal state dict
            return self.model.state_dict()

    def gradinit(self, data_iterable, optim_cfg, gradinit_cfg):
        """Run data-based initialization search as described in Zhu et al.,
        "GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training"

        Depends on functorch!

        This is gradinit without gradient aggregation, which allows higher-order derivatives
        """
        import functorch

        fmodel, params, buffers = functorch.make_functional_with_buffers(self.model)

        scales = [torch.tensor(1.0, **self.setup, requires_grad=True) for p in params]  # Modify all params by default
        # Prepare for functional optimizer:

        exp_avgs = [torch.tensor(0.0, **self.setup) for s in scales]
        exp_avg_sqs = [torch.tensor(0.0, **self.setup) for s in scales]
        state_steps = [torch.tensor(0.0, **self.setup) for s in scales]

        adam_fn = partial(torch.optim._functional.adam, amsgrad=False, beta1=0.9, beta2=0.98, weight_decay=0, eps=1e-6, maximize=False)

        eta = optim_cfg.lr
        for step in range(gradinit_cfg.steps):
            # scale params
            scaled_params = [p * s for p, s in zip(params, scales)]
            # ## Compute first step ##
            data_batch = self.to_device(next(data_iterable)[1])
            with torch.autocast(**self.amp_settings):
                loss0 = fmodel(**data_batch, params=scaled_params, buffers=buffers)["loss"]
            grads = torch.autograd.grad(loss0, scaled_params, create_graph=gradinit_cfg.second_order, only_inputs=True)
            gnorm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
            # Take first step
            # p <- p - eta*g
            if gradinit_cfg.step_type == "sign-grad":
                param_step = [p - eta * g.sign() for p, g in zip(scaled_params, grads)]
            elif gradinit_cfg.step_type == "norm-grad":
                param_step = [p - eta * g / gnorm for p, g in zip(scaled_params, grads)]
            else:
                param_step = [p - eta * g for p, g in zip(scaled_params, grads)]

            # ## Modify scales ##
            data_batch = self.to_device(next(data_iterable)[1])
            with torch.autocast(**self.amp_settings):
                loss1 = fmodel(**data_batch, params=param_step, buffers=buffers)["loss"]
            grads = torch.autograd.grad(loss1 / eta + (gnorm - 1).pow(2), scales, only_inputs=True)
            [g.zero_() for (name, _), g in zip(self.model.named_parameters(), grads) if "pos_embedding" in name]
            # Take adam step:
            with torch.no_grad():
                adam_fn(scales, grads, exp_avgs, exp_avg_sqs, [], state_steps, lr=gradinit_cfg.tau)
                # Project onto constraints and detach
                scales = [s.clamp_(min=gradinit_cfg.min_scale, max=gradinit_cfg.max_scale) for s in scales]
            # log.info(f"Gradient: Loss0: {loss0:2.4f}. Loss1: {loss1:2.4f}. Grad Norm: {gnorm:2.4f}.")
            # print([f"{name}:{s.item():2.4f}" for (name, _), s in zip(self.model.named_parameters(), scales)])

        # Finally copy scales into the existing model
        with torch.no_grad():
            for param, scale in zip(self.model.parameters(), scales):
                param.mul_(scale)

class CustomAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, fused=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdamW, self).__init__(params, defaults)
        self.noise = False
        if self.noise:
            self.noise_scale = 2**(-8)
            log.info(colored(f'Add noise to gradient with Mean 0, Std_dev {self.noise_scale}.', 'green'))
        else:
            log.info(colored(f'Not add noise to the gradients.', 'green'))
    def step(self):
        v_hat_maxs = []
        v_hat_mins = []
        v_maxs = []
        v_mins = []
        sqrt_v_hat_plus_eps_maxs = []
        sqrt_v_hat_plus_eps_mins = []
        count = 0        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group.get('amsgrad', False)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                 # Adding noise to the gradients
                if self.noise:
                    noise = torch.randn_like(grad) * self.noise_scale
                    grad.add_(noise)
                
                # exp_avg: m_t, exp_avg_sq: v_t
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                # print(f"step: {state['step']}")

                # Weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Momentum update
                # m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t ** 2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # v_maxs.append(torch.max(exp_avg_sq).item())
                # v_mins.append(torch.min(exp_avg_sq).item())
                
                # Bias correction
                # hat_m_t = m_t / (1 - beta_1 ** t)
                exp_avg = exp_avg / (1 - beta1 ** state['step'])
                # print(f"(1 - beta1 ** state['step']): {(1 - beta1 ** state['step'])}")
                # hat_v_t = v_t / (1 - beta_2 ** t)
                exp_avg_sq = exp_avg_sq / (1 - beta2 ** state['step'])
                v_hat_maxs.append(torch.max(exp_avg_sq).item())
                v_hat_mins.append(torch.min(exp_avg_sq).item())
                # print(f"(1 - beta2 ** state['step']): {(1 - beta2 ** state['step'])}")
                
                # sqrt(hat_v_t) + eps
                denom = exp_avg_sq.sqrt().add_(group['eps'])                
                
                # sqrt_v_hat_plus_eps_maxs.append(torch.max(exp_avg_sq.sqrt()+group['eps']).item())
                # sqrt_v_hat_plus_eps_mins.append(torch.min(exp_avg_sq.sqrt()+group['eps']).item())
                # print(f'exp_avg_sq.sqrt(): {exp_avg_sq.sqrt()}')
                # print(f"exp_avg_sq.sqrt().add_(group['eps']): {exp_avg_sq.sqrt().add_(group['eps'])}")

                step_size = group['lr']

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        # with open ('frac_max.txt', 'a') as f:
        #     f.write(f'{max(torch.abs(exp_avg/denom)).item()}\n')
        # with open ('frac_min.txt', 'a') as f:
        #     f.write(f'{min(torch.abs(exp_avg/denom)).item()}\n')
        # with open ('custom_adamw_v_max.txt', 'a') as f:
        #     f.write(f'{max(v_maxs)}\n')
        # with open ('custom_adamw_v_min.txt', 'a') as f:
            # f.write(f'{min(v_maxs)}\n')
        # with open ('custom_adamw_v_hat_max.txt', 'a') as f:
        #     f.write(f'{max(v_hat_maxs)}\n')
        # with open ('custom_adamw_v_hat_min.txt', 'a') as f:
        #     f.write(f'{min(v_hat_mins)}\n')
        # with open ('custom_adamw_lr.txt', 'a') as f:
        #     f.write(f"{group['lr']}\n")
        # with open ('custom_adamw_sqrt_v_hat_plus_eps_max.txt', 'a') as f:
        #     f.write(f'{max(sqrt_v_hat_plus_eps_maxs)}\n')
        # with open ('custom_adamw_sqrt_v_hat_plus_eps_min.txt', 'a') as f:
        #     f.write(f'{min(sqrt_v_hat_plus_eps_mins)}\n')
        # print(f'sqrt_v_hat_maxs: {max(sqrt_v_hat_maxs)}')
        # print(f'sqrt_v_hat_mins: {min(sqrt_v_hat_mins)}')

class CustomAdamW_variant(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, fused=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdamW_variant, self).__init__(params, defaults)
        self.noise = False
        self.task_name = None
        self.approx_sqrt_inverse = None
        # self.div_max = None
    def step(self):
        # print(f"self.state: {self.state['state']}")
        sqrt_input_maxs = []
        sqrt_input_mins = []
        # v_maxs = []
        # v_mins = []
        # sqrt_v_hat_plus_eps_maxs = []
        # sqrt_v_hat_plus_eps_mins = []     
        # print(f'self.param_groups: {self.param_groups}')      
        for group in self.param_groups:
            # print(f'group: {group}')
            for p in group['params']:
                if p.grad is None:
                    continue
                # print(f'p.shape: {p.shape}')
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group.get('amsgrad', False)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                 # Adding noise to the gradients
                if self.noise:
                    noise = torch.randn_like(grad) * self.noise_scale
                    grad.add_(noise)
                
                # exp_avg: m_t, exp_avg_sq: v_t
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1
                # print(f"step: {state['step']}")

                # Weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                # print(f'p: {p.shape}')
                # print(f"weight_decay: {group['weight_decay']}")
                # Momentum update
                # m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta_2 * v_{t-1} + (1 - beta_2) * (g_t ** 2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # v_maxs.append(torch.max(exp_avg_sq).item())
                # v_mins.append(torch.min(exp_avg_sq).item())
                
                # Bias correction
                # hat_m_t = m_t / (1 - beta_1 ** t)
                exp_avg = exp_avg / (1 - beta1 ** state['step'])
                # print(f"(1 - beta1 ** state['step']): {(1 - beta1 ** state['step'])}")
                # hat_v_t = v_t / (1 - beta_2 ** t)
                exp_avg_sq = exp_avg_sq / (1 - beta2 ** state['step'])
                # print(f"(1 - beta2 ** state['step']): {(1 - beta2 ** state['step'])}")
                
                # sqrt(hat_v_t + eps)
                sqrt_input = exp_avg_sq + (group['eps'])
                sqrt_input_maxs.append(torch.max(sqrt_input).item())
                # print(f'torch.max(sqrt_input).item(): {torch.max(sqrt_input).item()}')
                sqrt_input_mins.append(torch.min(sqrt_input).item())
                
                # x_min = torch.min(sqrt_input)
                # # print(f'x_min: {x_min}')
                # true_sqrt_inverse = 1/torch.sqrt(x_min)
                # approx_sqrt_inverse = Inverse_sqrt_approx(x_min / self.div_max) / math.sqrt(self.div_max)
                # # print(f'true_sqrt_inverse: {true_sqrt_inverse}')
                # # print(f'approx_sqrt_inverse: {approx_sqrt_inverse}')
                
                if self.approx_sqrt_inverse:
                    denom = 1 / (Inverse_sqrt_approx(sqrt_input / self.div_max) / math.sqrt(self.div_max))
                    # print(f'torch.min(sqrt_input).item()/self.div_max: {torch.min(sqrt_input).item()/self.div_max} max/div_max: {torch.max(sqrt_input).item()/self.div_max}')
                else:
                    denom = sqrt_input.sqrt()
                    
                # if self.approx_sqrt_inverse:
                #     denom_approx = denom
                #     denom_true = sqrt_input.sqrt()
                #     # print(f'diffrence between sqrt inputs: {torch.max(denom_approx-denom_true).item()}')
                
                # sqrt_v_hat_plus_eps_maxs.append(torch.max(exp_avg_sq.sqrt()+group['eps']).item())
                # sqrt_v_hat_plus_eps_mins.append(torch.min(exp_avg_sq.sqrt()+group['eps']).item())
                # print(f'exp_avg_sq.sqrt(): {exp_avg_sq.sqrt()}')
                # print(f"exp_avg_sq.sqrt().add_(group['eps']): {exp_avg_sq.sqrt().add_(group['eps'])}")

                step_size = group['lr']

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        # with open ('frac_max.txt', 'a') as f:
        #     f.write(f'{max(torch.abs(exp_avg/denom)).item()}\n')
        # with open ('frac_min.txt', 'a') as f:
        #     f.write(f'{min(torch.abs(exp_avg/denom)).item()}\n')
        # with open ('custom_adamw_v_max.txt', 'a') as f:
        #     f.write(f'{max(v_maxs)}\n')
        # with open ('custom_adamw_v_min.txt', 'a') as f:
        #     f.write(f'{min(v_mins)}\n')
        # with open (f'{self.task_name}/custom_adamw_var_sqrt_input_max.txt', 'a') as f:
        #     f.write(f'{max(sqrt_input_maxs)}\n')
        # with open (f'{self.task_name}/custom_adamw_var_sqrt_input_min.txt', 'a') as f:
        #     f.write(f'{min(sqrt_input_mins)}\n')
            # print(f'sqrt_input_max: {min(sqrt_input_mins)}')
        # with open ('custom_adamw_sqrt_v_hat_plus_eps_max.txt', 'a') as f:
        #     f.write(f'{max(sqrt_v_hat_plus_eps_maxs)}\n')
        # with open ('custom_adamw_sqrt_v_hat_plus_eps_min.txt', 'a') as f:
        #     f.write(f'{min(sqrt_v_hat_plus_eps_mins)}\n')
        # print(f'sqrt_v_hat_maxs: {max(sqrt_v_hat_maxs)}')
        # print(f'sqrt_v_hat_mins: {min(sqrt_v_hat_mins)}')
        # with open (f'{self.task_name}/custom_adamw_lr.txt', 'a') as f:
        #     f.write(f"{group['lr']}\n")
        # print(f"state['step']: {state['step']}")
        
    def graph_gradients(self, task_name, param_names, epoch):
        path_grad = f'{task_name}/grads'
        path_weight = f'{task_name}/weights'
        os.makedirs(path_grad, exist_ok=True)
        os.makedirs(path_weight, exist_ok=True)
        gradients = []
        weights = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    gradients.append(p.grad.data.clone())
                    weights.append(p.data.clone())
        
        for i in range(len(gradients)):
            # Print and store gradients
            grad_bins = [-np.inf] + [-2**(-i) for i in range(7, 18)] + [2**(-i) for i in range(17, 6, -1)] + [np.inf]
            grad_hist, grad_bin_edges = np.histogram(gradients[i].cpu().numpy(), bins=grad_bins)
            grad_ratios = grad_hist / grad_hist.sum()
            
            # Print and store weights
            weight_bins = [-np.inf] + [-2**(-i) for i in range(7, 18)] + [2**(-i) for i in range(17, 6, -1)] + [np.inf]
            weight_hist, weight_bin_edges = np.histogram(weights[i].cpu().numpy(), bins=weight_bins)
            weight_ratios = weight_hist / weight_hist.sum()
            
            def format_tick_label(low, high):
                if low == -np.inf:
                    return f'<= -2^(-7)'
                elif high == np.inf:
                    return f'>= 2^(-7)'
                elif low == -2**(-17):
                    return f'-2^(-17) to 2^(-17)'
                else:
                    if low > 0:
                        return f'2^({int(np.log2(low))}) to 2^({int(np.log2(high))})'
                    else:
                        return f'-2^({int(np.log2(-low))}) to -2^({int(np.log2(-high))})'
            grad_tick_label = [format_tick_label(grad_bin_edges[j], grad_bin_edges[j+1]) for j in range(len(grad_bin_edges)-1)]
            weight_tick_label = [format_tick_label(weight_bin_edges[j], weight_bin_edges[j+1]) for j in range(len(weight_bin_edges)-1)]
            
            # Plot gradients histogram
            plt.figure(figsize=(15, 12))
            bars = plt.bar(range(len(grad_ratios)), grad_ratios, tick_label=grad_tick_label)
            plt.xlabel('Gradient', fontsize=10)
            plt.ylabel('Percentage', fontsize=20)
            plt.title(f'Epoch {epoch}, {param_names[i]} Gradient', fontsize=20)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=14)
            plt.grid(True)
            for bar, ratio in zip(bars, grad_ratios):
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, f'{ratio:.2}', va='bottom', ha='center')
            
            plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.12)    
            plt.savefig(f'{path_grad}/{epoch}_{param_names[i]}_gradient.png')
            plt.close()
            
            # Plot weights histogram
            plt.figure(figsize=(15, 12))
            bars = plt.bar(range(len(weight_ratios)), weight_ratios, tick_label=weight_tick_label)
            plt.xlabel('Weight', fontsize=10)
            plt.ylabel('Percentage', fontsize=20)
            plt.title(f'Epoch {epoch}, {param_names[i]} Weight', fontsize=20)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=14)
            plt.grid(True)
            for bar, ratio in zip(bars, weight_ratios):
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, f'{ratio:.2}', va='bottom', ha='center')
            
            plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.12)    
            plt.savefig(f'{path_weight}/{epoch}_{param_names[i]}_weight.png')
            plt.close()
            
                        
def _load_optimizer(model, cfg_train, cfg_impl, initial_time=0):
    # Filter some parameters
    grouped_parameters = group_parameters(model, cfg_train)

    # Select optimizer implementation
    if cfg_train.optim.type == "AdamW":
        optimizer_class = torch.optim.AdamW
    elif cfg_train.optim.type == "CustomAdamW":
        optimizer_class = CustomAdamW
    elif cfg_train.optim.type == "CustomAdamW_variant":
        optimizer_class = CustomAdamW_variant
    elif cfg_train.optim.type == "Adam":
        optimizer_class = torch.optim.Adam
    elif cfg_train.optim.type == "RAdam":
        optimizer_class = torch.optim.RAdam
    elif cfg_train.optim.type == "SGD":
        optimizer_class = torch.optim.SGD
    elif cfg_train.optim.type == "Adafactor":
        optimizer_class = transformers.Adafactor
    elif cfg_train.optim.type == "Shampoo":
        optimizer_class = Shampoo
    elif cfg_train.optim.type == "AdaHessian":
        optimizer_class = Adahessian
    elif cfg_train.optim.type == "AdamWScale":
        optimizer_class = AdamWScale
    elif cfg_train.optim.type == "Sophia-G":
        optimizer_class = Sophia
    elif cfg_train.optim.type == "Lion":
        from lion_pytorch import Lion

        optimizer_class = Lion

    elif cfg_train.optim.type == "Adam8bit":
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.Adam8bit
    elif cfg_train.optim.type == "AGD":
        depth = len(list(model.parameters()))
        optimizer_class = partial(AGD, depth=depth)
    else:
        raise ValueError(f"Invalid optimizer {cfg_train.optim.type} given.")
    optimizer_args = {k: v for k, v in cfg_train.optim.items() if k != "type"}
    # print(f'optimizer_args: {optimizer_args}')
    if cfg_impl.foreach_optimizer and cfg_train.optim.type != "Shampoo": # 실행 x
        optimizer_args["foreach"] = True

    if torch.distributed.is_initialized() and cfg_impl.zero_redundancy_optimizer:
        # # The overlap option is a whole bucket of problems in itself for now...
        # optimizer = ZeroRedundancyOptimizer(
        #     grouped_parameters,
        #     optimizer_class=optimizer_class,
        #     parameters_as_bucket_view=True,
        #     overlap_with_ddp=False,
        #     **optimizer_args,
        # )
        ...
    else:
        optimizer = optimizer_class(grouped_parameters, **optimizer_args) # opt class: adamW

    if cfg_train.optim_mod.name == "none":
        optimizer_to_schedule = optimizer
    else:
        optim_params = {k: v for k, v in cfg_train.optim_mod.items() if k != "name"}
        if cfg_train.optim_mod.name == "LARS":
            optimizer = LARS(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "LARC":
            optimizer = LARS(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "SAM":
            optimizer = SAM(optimizer, **optim_params)
        elif cfg_train.optim_mod.name == "progressive-batching":
            optimizer = ProgressiveBatching(optimizer, **optim_params)

        optimizer_to_schedule = optimizer.optim

    scheduler = get_schedule_fn(initial_time, cfg_train)(optimizer_to_schedule)

    return optimizer, scheduler
