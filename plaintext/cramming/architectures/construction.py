"""Interface to construct models."""

from .huggingface_interface import construct_huggingface_model
from .funnel_transformers import construct_scriptable_funnel
from .recurrent_transformers import construct_scriptable_recurrent
from .sanity_check import SanityCheckforPreTraining
from .crammed_bert import construct_crammed_bert
from .crammed_bert_modified import construct_crammed_bert_modified, construct_crammed_bert_grad

import logging
from ..utils import is_main_process
from termcolor import colored

log = logging.getLogger(__name__)

def construct_model(cfg_arch, vocab_size, downstream_classes=None):
    # print(colored('cfg_arch: {}'.format(cfg_arch), 'yellow'))
    # print(colored('Model construction', 'yellow'))
    model = None
    if cfg_arch.architectures is not None:
        # attempt to solve locally
        if "ScriptableCrammedBERT" in cfg_arch.architectures:
            print(colored('ScriptableCrammedBERT', 'yellow'))
            model = construct_crammed_bert(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableCrammedBERT-modified" in cfg_arch.architectures:
            # print(colored('ScriptableCrammedBERT_modified', 'yellow'))
            model = construct_crammed_bert_modified(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableCrammedBERT-grad" in cfg_arch.architectures:
            # print(colored('ScriptableCrammedBERT_modified', 'yellow'))
            model = construct_crammed_bert_grad(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableFunnelLM" in cfg_arch.architectures:
            print(colored('ScriptableFunnelLM', 'yellow'))
            model = construct_scriptable_funnel(cfg_arch, vocab_size, downstream_classes)
        elif "ScriptableRecurrentLM" in cfg_arch.architectures:
            print(colored('ScriptableRecurrentLM', 'yellow'))
            model = construct_scriptable_recurrent(cfg_arch, vocab_size, downstream_classes)
        elif "SanityCheckLM" in cfg_arch.architectures:
            print(colored('SanityCheckLM', 'yellow'))
            model = SanityCheckforPreTraining(cfg_arch.width, vocab_size)
            

    if model is not None:  # Return local model arch
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with architecture {cfg_arch.architectures[0]} loaded with {num_params:,} parameters.")
        return model

    try:  # else try on HF
        model = construct_huggingface_model(cfg_arch, vocab_size, downstream_classes)
        num_params = sum([p.numel() for p in model.parameters()])
        if is_main_process():
            log.info(f"Model with config {cfg_arch} loaded with {num_params:,} parameters.")
        return model
    except Exception as e:
        raise ValueError(f"Invalid model architecture {cfg_arch.architectures} given. Error: {e}")
