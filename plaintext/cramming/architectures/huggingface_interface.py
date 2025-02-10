"""BERT variations based on the huggingface implementation."""

import transformers
from omegaconf import OmegaConf
from termcolor import colored


def construct_huggingface_model(cfg_arch, vocab_size, downstream_classes=None):
    print(colored('construct_huggingface_model', 'red'))
    print(colored(cfg_arch, 'red'))
    """construct model from given configuration. Only works if this arch exists on the hub."""
    if downstream_classes is None:
        if isinstance(cfg_arch, transformers.PretrainedConfig):
            configuration = cfg_arch
        else:
            print(colored('else', 'red'))
            configuration = transformers.BertConfig(**cfg_arch)
            # print(configuration)
        configuration.pad_token_id = None  # Need to drop this during pretraining, otherwise leads to a graph break in a HF warning
        configuration.vocab_size = vocab_size
        model = transformers.AutoModelForMaskedLM.from_config(configuration)
        model.vocab_size = model.config.vocab_size
        print(colored('model.vocab_size: {}'.format(model.vocab_size), 'red'))
    else:
        if isinstance(cfg_arch, transformers.PretrainedConfig):
            configuration = cfg_arch
            configuration.num_labels = downstream_classes
        else:
            configuration = OmegaConf.to_container(cfg_arch)
            configuration = transformers.BertConfig(**configuration, num_labels=downstream_classes)
        configuration.vocab_size = vocab_size

        configuration.problem_type = None  # always reset this!
        model = transformers.AutoModelForSequenceClassification.from_config(configuration)
        model.vocab_size = vocab_size
    return model
