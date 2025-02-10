"""This rewrite is a simplified version of the proposed changes that actually compiles statically in torch 2.0.

This model is the final, optimized crammed model.
OmegaConf
Not all ablations discussed in the paper are implemented as switches in this version,
for all those, check scriptable_bert.py on the old branch.

"""
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification

import time
from datetime import datetime
from typing import Optional
from omegaconf import OmegaConf
from termcolor import colored
import os
from .architectures_grad import ScriptableLMForPreTraining_grad, ScriptableLMForSequenceClassification_grad
from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent_modified,
    PoolingComponent,
    PoolingComponent_lora,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
    Custom_CrossEntropyLoss
)
from .small_FFNs import *
from .attention_modified import get_attention_mechanism
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
from .attention_modified import *
from .combined_transformers import *

import logging
log = logging.getLogger(__name__)

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)

def construct_crammed_bert_modified(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining_modified(config)
        # elif config.arch["objective_layout"] == "SCRIPT":
        #     model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification_modified(config)
    return model

def construct_crammed_bert_grad(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes
    
    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining_grad(config)
        # elif config.arch["objective_layout"] == "SCRIPT":
        #     model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification_grad(config)
    return model


class AttentionComponent_modified(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_arch, use_bias=True):
        super().__init__()
        cfg_attention = cfg_arch.attention
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_arch)
        
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        output, matmul_result = self.self_attention(hidden_states, attention_mask)
        output = self.dense(output)
        return output, matmul_result

class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    It actually turned out better not to scale it, so here the block is effectively smaller than may be expected.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        hidden_states = self.dense_in(hidden_states)

        #######################################
        if self.get_input_range:
            # print('ddddddddddd')
            _, nonlin_inputs = hidden_states.chunk(2, dim=-1)
        #######################################

        hidden_states = self.nonlin(hidden_states)

        if self.get_input_range:
            return self.dense_out(hidden_states), nonlin_inputs
        else:
            dense_output = self.dense_out(hidden_states)
            return dense_output

class TransformerLayer_modified(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.idx = idx
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        
        if cfg_arch.norm in ["Approx_LayerNorm"]:
            if idx == 0:
                div_max_1 = 10
                div_max_2 = 4000
                self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
            elif idx == 1:
                div_max_1 = 5000
                div_max_2 = 4000
                self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
            else:
                div_max_1 = 8000
                div_max_2 = 8000
                self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
                
        else:
            self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
            self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent_modified(
            idx,
            cfg_arch.hidden_size,
            # cfg_arch.attention,
            cfg_arch,
            cfg_arch.use_bias,
        )
        self.cfg_arch = cfg_arch
        self.LAYOUT = self.attn.LAYOUT
        
        default_ffn_component = FFNComponent

        n = 100

        ffn_component_mapping = {None: default_ffn_component}

        for i in range(1, n + 1):
            class_name = f"FFNComponent_SmallMatmul_{i}"
            if class_name in globals():
                ffn_component_mapping[i] = globals()[class_name]

        # cfg_arch.FFN_small 값에 따라 해당 클래스를 선택하여 초기화
        ffn_class = ffn_component_mapping.get(cfg_arch.FFN_small, default_ffn_component)

        self.ffn = ffn_class(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            self.cfg_arch.get_input_range,
            _get_nonlin_fn(cfg_arch.nonlin, cfg_arch.experiment_float64),
            cfg_arch.use_bias,
        )
        
    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        norm1_inputs = states
        after_norm1 = self.norm1(states)

        states2, matmul_result = self.attn(after_norm1, attention_mask)
        states2 = self.dropout(states2)

        states = states + states2
        norm2_inputs = states

        after_norm2 = self.norm2(states)
        if self.cfg_arch.get_input_range:
            states2, nonlin_inputs = self.ffn(after_norm2)
        else:
            states2 = self.ffn(after_norm2)

        states2 = self.dropout(states2)

        states = states + states2

        if self.cfg_arch.get_input_range:
            return states, matmul_result, norm1_inputs, norm2_inputs, nonlin_inputs
        else:
            return states, matmul_result

class ScriptableLM_modified(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.cfg.embedding.get_emb_input_range = self.cfg.get_input_range
        
        if self.cfg.larger_embedding:
            self.emb_to_hidden = torch.nn.Linear(self.cfg.larger_embedding_dim, self.cfg.hidden_size, bias=self.cfg.use_bias)
            self.real_emb_dim = self.cfg.larger_embedding_dim
            self.hidden_to_emb = torch.nn.Linear(self.cfg.hidden_size, self.cfg.larger_embedding_dim)
        else:
            self.real_emb_dim = self.cfg.hidden_size

        self.embedding = EmbeddingComponent_modified(self.cfg, self.cfg.norm, self.cfg.norm_eps)
        
        if self.cfg.FFN_combined:
            self.layers = torch.nn.ModuleList([TransformerLayer_Combined(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        elif self.cfg.FFN_combined_ver2:
            self.layers = torch.nn.ModuleList([TransformerLayer_Combined_ver2(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        elif self.cfg.FFN_combined_ver3:
            self.layers = torch.nn.ModuleList([TransformerLayer_Combined_ver3(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        else:
            self.layers = torch.nn.ModuleList([TransformerLayer_modified(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
         
        self.seq_first = True
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            if self.cfg.norm in ["Approx_LayerNorm"]:
                div_max = 8000
                self.final_norm = _get_norm_fn(self.cfg.norm)(self.real_emb_dim, div_max=div_max, eps=self.cfg.norm_eps)
            else:
                self.final_norm = _get_norm_fn(self.cfg.norm)(self.real_emb_dim, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        matmuls = []
        tf_norm1_inputs_list = []
        tf_norm2_inputs_list = []
        nonlin_inputs_list = []
        if self.cfg.distillation:
            attentions_before_exp = []
            hidden_states_of_all_layers = []

        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
       
        if self.cfg.embedding.get_emb_input_range:
            hidden_states, emb_norm_inputs = self.embedding(input_ids)
        else:
            hidden_states = self.embedding(input_ids)

        # b, sq, d
        embeddings = hidden_states
        
        if self.seq_first:
            # sq, b, d
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        
        if self.cfg.larger_embedding:
            # sq, b, D --> sq, b, d
            hidden_states = self.emb_to_hidden(hidden_states)
        
        for i, layer_module in enumerate(self.layers):
            if self.cfg.get_input_range:
                hidden_states, matmul, tf_norm1_inputs, tf_norm2_inputs, nonlin_inputs = layer_module(hidden_states, attention_mask)
                tf_norm1_inputs_list.append(tf_norm1_inputs)
                tf_norm2_inputs_list.append(tf_norm2_inputs)
                nonlin_inputs_list.append(nonlin_inputs)
            else:
                hidden_states, matmul = layer_module(hidden_states, attention_mask)
            
            if self.cfg.distillation:
                attentions_before_exp.append(matmul)
                hidden_states_of_all_layers.append(hidden_states)
            
            matmuls.append(matmul)
        
        if self.cfg.larger_embedding:
            # sq, b, d --> sq, b, D
            hidden_states = self.hidden_to_emb(hidden_states)
            
        if self.seq_first:
            # sq, b, d --> b, sq, d
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        final_norm_inputs = hidden_states
        hidden_states = self.final_norm(hidden_states)
        
        if self.cfg.get_input_range:
            return hidden_states, matmuls, emb_norm_inputs, tf_norm1_inputs_list, tf_norm2_inputs_list, final_norm_inputs, nonlin_inputs_list
        if self.cfg.distillation:
            return hidden_states, matmuls, embeddings, attentions_before_exp, hidden_states_of_all_layers
        else:
            return hidden_states, matmuls
        
class ScriptableLMForPreTraining_modified(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM_modified(config)

        if not self.cfg.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        if self.cfg.larger_embedding:
            self.cfg.real_emb_dim = self.cfg.larger_embedding_dim
        else:
            self.cfg.real_emb_dim = self.cfg.embedding.embedding_dim
        
        self.decoder = torch.nn.Linear(self.cfg.real_emb_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction

        self._init_weights()
        
        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_graph_interval_loss_list = []
        self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.matmul_norm_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.matmul_norm_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0
        # self.emb_norm_inputs_var_norms = []
        self.emb_norm_inputs_var_maxs = []
        self.emb_norm_inputs_var_mins = []
        self.emb_norm_inputs_var_ratios = []
        # self.tf_norm1_inputs_var_norms = [[] for _ in range(self.cfg.num_transformer_layers)]
        # self.tf_norm2_inputs_var_norms = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        # self.final_norm_inputs_var_norms = []
        self.final_norm_inputs_var_maxs = []
        self.final_norm_inputs_var_mins = []
        self.final_norm_inputs_var_ratios = []
        self.nonlin_inputs_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.nonlin_inputs_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        if self.cfg.get_input_range:
            os.makedirs('norms', exist_ok=True)
        os.makedirs('loss', exist_ok=True)
        os.makedirs('after_norm_penalty', exist_ok=True)
        square_layer = math.floor(math.sqrt(self.cfg.num_transformer_layers))
        if square_layer ** 2 >= self.cfg.num_transformer_layers:
            self.vertical_num = square_layer
            self.horizontal_num = square_layer
        elif square_layer * (square_layer+1) >= self.cfg.num_transformer_layers:
            self.vertical_num = square_layer
            self.horizontal_num = square_layer + 1
        else:
            self.vertical_num = square_layer + 1
            self.horizontal_num = square_layer + 1
        
    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            # print(f'module: {module}')
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        matmul_sup_list = []
        before_att_var_max_list = []
        before_FFN_var_max_list = []
        before_att_var_ratio_list = []
        before_FFN_var_ratio_list = []
        self.count += 1
        self.x_list.append(self.count)

        if self.cfg.get_input_range:
            outputs, matmuls_from_enc, emb_norm_inputs, tf_norm1_inputs, tf_norm2_inputs, final_norm_inputs, nonlin_inputs = self.encoder(input_ids, attention_mask)

            # Embedding Norm Input Variances
            mean = emb_norm_inputs.mean(dim=-1, keepdim=True)
            var = ((emb_norm_inputs - mean) ** 2).mean(dim=-1, keepdim=True)
            emb_var_max = torch.max(var)
            emb_var_min = torch.min(var)
            emb_var_ratio = emb_var_max / emb_var_min
            # self.emb_norm_inputs_var_norms.append(emb_norm_inputs_var_norm.item())
            self.emb_norm_inputs_var_maxs.append(emb_var_max.item())
            self.emb_norm_inputs_var_mins.append(emb_var_min.item())
            self.emb_norm_inputs_var_ratios.append(emb_var_ratio.item())
            
            for i in range(self.cfg.num_transformer_layers):
                # Input Variances of Norm Before Attention
                mean = tf_norm1_inputs[i].mean(dim=-1, keepdim=True)
                var = ((tf_norm1_inputs[i] - mean) ** 2).mean(dim=-1, keepdim=True)
                var_max = torch.max(var)
                var_min = torch.min(var)
                var_ratio = var_max / var_min
                before_att_var_max_list.append(var_max)
                before_att_var_ratio_list.append(var_ratio)
                self.tf_norm1_inputs_var_maxs[i].append(var_max.item())
                self.tf_norm1_inputs_var_mins[i].append(var_min.item())
                self.tf_norm1_inputs_var_ratios[i].append(var_ratio.item())
                
                # Input Variances of Norm Before FFN
                mean = tf_norm2_inputs[i].mean(dim=-1, keepdim=True)
                var = ((tf_norm2_inputs[i] - mean) ** 2).mean(dim=-1, keepdim=True)
                var_max = torch.max(var)
                var_min = torch.min(var)
                var_ratio = var_max / var_min
                before_FFN_var_max_list.append(var_max)
                before_FFN_var_ratio_list.append(var_ratio)
                self.tf_norm2_inputs_var_maxs[i].append(var_max.item())
                self.tf_norm2_inputs_var_mins[i].append(var_min.item())
                self.tf_norm2_inputs_var_ratios[i].append(var_ratio.item())
            
                # Inputs of Non-linear function
                nonlin_inputs_max = torch.max(nonlin_inputs[i]).detach().cpu()
                nonlin_inputs_min = torch.min(nonlin_inputs[i]).detach().cpu()
                self.nonlin_inputs_maxs[i].append(nonlin_inputs_max.item())
                self.nonlin_inputs_mins[i].append(nonlin_inputs_min.item())
            
                # Inputs of exp
                matmul_max = torch.max(matmuls_from_enc[i]).detach().cpu()
                matmul_min = torch.min(matmuls_from_enc[i]).detach().cpu()
                matmul_sup_list.append(-matmul_min)
                self.matmul_norm_maxs[i].append(matmul_max.item())
                self.matmul_norm_mins[i].append(matmul_min.item())
            
            # Inputs of Final Norm        
            mean = final_norm_inputs.mean(dim=-1, keepdim=True)
            var = ((final_norm_inputs - mean) ** 2).mean(dim=-1, keepdim=True)
            final_var_max = torch.max(var)
            final_var_min = torch.min(var)
            final_var_ratio = final_var_max / final_var_min
            self.final_norm_inputs_var_maxs.append(final_var_max.item())
            self.final_norm_inputs_var_mins.append(final_var_min.item())
            self.final_norm_inputs_var_ratios.append(final_var_ratio.item())
           
            if (self.count % self.cfg.graph_interval == 0) or (self.count  == self.cfg.full_steps): 
                # Embedding Variance
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.emb_norm_inputs_var_maxs[-self.cfg.graph_interval:])
                plt.title(f'Max of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'norms/max_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.emb_norm_inputs_var_mins[-self.cfg.graph_interval:])
                plt.title(f'Min of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'norms/min_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list[-self.cfg.graph_interval:], self.emb_norm_inputs_var_ratios[-self.cfg.graph_interval:])
                # plt.title(f'Ratio of Min/Max Variances of Embedding {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Max/Min')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'norms/ratio_of_variances_of_emb_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'norms/variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.emb_norm_inputs_var_maxs[-self.cfg.graph_interval:])}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.emb_norm_inputs_var_mins[-self.cfg.graph_interval:])}\n\n')
                # with open(f'norms/ratio_of_variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.emb_norm_inputs_var_ratios[-self.cfg.graph_interval:])}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.emb_norm_inputs_var_ratios[-self.cfg.graph_interval:])}\n\n')
                
                # Before Attention
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm1_inputs_var_maxs[i][-self.cfg.graph_interval:])
                    plt.title(f'Max of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/max_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm1_inputs_var_mins[i][-self.cfg.graph_interval:])
                    plt.title(f'Min of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/min_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm1_inputs_var_ratios[i][-self.cfg.graph_interval:])
                #     plt.title(f'Ratio of Max/Min Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'norms/ratio_of_of_variances_of_{self.cfg.norm}_before_attention.png')
                # plt.clf()
                with open(f'norms/variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm1_inputs_var_maxs[i][-self.cfg.graph_interval:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm1_inputs_var_mins[i][-self.cfg.graph_interval:])}\n')
                # with open(f'norms/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm1_inputs_var_ratios[i][-self.cfg.graph_interval:])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm1_inputs_var_ratios[i][-self.cfg.graph_interval:])}\n')
        
                # Before FFN
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm2_inputs_var_maxs[i][-self.cfg.graph_interval:])
                    plt.title(f'Max of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/max_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm2_inputs_var_mins[i][-self.cfg.graph_interval:])
                    plt.title(f'Min of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/min_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm2_inputs_var_ratios[i][-self.cfg.graph_interval:])
                #     plt.title(f'Ratio of Max/Min Vars of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'norms/ratio_of_of_variances_of_{self.cfg.norm}_before_ffn.png')
                # plt.clf()
                with open(f'norms/variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm2_inputs_var_maxs[i][-self.cfg.graph_interval:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm2_inputs_var_mins[i][-self.cfg.graph_interval:])}\n')
                # with open(f'norms/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm2_inputs_var_ratios[i][-self.cfg.graph_interval:])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm2_inputs_var_ratios[i][-self.cfg.graph_interval:])}\n')
        
                # Final Normalization
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.final_norm_inputs_var_maxs[-self.cfg.graph_interval:])
                plt.title(f'Max of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.savefig(f'norms/max_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.final_norm_inputs_var_mins[-self.cfg.graph_interval:])
                plt.title(f'Min of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'norms/min_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list[-self.cfg.graph_interval:], self.final_norm_inputs_var_ratios[-self.cfg.graph_interval:])
                # plt.title(f'Ratio of Max/Min Variances of Final {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Min/Max')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'norms/ratio_of_variances_of_final_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'norms/variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'Variances of Inputs of Final {self.cfg.norm}\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.final_norm_inputs_var_maxs[-self.cfg.graph_interval:])}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.final_norm_inputs_var_mins[-self.cfg.graph_interval:])}\n\n')
                # with open(f'norms/ratio_of_variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'Ratio of Min/Max Variances of Inputs of Final {self.cfg.norm}\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.final_norm_inputs_var_ratios[-self.cfg.graph_interval:])}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.final_norm_inputs_var_ratios[-self.cfg.graph_interval:])}\n\n')
                
                # Non-lin Inputs
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.nonlin_inputs_maxs[i][-self.cfg.graph_interval:])
                    plt.title(f'Max of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/max_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.nonlin_inputs_mins[i][-self.cfg.graph_interval:])
                    plt.title(f'Min of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/min_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                with open(f'norms/inputs_of_{self.cfg.nonlin}.txt', 'w') as file:
                    file.write(f'Inputs of {self.cfg.nonlin}\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.nonlin_inputs_maxs[i][-self.cfg.graph_interval:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.nonlin_inputs_mins[i][-self.cfg.graph_interval:])}\n')
            
            # Graph after norm-penalty steps
            if self.count  == self.cfg.full_steps and self.cfg.full_steps > self.cfg.penalty_step: 
                last_graph_steps = self.cfg.full_steps - self.cfg.penalty_step
                print(f'last_graph_steps: {last_graph_steps}')
                # Embedding Variance
                plt.plot(self.x_list[-last_graph_steps:], self.emb_norm_inputs_var_maxs[-last_graph_steps:])
                plt.title(f'After Norm-penalty Max of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'after_norm_penalty/max_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list[-last_graph_steps:], self.emb_norm_inputs_var_mins[-last_graph_steps:])
                plt.title(f'After Norm-penalty Min of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'after_norm_penalty/min_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list[-last_graph_steps:], self.emb_norm_inputs_var_ratios[-last_graph_steps:])
                # plt.title(f'Ratio of Min/Max Variances of Embedding {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Max/Min')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'after_norm_penalty/ratio_of_variances_of_emb_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'after_norm_penalty/variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'After Norm-penalty Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.emb_norm_inputs_var_maxs[-last_graph_steps:])}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.emb_norm_inputs_var_mins[-last_graph_steps:])}\n\n')
                # with open(f'after_norm_penalty/ratio_of_variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'After Norm-penalty Ratio of Max/Min Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.emb_norm_inputs_var_ratios[-last_graph_steps:])}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.emb_norm_inputs_var_ratios[-last_graph_steps:])}\n\n')
                
                # Before Attention
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.tf_norm1_inputs_var_maxs[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Max of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/max_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.tf_norm1_inputs_var_mins[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Min of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/min_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list[-last_graph_steps:], self.tf_norm1_inputs_var_ratios[i][-last_graph_steps:])
                #     plt.title(f'After Norm-penalty Ratio of Max/Min Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'after_norm_penalty/ratio_of_of_variances_of_{self.cfg.norm}_before_attention.png')
                # plt.clf()
                with open(f'after_norm_penalty/variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                    file.write(f'After Norm-penalty Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm1_inputs_var_maxs[i][-last_graph_steps:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm1_inputs_var_mins[i][-last_graph_steps:])}\n')
                # with open(f'after_norm_penalty/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                #     file.write(f'After Norm-penalty Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm1_inputs_var_ratios[i][-last_graph_steps:])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm1_inputs_var_ratios[i][-last_graph_steps:])}\n')
        
                # Before FFN
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.tf_norm2_inputs_var_maxs[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Max of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/max_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.tf_norm2_inputs_var_mins[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Min of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/min_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list[-last_graph_steps:], self.tf_norm2_inputs_var_ratios[i][-last_graph_steps:])
                #     plt.title(f'After Norm-penalty Ratio of Max/Min Vars of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'after_norm_penalty/ratio_of_of_variances_of_{self.cfg.norm}_before_ffn.png')
                # plt.clf()
                with open(f'after_norm_penalty/variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                    file.write(f'After Norm-penalty Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm2_inputs_var_maxs[i][-last_graph_steps:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm2_inputs_var_mins[i][-last_graph_steps:])}\n')
                # with open(f'after_norm_penalty/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                #     file.write(f'After Norm-penaltyRatio of Max/Min Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm2_inputs_var_ratios[i][-last_graph_steps:])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm2_inputs_var_ratios[i][-last_graph_steps:])}\n')
        
                # Final Normalization
                plt.plot(self.x_list[-last_graph_steps:], self.final_norm_inputs_var_maxs[-last_graph_steps:])
                plt.title(f'After Norm-penalty Max of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.savefig(f'after_norm_penalty/max_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list[-last_graph_steps:], self.final_norm_inputs_var_mins[-last_graph_steps:])
                plt.title(f'After Norm-penalty Min of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'after_norm_penalty/min_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list[-last_graph_steps:], self.final_norm_inputs_var_ratios[-last_graph_steps:])
                # plt.title(f'After Norm-penalty Ratio of Max/Min Variances of Final {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Min/Max')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'after_norm_penalty/ratio_of_variances_of_final_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'after_norm_penalty/variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'After Norm-penalty Variances of Inputs of Final {self.cfg.norm}\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.final_norm_inputs_var_maxs[-last_graph_steps:])}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.final_norm_inputs_var_mins[-last_graph_steps:])}\n\n')
                # with open(f'after_norm_penalty/ratio_of_variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'After Norm-penalty Ratio of Min/Max Variances of Inputs of Final {self.cfg.norm}\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.final_norm_inputs_var_ratios[-last_graph_steps:])}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.final_norm_inputs_var_ratios[-last_graph_steps:])}\n\n')
                
                # Non-lin Inputs
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.nonlin_inputs_maxs[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Max of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/max_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.nonlin_inputs_mins[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Min of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/min_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                with open(f'after_norm_penalty/inputs_of_{self.cfg.nonlin}.txt', 'w') as file:
                    file.write(f'After Norm-penalty Inputs of {self.cfg.nonlin}\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.nonlin_inputs_maxs[i][-last_graph_steps:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.nonlin_inputs_mins[i][-last_graph_steps:])}\n')
        
        # outputs: b, sq, d                
        elif self.cfg.distillation:
            outputs, matmuls_from_enc, embeddings, attentions_before_exp, hidden_states_of_all_layers = self.encoder(input_ids, attention_mask)
        else:
            outputs, matmuls_from_enc = self.encoder(input_ids, attention_mask)
        
        outputs = outputs.view(-1, outputs.shape[-1]) # 여기

        if self.sparse_prediction and labels is not None:            
            masked_lm_loss = self._forward_sparse(outputs, labels) ### loss 여기
            original_loss = masked_lm_loss.item()
            
            self.loss_list.append(original_loss)
            

            # Add a loss
            #####################################
            if self.count > self.cfg.penalty_step:
                if self.cfg.var_ratio_penalty:
                    if emb_var_ratio > self.cfg.var_ratio_penalty_scale:
                        masked_lm_loss += self.cfg.norm_penalty_coeff * emb_var_ratio
                    if final_var_ratio > self.cfg.var_ratio_penalty_scale:
                        masked_lm_loss += self.cfg.norm_penalty_coeff * final_var_ratio
                    for i in range(self.cfg.num_transformer_layers):
                        if before_att_var_ratio_list[i].item() > self.cfg.var_ratio_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * before_att_var_ratio_list[i]
                        if before_FFN_var_ratio_list[i].item() > self.cfg.var_ratio_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * before_FFN_var_ratio_list[i]
                    
                if self.cfg.matmul_range_penalty:
                    for i in range(self.cfg.num_transformer_layers):
                        if matmul_sup_list[i].item() > self.cfg.matmul_norm_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * matmul_sup_list[i]
                
                if self.cfg.max_var_penalty:
                    if emb_var_max.item() > self.cfg.var_max_penalty_scale:
                       masked_lm_loss += self.cfg.norm_penalty_coeff * emb_var_max
                    for i in range(self.cfg.num_transformer_layers):
                        if before_att_var_max_list[i].item() > self.cfg.var_max_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * before_att_var_max_list[i]
                    for i in range(self.cfg.num_transformer_layers):
                        if before_FFN_var_max_list[i].item() > self.cfg.var_max_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * before_FFN_var_max_list[i]
                    if final_var_max.item() > self.cfg.var_max_penalty_scale:
                        masked_lm_loss += self.cfg.norm_penalty_coeff * final_var_max
            #####################################
                            
            # if self.count < self.cfg.graph_interval:
            if self.count < 100:
                last_graph_interval_loss = sum(self.loss_list) / len(self.loss_list)
                self.last_graph_interval_loss_list.append(last_graph_interval_loss)
                # print(f'Loss: {original_loss}, Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
                print(f'Loss: {original_loss:8.6f}, Last_{100}_losses: {last_graph_interval_loss:8.6f}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
            else:
                # last_graph_interval_loss = sum(self.loss_list[-self.cfg.graph_interval :]) / len(self.loss_list[-self.cfg.graph_interval :])
                last_graph_interval_loss = sum(self.loss_list[-100 :]) / len(self.loss_list[-100 :])
                self.last_graph_interval_loss_list.append(last_graph_interval_loss)
                if self.best_loss == 0 or last_graph_interval_loss < self.best_loss:
                    self.best_loss = last_graph_interval_loss
                # print(f'Loss: {original_loss}, Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}, Best_{self.cfg.graph_interval}_losses: {self.best_loss}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
                if self.count % 1000 == 0:
                    log.info(f'Loss: {original_loss:8.6f}, Last_{100}_losses: {last_graph_interval_loss:8.6f}, Best_{self.cfg.graph_interval}_losses: {self.best_loss:8.6f}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
                else:
                    print(f'Loss: {original_loss:8.6f}, Last_{100}_losses: {last_graph_interval_loss:8.6f}, Best_{self.cfg.graph_interval}_losses: {self.best_loss:8.6f}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
            
            # Losses and Inputs of Exponential                
            if (self.count % self.cfg.graph_interval == 0) or (self.count  == self.cfg.full_steps):
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.loss_list[-self.cfg.graph_interval:])
                plt.title('Loss', fontsize=10)
                plt.xlabel('Steps', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.savefig('loss/losses.png')
                plt.clf()
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.last_graph_interval_loss_list[-self.cfg.graph_interval:])
                plt.title(f'Last {100} losses', fontsize=10)
                plt.xlabel('Steps', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.savefig(f'loss/last_{100}_losses.png')
                plt.clf()
                if self.cfg.get_input_range:
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list[-self.cfg.graph_interval:], self.matmul_norm_maxs[i][-self.cfg.graph_interval:])
                        plt.title(f'Max of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Max', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'norms/max_of_inputs_of_exp.png')
                    plt.clf()
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list[-self.cfg.graph_interval:], self.matmul_norm_mins[i][-self.cfg.graph_interval:])
                        plt.title(f'Min of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Min', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'norms/min_of_inputs_of_exp.png')
                    plt.clf()
                    with open(f'norms/inputs_of_exp.txt', 'w') as file:
                        file.write(f'Inputs of exp\n\n')
                        file.write(f'Max\n\n')
                        for i in range(self.cfg.num_transformer_layers):
                            file.write(f'{max(self.matmul_norm_maxs[i][-self.cfg.graph_interval:])}\n')
                        file.write('\n')
                        file.write(f'Min\n\n')
                        for i in range(self.cfg.num_transformer_layers):
                            file.write(f'{min(self.matmul_norm_mins[i][-self.cfg.graph_interval:])}\n')
        
            # Graph after norm-penalty steps                
            if self.count == self.cfg.full_steps and self.cfg.full_steps > self.cfg.penalty_step:
                plt.plot(self.x_list[-last_graph_steps:], self.loss_list[-last_graph_steps:])
                plt.title('After Norm-Penalty Loss', fontsize=10)
                plt.xlabel('Steps', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.savefig('after_norm_penalty/losses.png')
                plt.clf()
                if self.cfg.get_input_range:
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list[-last_graph_steps:], self.matmul_norm_maxs[i][-last_graph_steps:])
                        plt.title(f'After Norm-penalty Max of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Max', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'after_norm_penalty/max_of_inputs_of_exp.png')
                    plt.clf()
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list[-last_graph_steps:], self.matmul_norm_mins[i][-last_graph_steps:])
                        plt.title(f'After Norm-penalty Min of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Min', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'after_norm_penalty/min_of_inputs_of_exp.png')
                    plt.clf()
                with open(f'after_norm_penalty/inputs_of_exp.txt', 'w') as file:
                    file.write(f'After Norm-penalty Inputs of exp\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.matmul_norm_maxs[i][-last_graph_steps:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.matmul_norm_mins[i][-last_graph_steps:])}\n')
                with open(f'results.txt', 'w') as file:
                    file.write(f'Loss: {original_loss}\n\n')
                    # file.write(f'Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}\n\n')
                    file.write(f'Last_{100}_losses: {last_graph_interval_loss}\n\n')
                    # file.write(f'Best_{self.cfg.graph_interval}_losses: {self.best_loss}\n\n')
                    file.write(f'Best_{100}_losses: {self.best_loss}\n\n')
                    file.write(f'Layers: {self.cfg.num_transformer_layers}\n\n')
                    file.write(f'Count: {self.count}\n\n')
            
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))
        if self.cfg.distillation:
            return {"loss": masked_lm_loss, "outputs": outputs}, embeddings, attentions_before_exp, hidden_states_of_all_layers
        else:
            return {"loss": masked_lm_loss, "outputs": outputs}

    # Sparse prediction usually has an unpredictable number of entries in each batch
    # but the dataloader was modified so that 25% of the batch is ALWAYS masked.
    # This allows for static compilation. If you modify the dataloader, this function will fill your compile cache
    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        labels = labels.view(-1)
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]  # ugh
        outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        labels = labels[indices]
        outputs = self.decoder(self.prediction_head(outputs)) # Prediction_head: Identity()
        masked_lm_loss = self.loss_fn(outputs, labels)
        return masked_lm_loss

class ScriptableLMForSequenceClassification_modified(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels
        self.cfg.classification_head.experiment_float64 = self.cfg.experiment_float64
        
        self.temperature = 1
        self.cfg.classification_head.get_input_range = self.cfg.get_input_range
        if not self.cfg.get_grad == None:
            self.cfg.classification_head.get_grad = self.cfg.get_grad

        self.encoder = ScriptableLM_modified(config)
        if self.cfg.larger_embedding:
            self.cfg.hidden_size = self.cfg.larger_embedding_dim
        # self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.pooler = PoolingComponent_lora(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_graph_interval_loss_list = []
        # self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.matmul_norm_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.matmul_norm_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0
        self.emb_norm_inputs_var_maxs = []
        self.emb_norm_inputs_var_mins = []
        self.emb_norm_inputs_var_ratios = []
        # self.tf_norm1_inputs_var_norms = [[] for _ in range(self.cfg.num_transformer_layers)]
        # self.tf_norm2_inputs_var_norms = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        # self.final_norm_inputs_var_norms = []
        self.final_norm_inputs_var_maxs = []
        self.final_norm_inputs_var_mins = []
        self.final_norm_inputs_var_ratios = []
        self.nonlin_inputs_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.nonlin_inputs_mins = [[] for _ in range(self.cfg.num_transformer_layers)]       
        self.act_ftn_inputs_maxs = []
        self.act_ftn_inputs_mins = []
        
        os.makedirs(self.cfg.task_name, exist_ok=True)
        self.act_ftn_path = os.path.join(self.cfg.task_name, 'activation_ftn')
        os.makedirs(self.act_ftn_path, exist_ok=True)
        if self.cfg.get_input_range:
            self.norm_path = os.path.join(self.cfg.task_name, 'norms')
            os.makedirs(self.norm_path, exist_ok=True)
        self.loss_path = os.path.join(self.cfg.task_name, 'loss')
        os.makedirs(self.loss_path, exist_ok=True)
        
        # print(f'self.cfg.num_transformer_layers: {self.cfg.num_transformer_layers}')
        square_layer = math.floor(math.sqrt(self.cfg.num_transformer_layers))        
        if square_layer ** 2 >= self.cfg.num_transformer_layers:
            self.vertical_num = square_layer
            self.horizontal_num = square_layer
        elif square_layer * (square_layer+1) >= self.cfg.num_transformer_layers:
            self.vertical_num = square_layer
            self.horizontal_num = square_layer + 1
        else:
            self.vertical_num = square_layer + 1
            self.horizontal_num = square_layer + 1

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        matmul_sup_list = []
        before_att_var_max_list = []
        before_FFN_var_max_list = []
        before_att_var_ratio_list = []
        before_FFN_var_ratio_list = []
              
        self.count += 1
        self.x_list.append(self.count)  
        
        if self.cfg.get_input_range:
            encoder_output,  matmuls_from_enc, emb_norm_inputs, tf_norm1_inputs, tf_norm2_inputs, final_norm_inputs, nonlin_inputs = self.encoder(input_ids, attention_mask)
        else:
            encoder_output,  matmuls_from_enc = self.encoder(input_ids, attention_mask)
        if self.cfg.get_input_range:
            pooler_output, act_ftn_inputs = self.pooler(encoder_output)
        elif self.cfg.get_grad:
            pooler_output, before_zero_indexing_hidden_states, first_token_tensor = self.pooler(encoder_output)
        else:
            pooler_output = self.pooler(encoder_output)
        logits = self.head(pooler_output)
        
        if self.cfg.get_input_range:
            # Embedding Norm Input Variances
            mean = emb_norm_inputs.mean(dim=-1, keepdim=True)
            var = ((emb_norm_inputs - mean) ** 2).mean(dim=-1, keepdim=True)
            emb_var_max = torch.max(var).detach().cpu()
            emb_var_min = torch.min(var).detach().cpu()
            emb_var_ratio = emb_var_max / emb_var_min
            self.emb_norm_inputs_var_maxs.append(emb_var_max.item())
            self.emb_norm_inputs_var_mins.append(emb_var_min.item())
            self.emb_norm_inputs_var_ratios.append(emb_var_ratio.item())
            
            for i in range(self.cfg.num_transformer_layers):
                # Input Variances of Norm Before Attention
                mean = tf_norm1_inputs[i].mean(dim=-1, keepdim=True)
                var = ((tf_norm1_inputs[i] - mean) ** 2).mean(dim=-1, keepdim=True)
                var_max = torch.max(var).detach().cpu()
                var_min = torch.min(var).detach().cpu()
                var_ratio = var_max / var_min
                before_att_var_max_list.append(var_max)
                before_att_var_ratio_list.append(var_ratio)
                self.tf_norm1_inputs_var_maxs[i].append(var_max.item())
                self.tf_norm1_inputs_var_mins[i].append(var_min.item())
                self.tf_norm1_inputs_var_ratios[i].append(var_ratio.item())
                
                # Input Variances of Norm Before FFN
                mean = tf_norm2_inputs[i].mean(dim=-1, keepdim=True)
                var = ((tf_norm2_inputs[i] - mean) ** 2).mean(dim=-1, keepdim=True)
                var_max = torch.max(var).detach().cpu()
                var_min = torch.min(var).detach().cpu()
                var_ratio = var_max / var_min
                before_FFN_var_max_list.append(var_max)
                before_FFN_var_ratio_list.append(var_ratio)
                self.tf_norm2_inputs_var_maxs[i].append(var_max.item())
                self.tf_norm2_inputs_var_mins[i].append(var_min.item())
                self.tf_norm2_inputs_var_ratios[i].append(var_ratio.item())
            
                # Inputs of Non-linear function
                nonlin_inputs_max = torch.max(nonlin_inputs[i]).detach().cpu()
                nonlin_inputs_min = torch.min(nonlin_inputs[i]).detach().cpu()
                self.nonlin_inputs_maxs[i].append(nonlin_inputs_max.item())
                self.nonlin_inputs_mins[i].append(nonlin_inputs_min.item())
            
                # Inputs of exp
                matmul_max = torch.max(matmuls_from_enc[i]).detach().cpu()
                matmul_min = torch.min(matmuls_from_enc[i])
                matmul_sup_list.append(-matmul_min)
                self.matmul_norm_maxs[i].append(matmul_max.item())
                self.matmul_norm_mins[i].append(matmul_min.item())
            
            # Inputs of Final Norm        
            mean = final_norm_inputs.mean(dim=-1, keepdim=True)
            var = ((final_norm_inputs - mean) ** 2).mean(dim=-1, keepdim=True)
            final_var_max = torch.max(var).detach().cpu()
            final_var_min = torch.min(var).detach().cpu()
            final_var_ratio = final_var_max / final_var_min
            self.final_norm_inputs_var_maxs.append(final_var_max.item())
            self.final_norm_inputs_var_mins.append(final_var_min.item())
            self.final_norm_inputs_var_ratios.append(final_var_ratio.item())
            
            if self.count % self.cfg.eval_graph_interval == 0:
                plt.plot(self.x_list, self.emb_norm_inputs_var_maxs)
                plt.title(f'Max of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.norm_path}/max_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list, self.emb_norm_inputs_var_mins)
                plt.title(f'Min of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.norm_path}/min_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list, self.emb_norm_inputs_var_ratios)
                # plt.title(f'Ratio of Min/Max Variances of Embedding {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Max/Min')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'{self.norm_path}/ratio_of_variances_of_emb_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'{self.norm_path}/variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.emb_norm_inputs_var_maxs)}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.emb_norm_inputs_var_mins)}\n\n')
                # with open(f'{self.norm_path}/ratio_of_variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.emb_norm_inputs_var_ratios)}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.emb_norm_inputs_var_ratios)}\n\n')
                
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.tf_norm1_inputs_var_maxs[i])
                    plt.title(f'Max of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/max_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.tf_norm1_inputs_var_mins[i])
                    plt.title(f'Min of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/min_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list, self.tf_norm1_inputs_var_ratios[i])
                #     plt.title(f'Ratio of Max/Min Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'{self.norm_path}/ratio_of_of_variances_of_{self.cfg.norm}_before_attention.png')
                # plt.clf()
                with open(f'{self.norm_path}/variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm1_inputs_var_maxs[i])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm1_inputs_var_mins[i])}\n')
                # with open(f'{self.norm_path}/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm1_inputs_var_ratios[i])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm1_inputs_var_ratios[i])}\n')
        
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.tf_norm2_inputs_var_maxs[i])
                    plt.title(f'Max of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/max_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.tf_norm2_inputs_var_mins[i])
                    plt.title(f'Min of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/min_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list, self.tf_norm2_inputs_var_ratios[i])
                #     plt.title(f'Ratio of Max/Min Vars of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'{self.norm_path}/ratio_of_of_variances_of_{self.cfg.norm}_before_ffn.png')
                # plt.clf()
                with open(f'{self.norm_path}/variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm2_inputs_var_maxs[i])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm2_inputs_var_mins[i])}\n')
                # with open(f'{self.norm_path}/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm2_inputs_var_ratios[i])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm2_inputs_var_ratios[i])}\n')
        
                # Final Normalization
                plt.plot(self.x_list, self.final_norm_inputs_var_maxs)
                plt.title(f'Max of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.savefig(f'{self.norm_path}/max_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list, self.final_norm_inputs_var_mins)
                plt.title(f'Min of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.norm_path}/min_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list, self.final_norm_inputs_var_ratios)
                # plt.title(f'Ratio of Max/Min Variances of Final {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Min/Max')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'{self.norm_path}/ratio_of_variances_of_final_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'{self.norm_path}/variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'Variances of Inputs of Final {self.cfg.norm}\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.final_norm_inputs_var_maxs)}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.final_norm_inputs_var_mins)}\n\n')
                # with open(f'{self.norm_path}/ratio_of_variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'Ratio of Min/Max Variances of Inputs of Final {self.cfg.norm}\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.final_norm_inputs_var_ratios)}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.final_norm_inputs_var_ratios)}\n\n')
                                
                # Non-lin Inputs
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.nonlin_inputs_maxs[i])
                    plt.title(f'Max of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/max_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.nonlin_inputs_mins[i])
                    plt.title(f'Min of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/min_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                with open(f'{self.norm_path}/inputs_of_{self.cfg.nonlin}.txt', 'w') as file:
                    file.write(f'Inputs of {self.cfg.nonlin}\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.nonlin_inputs_maxs[i])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.nonlin_inputs_mins[i])}\n')
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list, self.matmul_norm_maxs[i])
                        plt.title(f'Max of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Max', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'{self.norm_path}/max_of_inputs_of_exp.png')
                    plt.clf()
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list, self.matmul_norm_mins[i])
                        plt.title(f'Min of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Min', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'{self.norm_path}/min_of_inputs_of_exp.png')
                    plt.clf()
                with open(f'{self.norm_path}/inputs_of_exp.txt', 'w') as file:
                    file.write(f'Inputs of exp\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.matmul_norm_maxs[i])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.matmul_norm_mins[i])}\n')
        
        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"
            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = Custom_CrossEntropyLoss(temperature=self.temperature)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        self.loss_list.append(loss.item())
        
        # if self.count < self.cfg.eval_graph_interval:
        if self.count < 100:
            last_graph_interval_loss = sum(self.loss_list) / len(self.loss_list)
            self.last_graph_interval_loss_list.append(last_graph_interval_loss)
        else:
            # last_graph_interval_loss = sum(self.loss_list[-self.cfg.eval_graph_interval:]) / len(self.loss_list[-self.cfg.eval_graph_interval:])
            last_graph_interval_loss = sum(self.loss_list[-100:]) / len(self.loss_list[-100:])
            self.last_graph_interval_loss_list.append(last_graph_interval_loss)
            if self.best_loss == 0 or last_graph_interval_loss < self.best_loss:
                self.best_loss = last_graph_interval_loss
            
        # for i in range(self.cfg.num_transformer_layers):
        #     norm_i = torch.norm(matmuls_from_enc[i], p=float('inf'))
        #     # print('Matmul_{}: {}'.format(i, norm_i.item()))
        #     self.matmul_results[i].append(norm_i.item())
            
            # # Loss 추가
            # if norm_i > 450:
            #     loss += 0.1 * norm_i
            
        if self.count % self.cfg.eval_graph_interval == 0:
            plt.plot(self.x_list, self.loss_list)
            plt.title('Loss')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'loss', 'losses.png'))
            plt.clf()
            plt.plot(self.x_list, self.last_graph_interval_loss_list)
            # plt.title(f'Last {self.cfg.eval_graph_interval} losses')
            plt.title(f'Last {100} losses')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'loss', f'last_{100}_losses.png'))
            plt.clf()
            # for i in range(self.cfg.num_transformer_layers):
            #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
            #     plt.plot(self.x_list, self.matmul_results[i])
            #     plt.title('Matmul_{}'.format(i))
            #     plt.xlabel('Steps')
            # plt.savefig(os.path.join(self.cfg.task_name, 'matmuls.png'))
            # plt.clf()
        if self.cfg.get_input_range:
            act_ftn_inputs_max = torch.max(act_ftn_inputs).detach().cpu()
            act_ftn_inputs_min = torch.min(act_ftn_inputs).detach().cpu()
            self.act_ftn_inputs_maxs.append(act_ftn_inputs_max)
            self.act_ftn_inputs_mins.append(act_ftn_inputs_min)
            # print(f'self.x_list: {len(self.x_list)}')
            # print(f'self.act_ftn_inputs_maxs: {len(self.act_ftn_inputs_maxs)}')
            # print(f'Max of Inputs of {self.cfg.classification_head.nonlin}: {max(self.act_ftn_inputs_maxs[-self.cfg.eval_graph_interval:])}')
            # print(f'Min of Inputs of {self.cfg.classification_head.nonlin}: {min(self.act_ftn_inputs_mins[-self.cfg.eval_graph_interval:])}')
            if self.count % self.cfg.eval_graph_interval == 0:
                plt.plot(self.x_list, self.act_ftn_inputs_maxs)
                # print(f'len(self.x_list[-self.cfg.eval_graph_interval:]): {len(self.x_list[-self.cfg.eval_graph_interval:])}')
                # print(f'len(self.act_ftn_inputs_maxs[-self.cfg.eval_graph_interval:]): {len(self.act_ftn_inputs_maxs[-self.cfg.eval_graph_interval:])}')
                plt.title(f'Max of Inputs of {self.cfg.classification_head.nonlin}')
                plt.xlabel('Steps')
                plt.ylabel('Max')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.act_ftn_path}/max_of_inputs_of_{self.cfg.classification_head.nonlin}.png')
                plt.clf()
                plt.plot(self.x_list, self.act_ftn_inputs_mins)
                plt.title(f'Min of Inputs of {self.cfg.classification_head.nonlin}')
                plt.xlabel('Steps')
                plt.ylabel('Min')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.act_ftn_path}/min_of_inputs_of_{self.cfg.classification_head.nonlin}.png')
                plt.clf()
                with open(f'{self.act_ftn_path}/{self.cfg.classification_head.nonlin}.txt', 'w') as file:
                    file.write(f'{self.cfg.classification_head.nonlin}\n')
                    file.write(f'Max: {max(self.act_ftn_inputs_maxs)}\n')
                    file.write(f'Min: {min(self.act_ftn_inputs_mins)}\n')
        # if self.cfg.get_grad:
            # return dict(logits=logits, loss=loss), before_zero_indexing_hidden_states, first_token_tensor
        # else:
        return dict(logits=logits, loss=loss)

class ScriptableLMForPreTraining_modified_LoRA(PreTrainedModel):
    ...
    
class ScriptableLMForSequenceClassification_modified_LoRA(PreTrainedModel):
    ...