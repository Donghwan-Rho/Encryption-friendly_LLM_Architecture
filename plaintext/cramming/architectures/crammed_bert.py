"""This rewrite is a simplified version of the proposed changes that actually compiles statically in torch 2.0.

This model is the final, optimized crammed model.
OmegaConf
Not all ablations discussed in the paper are implemented as switches in this version,
for all those, check scriptable_bert.py on the old branch.

"""
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification

from typing import Optional
from omegaconf import OmegaConf

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent,
    PoolingComponent_lora,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
)
from .attention import get_attention_mechanism
import matplotlib.pyplot as plt
from termcolor import colored
import os
import math

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_crammed_bert(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    print('construct_crammed_bert')
    print('cfg_arch\n',cfg_arch)
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining(config)
        elif config.arch["objective_layout"] == "SCRIPT":
            model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model

class AttentionComponent(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # Ordinary
        #######################################
        output = self.self_attention(hidden_states, attention_mask)
        # print('att output', output.shape)
        output = self.dense(output)
        return output
        #######################################
        # # Ordinary-matmul
        # #######################################
        # output, matmul_result = self.self_attention(hidden_states, attention_mask)
        # # print('att output', output.shape)
        # output = self.dense(output)
        # return output, matmul_result
        # #######################################
        # # Heatmap
        # #######################################
        # output, matmul_result, heatmap_result = self.self_attention(hidden_states, attention_mask)
        # output = self.dense(output)
        # return output, matmul_result, heatmap_result
        # #######################################
        # # Q, K L2 normalization scheme
        # #######################################
        # # output, matmul_result, norm_sum = self.self_attention(hidden_states, attention_mask)
        # # output = self.dense(output)
        # # return output, matmul_result, norm_sum
        # output, matmul_result, query_norm, key_norm = self.self_attention(hidden_states, attention_mask)
        # output = self.dense(output)
        # return output, matmul_result, query_norm, key_norm
        # #######################################


class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    It actually turned out better not to scale it, so here the block is effectively smaller than may be expected.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        # print('self.nonlin', self.nonlin)
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)
        
    def forward(self, hidden_states):
        # Ordinary
        #######################################
        # if torch.isnan(self.dense_out(self.nonlin(self.dense_in(hidden_states)))).any():
        #     print('FFN nan')
        #     return
        return self.dense_out(self.nonlin(self.dense_in(hidden_states)))
        #######################################
        # # Before normalization / GELU checking
        # #######################################
        # dense_in_output = self.dense_in(hidden_states)
        # before_FFN_GELU_norm = torch.norm(dense_in_output, p=float('inf')).item()
        # # print('before_FFN_GELU_norm', before_FFN_GELU_norm.item())
        # return self.dense_out(self.nonlin(dense_in_output)), before_FFN_GELU_norm
        # #######################################


class TransformerLayer(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        # print('idx', idx) # 0, 1
        # print('cfg_arch.hidden_size', cfg_arch.hidden_size)
        # print('cfg_arch.attention', cfg_arch.attention)
        # print('cfg_arch.use_bias', cfg_arch.use_bias)
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.LAYOUT = self.attn.LAYOUT

        # GELU activation 조정
        self.ffn = FFNComponent(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            # cfg_arch.nonlin: GELUglu
            _get_nonlin_fn(cfg_arch.nonlin, cfg_arch.experiment_float64),
            cfg_arch.use_bias,
        )

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        # Ordinary
        #######################################
        states2 = self.attn(self.norm1(states), attention_mask)
        states = states + self.dropout(states2)
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states
        #######################################
        # # Ordinary-matmul
        # #######################################
        # states2, matmul_result = self.attn(self.norm1(states), attention_mask)
        # states = states + self.dropout(states2)
        # states = states + self.dropout(self.ffn(self.norm2(states)))
        # return states, matmul_result
        # #######################################
        # Before normalization / GELU checking
        # #######################################
        # before_layernorm_1 = torch.norm(states, p=float('inf')).item()
        # # print('before_layernorm_1', before_layernorm_1)
                
        # states2, matmul_result = self.attn(self.norm1(states), attention_mask)
        # states = states + self.dropout(states2)
        
        # before_layernorm_2 = torch.norm(states, p=float('inf')).item()
        # # print('before_layernorm_2', before_layernorm_2)
        
        # ffn_output, before_FFN_GELU_norm = self.ffn(self.norm2(states))
        # states = states + self.dropout(ffn_output)
        # return states, matmul_result, before_layernorm_1, before_layernorm_2, before_FFN_GELU_norm
        # #######################################
        # # Heatmap
        #######################################
        # states2, matmul_result, heatmap_result = self.attn(self.norm1(states), attention_mask)
        # states = states + self.dropout(states2)
        # states = states + self.dropout(self.ffn(self.norm2(states)))
        # return states, matmul_result, heatmap_result
        #######################################
        # # # Q, K L2 normalization scheme
        # #######################################
        # # states2, matmul_result, norm_sum = self.attn(self.norm1(states), attention_mask)
        # # states = states + self.dropout(states2)
        # # states = states + self.dropout(self.ffn(self.norm2(states)))
        # # return states, matmul_result, norm_sum
        # states2, matmul_result, query_norm, key_norm = self.attn(self.norm1(states), attention_mask)
        # states = states + self.dropout(states2)
        # states = states + self.dropout(self.ffn(self.norm2(states)))
        # return states, matmul_result, query_norm, key_norm
        # #######################################



class ScriptableLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        # print('self.seq_first', self.seq_first)
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        # Before normalization / GELU checking
        #######################################
        self.layernorms_1 = []
        self.layernorms_2 = []
        self.gelus = []
        #######################################
        
    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        self.layernorms_1 = []
        self.layernorms_2 = []
        self.gelus = []
    
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
        
        hidden_states = self.embedding(input_ids)
        
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        
        return self.final_norm(hidden_states)


class ScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)

        if not self.cfg.skip_head_transform:
            # print('not self.cfg.skip_head_transform:')
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            # print('not not self.cfg.skip_head_transform:')
            # 여기
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
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
            os.makedirs('norms')
        os.makedirs('loss')
        os.makedirs('after_norm_penalty')
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
        self.count += 1
        self.x_list.append(self.count)
        
        outputs = self.encoder(input_ids, attention_mask)
        
        outputs = outputs.view(-1, outputs.shape[-1])

        if self.sparse_prediction and labels is not None:
            # print('self.sparse_prediction and labels is not None')
            # 여기
            # Loss 계산되는 부분
            masked_lm_loss = self._forward_sparse(outputs, labels)
            original_loss = masked_lm_loss.item()
            
            self.loss_list.append(original_loss)
            
            if self.count < self.cfg.graph_interval:
                last_graph_interval_loss = sum(self.loss_list) / len(self.loss_list)
                self.last_graph_interval_loss_list.append(last_graph_interval_loss)
                print(f'Loss: {original_loss}, Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
            else:
                last_graph_interval_loss = sum(self.loss_list[-self.cfg.graph_interval :]) / len(self.loss_list[-self.cfg.graph_interval :])
                self.last_graph_interval_loss_list.append(last_graph_interval_loss)
                if self.best_loss == 0 or last_graph_interval_loss < self.best_loss:
                    self.best_loss = last_graph_interval_loss
                print(f'Loss: {original_loss}, Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}, Best_{self.cfg.graph_interval}_losses: {self.best_loss}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
            
            if (self.count % self.cfg.graph_interval == 0) or (self.count  == self.cfg.full_steps): 
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.loss_list[-self.cfg.graph_interval:])
                plt.title('Loss', fontsize=10)
                plt.xlabel('Steps', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.savefig('loss/losses.png')
                plt.clf()
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.last_graph_interval_loss_list[-self.cfg.graph_interval:])
                plt.title(f'Last {self.cfg.graph_interval} losses', fontsize=10)
                plt.xlabel('Steps', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.savefig(f'loss/last_{self.cfg.graph_interval}_losses.png')
                plt.clf()
            if self.count == self.cfg.full_steps:
                with open(f'results.txt', 'w') as file:
                    file.write(f'Loss: {original_loss}\n\n')
                    file.write(f'Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}\n\n')
                    file.write(f'Best_{self.cfg.graph_interval}_losses: {self.best_loss}\n\n')
                    file.write(f'Layers: {self.cfg.num_transformer_layers}\n\n')
                    file.write(f'Count: {self.count}\n\n')
                
            
        else:
            # print('not self.sparse_prediction and labels is not None')
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

        return {"loss": masked_lm_loss, "outputs": outputs}

    # Sparse prediction usually has an unpredictable number of entries in each batch
    # but the dataloader was modified so that 25% of the batch is ALWAYS masked.
    # This allows for static compilation. If you modify the dataloader, this function will fill your compile cache
    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):

        labels = labels.view(-1)
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        # outputs = outputs[mask_positions]  # not allowed as dynamic shape op
        # labels = labels[mask_positions]
        # torch.masked_select(labels, mask_positions)  # not allowed as a dynamic shape operator

        # indices = torch.arange(mask_positions.shape[0], device=outputs.device)[mask_positions] # not allowed
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]  # ugh

        outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        labels = labels[indices]
        # alternative:
        # outputs = torch.take_along_dim(outputs, indices.view(-1, 1), 0)
        # labels = torch.take(labels, indices)

        outputs = self.decoder(self.prediction_head(outputs))
        # print('outputs', outputs.shape)
        masked_lm_loss = self.loss_fn(outputs, labels)
        return masked_lm_loss
 
class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        # print('dddd self.cfg\n', self.cfg)
        # print('self.cfg.attention.seq_op_in_fp32', self.cfg.attention.seq_op_in_fp32)
        self.num_labels = self.cfg.num_labels
        self.cfg.classification_head.experiment_float64 = self.cfg.experiment_float64
        self.cfg.classification_head.get_input_range = self.cfg.get_input_range
        if not self.cfg.get_grad == None:
            self.cfg.classification_head.get_grad = self.cfg.get_grad

        self.encoder = ScriptableLM(config)
        self.pooler = PoolingComponent_lora(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_100_loss_list = []
        self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0
        self.before_Tanh_norm_list = []
        self.last_100_before_Tanh_norm_list = []
        os.makedirs(self.cfg.task_name)

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
        # Ordinary
        ######################################
        encoder_output = self.encoder(input_ids, attention_mask)
        ######################################
        # # Ordinary-matmul
        #######################################
        # encoder_output,  matmuls_from_enc = self.encoder(input_ids, attention_mask)
        #######################################
        # # Before normalization / GELU checking
        # #######################################
        # encoder_output,  matmuls_from_enc, before_emb_layernorm, layernorms_1, layernorms_2, gelus, before_layernorm_final = self.encoder(input_ids, attention_mask)
        # #######################################
        
        # if not self.cfg.attention.seq_op_in_fp32:
        #     encoder_output = encoder_output.type(torch.float16)
        # print('encoder_output', encoder_output.dtype)
        
        # Ordinary / Ordinary-matmul
        #######################################
        logits = self.head(self.pooler(encoder_output))
        #######################################
        # # Before normalization / GELU checking
        # #######################################
        # pooler_output , before_Tanh_norm = self.pooler(encoder_output)
        # logits = self.head(pooler_output)
        # #######################################
        
        # print(colored('logits: {}'.format(logits.shape), 'green'))
        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"
            # print(colored('self.problem_type: {}'.format(self.problem_type), 'green'))
            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        # # Ordinary-matmul
        # #######################################
        # len_matmuls = len(matmuls_from_enc)
        # #######################################
        
        self.count += 1
        self.x_list.append(self.count)
        self.loss_list.append(loss.item())
        # # Before normalization / GELU checking
        # self.before_Tanh_norm_list.append(before_Tanh_norm)
        
        if self.count < 100:
            last_100_loss = sum(self.loss_list) / len(self.loss_list)
            self.last_100_loss_list.append(last_100_loss)
            last_100_before_Tanh_norm = sum(self.before_Tanh_norm_list) / len(self.before_Tanh_norm_list)
            self.last_100_before_Tanh_norm_list.append(last_100_before_Tanh_norm)
            # print('\nLoss: {}, Last_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.cfg.num_transformer_layers, self.count))
        else:
            last_100_loss = sum(self.loss_list[-100 :]) / len(self.loss_list[-100 :])
            self.last_100_loss_list.append(last_100_loss)
            if self.best_loss == 0 or last_100_loss < self.best_loss:
                self.best_loss = last_100_loss
            last_100_before_Tanh_norm = sum(self.before_Tanh_norm_list[-100 :]) / len(self.before_Tanh_norm_list[-100 :])
            self.last_100_before_Tanh_norm_list.append(last_100_before_Tanh_norm)
            # print('\nLoss: {}, Last_100_losses: {}, Best_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.best_loss, self.cfg.num_transformer_layers, self.count))
            
        # # Ordinary-matmul
        # #######################################
        # for i in range(len_matmuls):
        #     norm_i = torch.norm(matmuls_from_enc[i], p=float('inf'))
        #     # print('Matmul_{}: {}'.format(i, norm_i.item()))
        #     self.matmul_results[i].append(norm_i.item())
            
        #     # # Loss 추가
        #     # if norm_i > 450:
        #     #     loss += 0.1 * norm_i
        # #######################################
            
        if self.count % 100 == 0:
            plt.plot(self.x_list, self.loss_list)
            plt.title('Loss')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'losses.png'))
            plt.clf()
            plt.plot(self.x_list, self.last_100_loss_list)
            plt.title('Last 100 losses')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'last_100_losses.png'))
            plt.clf()
            # # Ordinary-matmul
            # #######################################
            # for i in range(len_matmuls):
            #     plt.subplot(5, 4, i + 1)
            #     plt.plot(self.x_list, self.matmul_results[i])
            #     plt.title('Matmul_{}'.format(i))
            #     plt.xlabel('Steps')
            # plt.savefig(os.path.join(self.cfg.task_name, 'matmuls.png'))
            # plt.clf()
            # #######################################
            # # Before normalization / GELU checking
            # #######################################
            # plt.plot(self.x_list, self.before_Tanh_norm_list)
            # plt.title('Before Tanh norm')
            # plt.xlabel('Steps')
            # plt.savefig(os.path.join(self.cfg.task_name, 'before_Tanh_norm.png'))
            # plt.clf()
            # plt.plot(self.x_list, self.last_100_before_Tanh_norm_list)
            # plt.title('Last 100 before Tanh norm')
            # plt.xlabel('Steps')
            # plt.savefig(os.path.join(self.cfg.task_name, 'last_100_before_Tanh_norm.png'))
            # plt.clf()
            # #######################################
                     
        
        return dict(logits=logits, loss=loss)


class ScriptableLMForSCRIPTTraining(PreTrainedModel):
    """Pretraining machinery using SCRIPT from Nijkamp et al., 2021. Always running sparse prediction."""

    config_class = crammedBertConfig
    ALPHA = 1.0  # SCRIPT constant

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM(config)
        self.prediction_head = PredictionHeadComponent(self.cfg)

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction
        assert self.sparse_prediction

        self._init_weights()

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

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        loss = torch.tensor(0.0, dtype=torch.float, device=input_ids.device)

        outputs = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        if labels is not None:
            # ## Generation pass ##
            labels = labels.view(-1)
            mask_positions = labels.view(-1) != self.loss_fn.ignore_index
            num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
            indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]

            # sparse outputs for prediction
            outputs = outputs[indices]
            labels = labels[indices]

            logits = self.decoder(self.prediction_head(outputs))  # sparse logits
            loss += self.loss_fn(logits, labels)

            # ## Discrimination pass ##
            resampled_token_ids = self._gumbel_sample(logits.detach())
            discriminator_input_ids = input_ids.clone().view(-1)
            discriminator_input_ids[indices] = resampled_token_ids

            critic_labels = (input_ids.view(-1) != discriminator_input_ids).to(outputs.dtype)

            outputs = self.encoder(discriminator_input_ids.view_as(input_ids), attention_mask).view(-1, outputs.shape[-1])
            disc_logits = self.decoder(self.prediction_head(outputs))  # full logits
            binary_logits = self._get_binary_logits(disc_logits)

            # ELECTRA-type discriminator:
            loss += self.ALPHA * torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, critic_labels)

        else:
            logits = self.decoder(self.prediction_head(outputs))
            loss += outputs.new_zeros((1,))

        return {"loss": loss, "logits": logits}

    def _get_binary_logits(self, logits):
        # Convert to binary decision as described in SCRIPT
        # exp_logitsum = torch.exp(disc_logits).sum(dim=-1)  # autocast ok?
        # binary_logits = torch.stack([1 / (exp_logitsum + 1), exp_logitsum / (exp_logitsum + 1)], dim=-1)  # stack minus and plus
        # instead, we can also compute logit[binary_logits], which is

        # let y = sum(exp(logits)) / ( sum(exp(logits))+1 ), 1-y = 1 / ( sum(exp(logits))+1 )
        # log(y / (1-y)) = log( sum(exp(logits)) / ( sum(exp(logits))+1 ) * ( sum(exp(logits))+1 ) / 1)
        #                = log(sum(exp(logits))
        # Then, we can use BCEWithLogitsLoss, to safely compute logit probs via sigmoids
        return torch.logsumexp(logits, dim=-1)

    def _gumbel_sample(self, logits, temperature=1.0):
        """via https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py"""
        return ((logits / temperature) + self._gumbel_noise(logits)).argmax(dim=-1)

    def _gumbel_noise(self, inputs, eps=1e-9):
        """via https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py"""
        noise = torch.zeros_like(inputs).uniform_(0, 1)
        return -torch.log(-torch.log(noise + eps) + eps)


class ScriptableLMForTokenClassification(PreTrainedModel):
    """Classification head without pooling."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

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

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        logits = self.head(self.encoder(input_ids, attention_mask))

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
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                raise ValueError("Wrong problem type!")
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)


# ###### HF registry here ############### #

AutoConfig.register("crammedBERT", crammedBertConfig)
AutoModel.register(crammedBertConfig, ScriptableLM)
AutoModelForMaskedLM.register(crammedBertConfig, ScriptableLMForPreTraining)
AutoModelForSequenceClassification.register(crammedBertConfig, ScriptableLMForSequenceClassification)
AutoModelForTokenClassification.register(crammedBertConfig, ScriptableLMForTokenClassification)
