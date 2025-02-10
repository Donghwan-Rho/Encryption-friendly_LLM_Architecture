import torch
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
from .attention_modified import *


class TransformerLayer_Combined(torch.nn.Module):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    Self-attention layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    # def __init__(self, hidden_size: int, cfg_attention):
    def __init__(self, idx, cfg_arch):
        super().__init__()
        # print(f'LegacySeqFirstSelfAttention_modified cfg_attention: {cfg_attention}')
        self.cfg_arch = cfg_arch
        self.idx = idx
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)

        cfg_attention = cfg_arch.attention
        self.hidden_size = cfg_arch.hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        self.num_combined_heads = cfg_arch.num_combined_heads
        self.num_FFN = self.num_attention_heads // self.num_combined_heads
        # 64
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(self.hidden_per_head, eps=cfg_arch.norm_eps)
        # print('self.hidden_per_head', self.hidden_per_head)
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())
        
        self.qkv_each = cfg_arch.qkv_each
        # Linear layer.
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.attention_dense = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        
        self.output_dim = cfg_arch.hidden_size # 768
        
        self.rotary_emb = None

        if cfg_attention.is_train:
            if cfg_attention.train_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.train_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.train_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")
        else:
            if cfg_attention.eval_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.eval_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.eval_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.exp_power_deg, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        self.attention_dropout: float = cfg_attention.dropout_prob
        
        self.FFN_multiple = self.cfg_arch.combined_FFN_multiple
        # FFN
        self.dense_in = torch.nn.ModuleList([torch.nn.Linear(self.num_combined_heads*self.hidden_per_head, self.FFN_multiple*self.num_combined_heads*self.hidden_per_head, bias=cfg_arch.use_bias) for _ in range(self.num_FFN)])
        # weights_list = []
        # for layer in self.dense_in:
        #     weights_list.append(layer.weight.unsqueeze(0))
        # # nh, np, 4*np
        # self.dense_in_weights = torch.cat(weights_list, dim=0).transpose(1, 2).to('cuda')
        self.nonlin = torch.nn.ReLU()
        self.dense_out = torch.nn.ModuleList([torch.nn.Linear((self.FFN_multiple//2)*self.num_combined_heads*self.hidden_per_head, self.num_combined_heads*self.hidden_per_head, bias=cfg_arch.use_bias) for _ in range(self.num_FFN)])
        # weights_list = []
        # for layer in self.dense_out:
        #     weights_list.append(layer.weight.unsqueeze(0))
        # # nh, 2*np, np
        # self.dense_out_weights = torch.cat(weights_list, dim=0).transpose(1, 2).to('cuda')

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        norm1_inputs = hidden_states
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.device.type}')
        
        # sq, b, d
        hidden_states = self.norm1(hidden_states)
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.device.type}')
                
        query_layer = self.query(hidden_states) # sq, b, d
        key_layer = self.key(hidden_states) # sq, b, d
        value_layer = self.value(hidden_states) # sq, b, d
        # print(f'query_layer: {query_layer.shape}')
        # print(f'query_layer: {query_layer.device.type}')
        
        # sq, b, nh, np
        query_layer = query_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        key_layer = key_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        value_layer = value_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        # print(f'192 query_layer: {query_layer.shape}')

        if self.rotary_emb is not None:
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer)

        # Multi-head Attention
        #######################
        # b, nh, sq, sq
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        # sq, b*nh, np
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # print(f'203 query_layer: {query_layer.shape}')
        # sq, b*nh, np
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # print(f'203 key_layer: {key_layer.shape}')
        # b*nh, sq, np
        query_layer_matmul = query_layer.transpose(0, 1)
        # print(f'query_layer_matmul: {query_layer_matmul.shape}')
        # b*nh, sq, np
        key_layer_matmul = key_layer.transpose(0, 1)
        if self.operation in ['torch-softmax']:
            # b*nh, sq, sq
            matmul_result = torch.bmm(query_layer_matmul, key_layer_matmul.transpose(1, 2)) * self.norm_factor
        # GK
        elif self.operation in ['exp', 'exp_power_app']:   
            # b*nh, sq, sq
            matmul_result = subtraction_gaussian_kernel_torch(query_layer_matmul, key_layer_matmul)
            matmul_result *= -self.norm_factor * 0.5
        # print(f'matmul_result: {matmul_result.shape}')
        # print(f'matmul_result: {matmul_result.device.type}')
        # b, nh, sq, sq    
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])
        # print(f'attention_scores: {attention_scores.shape}')
        # b, nh, sq, sq
        attention_probs = self.sequence_op(attention_scores, attention_mask)
        # print(f'attention after exp: {attention_probs.shape}')
        # b, nh, sq, sq
        attention_probs = self.dropout(attention_probs)
        # b, nh, sq, np
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # sq, b*nh, np
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        # print(f'232 value_layer: {value_layer.shape}')
        # b*nh, sq, sq
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # print(f'attention_probs: {attention_probs.shape}')
        # b*nh, sq, np
        value_layer = value_layer.transpose(0, 1)
        # print(f'237 value_layer: {value_layer.shape}')
        # b*nh, sq, np
        context_layer = torch.bmm(attention_probs, value_layer) # b*nh, sq, np
        # print(f'238 context_layer: {context_layer.shape}')
        # b, nh, sq, np, attention output, 여기서 FFN 들어가야 함
        context_layer = context_layer.view(*output_size)
        # print(f'241 context_layer: {context_layer.shape}')
        # b, nh, sq, np
        context_layer = self.dropout(context_layer)
        # sq, b, nh, np
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # print(f'246 context_layer: {context_layer.shape}')
        # sq, b, d
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size)
        # print(f'249 context_layer: {context_layer.shape}')
        #######################
        # sq, b, d
        context_layer = hidden_states + context_layer
        # print(f'253 context_layer: {context_layer.shape}')
        # sq, b, nh, np
        context_layer = context_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        # print(f'256 context_layer: {context_layer.shape}')
        
        norm2_inputs = context_layer
        
        context_layer = self.norm2(context_layer)
        # b, nh, sq, np
        context_layer = context_layer.permute(1, 2, 0, 3).contiguous()
        # print(f'after permute context_layer: {context_layer.shape}')
        
        combined_tensors = []
        for i in range(self.num_FFN):
            # print(f'i: {i}')
            # b, num_combined_heads, sq, np
            combined_tensor = context_layer[:, i*self.num_combined_heads:(i+1)*self.num_combined_heads, :, :]
            # print(f'combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np
            combined_tensor = combined_tensor.permute(2, 0, 1, 3).reshape(combined_tensor.shape[2], combined_tensor.shape[0], -1)
            # print(f'after permute combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np*FFN_multiple
            combined_tensor = self.dense_in[i](combined_tensor)
            # print(f'after dense_in combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np*FFN_multiple/2
            x, y = combined_tensor.chunk(2, dim=-1)
            nonlin_input = y
            # sq, b, num_combined_heads*np*FFN_multiple/2
            y = self.nonlin(nonlin_input)
            # sq, b, num_combined_heads*np*FFN_multiple/2
            combined_tensor = x * y
            # print(f'after hadamard combined_tensor: {combined_tensor.shape}')
            
            # sq, b, num_combined_heads*np
            combined_tensor = self.dense_out[i](combined_tensor)   
            combined_tensors.append(combined_tensor)
            # print(f'after dense_out combined_tensor: {combined_tensor.shape}')

        # sq, b, d            
        context_layer = torch.cat(combined_tensors, dim=-1)
        # print(f'after combined FFN context_layer: {context_layer.shape}')
        # sq, b, d
        context_layer = self.attention_dense(context_layer)
        # print(f'after att_dense combined_tensor: {combined_tensor.shape}')
        # torch.cuda.empty_cache()
        if self.cfg_arch.get_input_range:
            return context_layer, matmul_result, norm1_inputs, norm2_inputs, nonlin_input
        else: # states = last_hidden_states, matmul_result = attention_before_exp
            return context_layer, matmul_result

class TransformerLayer_Combined_ver2(torch.nn.Module):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    Self-attention layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    # def __init__(self, hidden_size: int, cfg_attention):
    def __init__(self, idx, cfg_arch):
        super().__init__()
        # print(f'LegacySeqFirstSelfAttention_modified cfg_attention: {cfg_attention}')
        self.cfg_arch = cfg_arch
        self.idx = idx
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)

        cfg_attention = cfg_arch.attention
        self.hidden_size = cfg_arch.hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        self.num_combined_heads = cfg_arch.num_combined_heads
        self.num_FFN = self.num_attention_heads // self.num_combined_heads
        # 64
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        # print('self.hidden_per_head', self.hidden_per_head)
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())
        
        self.qkv_each = cfg_arch.qkv_each
        # Linear layer.
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.attention_dense = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        
        self.output_dim = cfg_arch.hidden_size # 768
        
        self.rotary_emb = None

        if cfg_attention.is_train:
            if cfg_attention.train_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.train_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.train_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")
        else:
            if cfg_attention.eval_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.eval_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.eval_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.exp_power_deg, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        self.attention_dropout: float = cfg_attention.dropout_prob
        
        self.FFN_multiple = self.cfg_arch.combined_FFN_multiple
        # FFN
        self.dense_in = torch.nn.ModuleList([torch.nn.Linear(self.num_combined_heads*self.hidden_per_head, self.FFN_multiple*self.num_combined_heads*self.hidden_per_head, bias=cfg_arch.use_bias) for _ in range(self.num_FFN)])
        # weights_list = []
        # for layer in self.dense_in:
        #     weights_list.append(layer.weight.unsqueeze(0))
        # # nh, np, 4*np
        # self.dense_in_weights = torch.cat(weights_list, dim=0).transpose(1, 2).to('cuda')
        self.nonlin = torch.nn.ReLU()
        self.dense_out = torch.nn.ModuleList([torch.nn.Linear((self.FFN_multiple//2)*self.num_combined_heads*self.hidden_per_head, self.num_combined_heads*self.hidden_per_head, bias=cfg_arch.use_bias) for _ in range(self.num_FFN)])
        # weights_list = []
        # for layer in self.dense_out:
        #     weights_list.append(layer.weight.unsqueeze(0))
        # # nh, 2*np, np
        # self.dense_out_weights = torch.cat(weights_list, dim=0).transpose(1, 2).to('cuda')

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        norm1_inputs = hidden_states
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.device.type}')
        
        # sq, b, d
        hidden_states = self.norm1(hidden_states)
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.device.type}')
                
        query_layer = self.query(hidden_states) # sq, b, d
        key_layer = self.key(hidden_states) # sq, b, d
        value_layer = self.value(hidden_states) # sq, b, d
        # print(f'query_layer: {query_layer.shape}')
        # print(f'query_layer: {query_layer.device.type}')
        
        # sq, b, nh, np
        query_layer = query_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        key_layer = key_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        value_layer = value_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        # print(f'192 query_layer: {query_layer.shape}')

        if self.rotary_emb is not None:
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer)

        # Multi-head Attention
        #######################
        # b, nh, sq, sq
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        # sq, b*nh, np
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # print(f'203 query_layer: {query_layer.shape}')
        # sq, b*nh, np
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # print(f'203 key_layer: {key_layer.shape}')
        # b*nh, sq, np
        query_layer_matmul = query_layer.transpose(0, 1)
        # print(f'query_layer_matmul: {query_layer_matmul.shape}')
        # b*nh, sq, np
        key_layer_matmul = key_layer.transpose(0, 1)
        if self.operation in ['torch-softmax']:
            # b*nh, sq, sq
            matmul_result = torch.bmm(query_layer_matmul, key_layer_matmul.transpose(1, 2)) * self.norm_factor
        # GK
        elif self.operation in ['exp', 'exp_power_app']:   
            # b*nh, sq, sq
            matmul_result = subtraction_gaussian_kernel_torch(query_layer_matmul, key_layer_matmul)
            matmul_result *= -self.norm_factor * 0.5
        # print(f'matmul_result: {matmul_result.shape}')
        # print(f'matmul_result: {matmul_result.device.type}')
        # b, nh, sq, sq    
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])
        # print(f'attention_scores: {attention_scores.shape}')
        # b, nh, sq, sq
        attention_probs = self.sequence_op(attention_scores, attention_mask)
        # print(f'attention after exp: {attention_probs.shape}')
        # b, nh, sq, sq
        attention_probs = self.dropout(attention_probs)
        # b, nh, sq, np
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # sq, b*nh, np
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        # print(f'232 value_layer: {value_layer.shape}')
        # b*nh, sq, sq
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # print(f'attention_probs: {attention_probs.shape}')
        # b*nh, sq, np
        value_layer = value_layer.transpose(0, 1)
        # print(f'237 value_layer: {value_layer.shape}')
        # b*nh, sq, np
        context_layer = torch.bmm(attention_probs, value_layer) # b*nh, sq, np
        # print(f'238 context_layer: {context_layer.shape}')
        # b, nh, sq, np, attention output, 여기서 FFN 들어가야 함
        context_layer = context_layer.view(*output_size)
        # print(f'241 context_layer: {context_layer.shape}')
        # b, nh, sq, np
        context_layer = self.dropout(context_layer)
        # sq, b, nh, np
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # print(f'246 context_layer: {context_layer.shape}')
        # sq, b, d
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size)
        # print(f'249 context_layer: {context_layer.shape}')
        #######################
        # sq, b, d
        context_layer = hidden_states + context_layer
        # print(f'253 context_layer: {context_layer.shape}')
        context_layer = self.norm2(context_layer)
        # sq, b, nh, np
        context_layer = context_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        # print(f'256 context_layer: {context_layer.shape}')
        
        norm2_inputs = context_layer
        
        # b, nh, sq, np
        context_layer = context_layer.permute(1, 2, 0, 3).contiguous()
        # print(f'after permute context_layer: {context_layer.shape}')
        
        combined_tensors = []
        for i in range(self.num_FFN):
            # print(f'i: {i}')
            # b, num_combined_heads, sq, np
            combined_tensor = context_layer[:, i*self.num_combined_heads:(i+1)*self.num_combined_heads, :, :]
            # print(f'combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np
            combined_tensor = combined_tensor.permute(2, 0, 1, 3).reshape(combined_tensor.shape[2], combined_tensor.shape[0], -1)
            # print(f'after permute combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np*FFN_multiple
            combined_tensor = self.dense_in[i](combined_tensor)
            # print(f'after dense_in combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np*FFN_multiple/2
            x, y = combined_tensor.chunk(2, dim=-1)
            nonlin_input = y
            # sq, b, num_combined_heads*np*FFN_multiple/2
            y = self.nonlin(nonlin_input)
            # sq, b, num_combined_heads*np*FFN_multiple/2
            combined_tensor = x * y
            # print(f'after hadamard combined_tensor: {combined_tensor.shape}')
            
            # sq, b, num_combined_heads*np
            combined_tensor = self.dense_out[i](combined_tensor)   
            combined_tensors.append(combined_tensor)
            # print(f'after dense_out combined_tensor: {combined_tensor.shape}')

        # sq, b, d            
        context_layer = torch.cat(combined_tensors, dim=-1)
        # print(f'after combined FFN context_layer: {context_layer.shape}')
        # sq, b, d
        context_layer = self.attention_dense(context_layer)
        # print(f'after att_dense combined_tensor: {combined_tensor.shape}')
        # torch.cuda.empty_cache()
        if self.cfg_arch.get_input_range:
            return context_layer, matmul_result, norm1_inputs, norm2_inputs, nonlin_input
        else: # states = last_hidden_states, matmul_result = attention_before_exp
            return context_layer, matmul_result
        
class TransformerLayer_Combined_ver3(torch.nn.Module):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    Self-attention layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    # def __init__(self, hidden_size: int, cfg_attention):
    def __init__(self, idx, cfg_arch):
        super().__init__()
        # print(f'LegacySeqFirstSelfAttention_modified cfg_attention: {cfg_attention}')
        self.cfg_arch = cfg_arch
        self.idx = idx
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)

        cfg_attention = cfg_arch.attention
        self.hidden_size = cfg_arch.hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        self.num_combined_heads = cfg_arch.num_combined_heads
        self.num_FFN = self.num_attention_heads // self.num_combined_heads
        # 64
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        # print('self.hidden_per_head', self.hidden_per_head)
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())
        
        self.qkv_each = cfg_arch.qkv_each
        # Linear layer.
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.attention_dense = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        
        self.output_dim = cfg_arch.hidden_size # 768
        
        self.rotary_emb = None

        if cfg_attention.is_train:
            if cfg_attention.train_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.train_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.train_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")
        else:
            if cfg_attention.eval_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.eval_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.eval_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.exp_power_deg, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        self.attention_dropout: float = cfg_attention.dropout_prob
        
        self.FFN_multiple = self.cfg_arch.combined_FFN_multiple
        # FFN
        self.dense_in = torch.nn.ModuleList([torch.nn.Linear(self.num_combined_heads*self.hidden_per_head, self.FFN_multiple*self.num_combined_heads*self.hidden_per_head, bias=cfg_arch.use_bias) for _ in range(self.num_FFN)])
        # weights_list = []
        # for layer in self.dense_in:
        #     weights_list.append(layer.weight.unsqueeze(0))
        # # nh, np, 4*np
        # self.dense_in_weights = torch.cat(weights_list, dim=0).transpose(1, 2).to('cuda')
        self.nonlin = torch.nn.ReLU()
        self.dense_out = torch.nn.ModuleList([torch.nn.Linear((self.FFN_multiple//2)*self.num_combined_heads*self.hidden_per_head, self.num_combined_heads*self.hidden_per_head, bias=cfg_arch.use_bias) for _ in range(self.num_FFN)])
        # weights_list = []
        # for layer in self.dense_out:
        #     weights_list.append(layer.weight.unsqueeze(0))
        # # nh, 2*np, np
        # self.dense_out_weights = torch.cat(weights_list, dim=0).transpose(1, 2).to('cuda')

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        norm1_inputs = hidden_states
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.device.type}')
        
        # sq, b, d
        hidden_states = self.norm1(hidden_states)
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.device.type}')
                
        query_layer = self.query(hidden_states) # sq, b, d
        key_layer = self.key(hidden_states) # sq, b, d
        value_layer = self.value(hidden_states) # sq, b, d
        # print(f'query_layer: {query_layer.shape}')
        # print(f'query_layer: {query_layer.device.type}')
        
        # sq, b, nh, np
        query_layer = query_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        key_layer = key_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        value_layer = value_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        # print(f'192 query_layer: {query_layer.shape}')

        if self.rotary_emb is not None:
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer)

        # Multi-head Attention
        #######################
        # b, nh, sq, sq
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        # sq, b*nh, np
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # print(f'203 query_layer: {query_layer.shape}')
        # sq, b*nh, np
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # print(f'203 key_layer: {key_layer.shape}')
        # b*nh, sq, np
        query_layer_matmul = query_layer.transpose(0, 1)
        # print(f'query_layer_matmul: {query_layer_matmul.shape}')
        # b*nh, sq, np
        key_layer_matmul = key_layer.transpose(0, 1)
        if self.operation in ['torch-softmax']:
            # b*nh, sq, sq
            matmul_result = torch.bmm(query_layer_matmul, key_layer_matmul.transpose(1, 2)) * self.norm_factor
        # GK
        elif self.operation in ['exp', 'exp_power_app']:   
            # b*nh, sq, sq
            matmul_result = subtraction_gaussian_kernel_torch(query_layer_matmul, key_layer_matmul)
            matmul_result *= -self.norm_factor * 0.5
        # print(f'matmul_result: {matmul_result.shape}')
        # print(f'matmul_result: {matmul_result.device.type}')
        # b, nh, sq, sq    
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])
        # print(f'attention_scores: {attention_scores.shape}')
        # b, nh, sq, sq
        attention_probs = self.sequence_op(attention_scores, attention_mask)
        # print(f'attention after exp: {attention_probs.shape}')
        # b, nh, sq, sq
        attention_probs = self.dropout(attention_probs)
        # b, nh, sq, np
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # sq, b*nh, np
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        # print(f'232 value_layer: {value_layer.shape}')
        # b*nh, sq, sq
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # print(f'attention_probs: {attention_probs.shape}')
        # b*nh, sq, np
        value_layer = value_layer.transpose(0, 1)
        # print(f'237 value_layer: {value_layer.shape}')
        # b*nh, sq, np
        context_layer = torch.bmm(attention_probs, value_layer) # b*nh, sq, np
        # print(f'238 context_layer: {context_layer.shape}')
        # b, nh, sq, np, attention output, 여기서 FFN 들어가야 함
        context_layer = context_layer.view(*output_size)
        # print(f'241 context_layer: {context_layer.shape}')
        # b, nh, sq, np
        context_layer = self.dropout(context_layer)
        # sq, b, nh, np
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # print(f'246 context_layer: {context_layer.shape}')
        # sq, b, d
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size)
        # print(f'249 context_layer: {context_layer.shape}')
        #######################
        # sq, b, d
        context_layer = hidden_states + context_layer
        att_output = context_layer
        # print(f'253 context_layer: {context_layer.shape}')
        context_layer = self.norm2(context_layer)
        # sq, b, nh, np
        context_layer = context_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        # print(f'256 context_layer: {context_layer.shape}')
        
        norm2_inputs = context_layer
        
        # b, nh, sq, np
        context_layer = context_layer.permute(1, 2, 0, 3).contiguous()
        # print(f'after permute context_layer: {context_layer.shape}')
        
        combined_tensors = []
        for i in range(self.num_FFN):
            # print(f'i: {i}')
            # b, num_combined_heads, sq, np
            combined_tensor = context_layer[:, i*self.num_combined_heads:(i+1)*self.num_combined_heads, :, :]
            # print(f'combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np
            combined_tensor = combined_tensor.permute(2, 0, 1, 3).reshape(combined_tensor.shape[2], combined_tensor.shape[0], -1)
            # print(f'after permute combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np*FFN_multiple
            combined_tensor = self.dense_in[i](combined_tensor)
            # print(f'after dense_in combined_tensor: {combined_tensor.shape}')
            # sq, b, num_combined_heads*np*FFN_multiple/2
            x, y = combined_tensor.chunk(2, dim=-1)
            nonlin_input = y
            # sq, b, num_combined_heads*np*FFN_multiple/2
            y = self.nonlin(nonlin_input)
            # sq, b, num_combined_heads*np*FFN_multiple/2
            combined_tensor = x * y
            # print(f'after hadamard combined_tensor: {combined_tensor.shape}')
            
            # sq, b, num_combined_heads*np
            combined_tensor = self.dense_out[i](combined_tensor)   
            combined_tensors.append(combined_tensor)
            # print(f'after dense_out combined_tensor: {combined_tensor.shape}')

        # sq, b, d, FFNs output            
        context_layer = torch.cat(combined_tensors, dim=-1)
        context_layer = self.dropout(context_layer)
        context_layer = context_layer + att_output
        # sq, b, d
        context_layer = self.attention_dense(context_layer)
        # print(f'after att_dense combined_tensor: {combined_tensor.shape}')
        # torch.cuda.empty_cache()
        if self.cfg_arch.get_input_range:
            return context_layer, matmul_result, norm1_inputs, norm2_inputs, nonlin_input
        else: # states = last_hidden_states, matmul_result = attention_before_exp
            return context_layer, matmul_result
