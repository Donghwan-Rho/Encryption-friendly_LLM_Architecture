"""Attention modules. The final model uses "self-attention", but other options were tried and are still documented here."""
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention

from .embeddings import Rotary, RotarySanityCheck, RotaryEleutherAI, RotaryLLAMA
from typing import Optional
from einops.layers.torch import Rearrange
from einops import rearrange
import logging
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast
import time
from datetime import datetime

log = logging.getLogger(__name__)

def block_matmul(A, B, block_num):
    b, a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    
    result = torch.zeros((b, a_rows, b_cols), dtype=A.dtype, device=A.device)
    
    a_block_rows = a_rows // block_num
    a_block_cols = a_cols // block_num
    b_block_rows = b_rows // block_num
    b_block_cols = b_cols // block_num
    
    for i in range(block_num):
        for j in range(block_num):
            A_block = A[:, i*a_block_rows:(i+1)*a_block_rows, j*a_block_cols:(j+1)*a_block_cols]
            B_block = B[i*b_block_rows:(i+1)*b_block_rows, j*b_block_cols:(j+1)*b_block_cols]
            result[:, i*a_block_rows:(i+1)*a_block_rows, j*b_block_cols:(j+1)*b_block_cols] = torch.matmul(A_block, B_block)
    return result
            
class Block_Matmul_Module(nn.Module):
    def __init__(self, in_features, out_features, block_num):
        super(Block_Matmul_Module, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_num = block_num
        
        # W: (in_features, out_features)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

    def forward(self, x):
        return block_matmul(x, self.weight, self.block_num)            
            
class LegacySeqFirstSelfAttention_modified_LoRA(torch.nn.Module):
    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor
    
    def __init__(self, hidden_size: int, cfg_attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        qkv_interm_dim = cfg_attention.qkv_interm_dim
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())

        # Strided linear layer.
        self.query_a = torch.nn.Linear(self.hidden_size, qkv_interm_dim, bias=cfg_attention.qkv_bias)
        self.query_b = torch.nn.Linear(qkv_interm_dim, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.key_a = torch.nn.Linear(self.hidden_size, qkv_interm_dim, bias=cfg_attention.qkv_bias)
        self.key_b = torch.nn.Linear(qkv_interm_dim, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.value_a = torch.nn.Linear(self.hidden_size, qkv_interm_dim, bias=cfg_attention.qkv_bias)
        self.value_b = torch.nn.Linear(qkv_interm_dim, self.hidden_size, bias=cfg_attention.qkv_bias)
        
        self.output_dim = hidden_size # 768
        
        self.rotary_emb = None

        if cfg_attention.is_train:
            if cfg_attention.train_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.train_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            # elif cfg_attention.train_sequence_op == "exp_power_app":
            #     self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
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

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # hidden_states: 128, b, 768
        query_layer = self.query_a(hidden_states) # 128, b, 128
        query_layer = self.query_b(query_layer) # 128, b, 768
        key_layer = self.key_a(hidden_states)
        key_layer = self.key_b(key_layer)
        value_layer = self.value_a(hidden_states)
        value_layer = self.value_b(value_layer)
        
        query_layer = query_layer.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head
        )
        key_layer = key_layer.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head
        )
        value_layer = value_layer.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head
        )
        if self.rotary_emb is not None:
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer)
            
        context_layer, matmul_result = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training) # C'
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous() # C''
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size) # C'''
        return context_layer, matmul_result
    
    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False):
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        query_layer_matmul = query_layer.transpose(0, 1)
        key_layer_matmul = key_layer.transpose(0, 1)

        # Softmax
        if self.operation in ['torch-softmax']:
            # print(f'softmax')
            matmul_result = torch.bmm(query_layer_matmul, key_layer_matmul.transpose(1, 2)) * self.norm_factor
        # GK
        elif self.operation in ['exp', 'exp_power_app']:   
            # print('GK')
            matmul_result = subtraction_gaussian_kernel_torch(query_layer_matmul, key_layer_matmul)
            matmul_result *= -self.norm_factor * 0.5
            
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])
        attention_probs = self.sequence_op(attention_scores, attention_mask)
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout, training=training)
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        value_layer = value_layer.transpose(0, 1)
        context_layer = torch.bmm(attention_probs, value_layer) # C
        context_layer = context_layer.view(*output_size) # C'
        return context_layer, matmul_result

def subtraction_gaussian_kernel_torch(q, k):
    k = k.transpose(-1, -2) 

    # [B, H, H1*W1, C] @ [C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()

    # [H1*W1, C] @ [B, H, C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)

def get_attention_mechanism(
    idx,
    hidden_size,
    # cfg_attention,
    cfg_arch,
):
    cfg_attention = cfg_arch.attention
    cfg_attention.type = cfg_attention['type']
    
    if  cfg_attention.type == "self-attention-modified":
        mechanism = SeqFirstSelfAttention_modified(hidden_size, cfg_arch)
    elif cfg_attention.type == "self-attention-modified_LoRA_style":
        mechanism = LegacySeqFirstSelfAttention_modified_LoRA(hidden_size, cfg_attention)
    else:
        raise ValueError(f"Invalid attention type {cfg_attention.type} given.")
    return mechanism

class LegacySeqFirstSelfAttention_modified(torch.nn.Module):
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
    def __init__(self, hidden_size: int, cfg_arch):
        super().__init__()
        # print(f'LegacySeqFirstSelfAttention_modified cfg_attention: {cfg_attention}')
        cfg_attention = cfg_arch.attention
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())
        
        self.qkv_each = cfg_arch.qkv_each
        # Strided linear layer.
        
        # Q != K setting
        ################################
        # Linear(768, 2304)
        if cfg_arch.qkv_each:
            if cfg_arch.block_matmul:
                ...
            else:
                self.query = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
                self.key = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
                self.value = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        else:
            if cfg_arch.block_matmul:
                self.query_key_value = Block_Matmul_Module(self.hidden_size, 3 * self.hidden_size, block_num=cfg_arch.block_num)
            else:
                self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=cfg_attention.qkv_bias)
        ################################
        
        self.output_dim = hidden_size # 768
        if cfg_attention.rotary_embedding == "sanity":
            self.rotary_emb = RotarySanityCheck(self.hidden_per_head, seq_dim=0)
        elif cfg_attention.rotary_embedding == "v2":
            self.rotary_emb = RotaryEleutherAI(self.hidden_per_head)
        elif cfg_attention.rotary_embedding == "llama":
            self.rotary_emb = RotaryLLAMA(self.hidden_per_head)
        elif cfg_attention.rotary_embedding:
            self.rotary_emb = Rotary(self.hidden_per_head, seq_dim=0)
        else:
            self.rotary_emb = None

        if cfg_attention.is_train:
            if cfg_attention.train_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.train_sequence_op == "torch-relu":
                self.sequence_op = TorchReLU(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "torch-relu-norm":
                self.sequence_op = TorchReLU_Norm(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "torch-norm":
                self.sequence_op = TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.train_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.train_sequence_op == "exp_poly_app":
                self.sequence_op = exp_poly_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "exp_taylor_app":
                self.sequence_op = exp_taylor_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "poly":
                self.sequence_op = Polynorm(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "none":
                self.sequence_op = ScaledIdentity(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "cumsum":
                self.sequence_op = Cumsum(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "cumsumexp":
                self.sequence_op = CumsumExp(cfg_attention.seq_op_in_fp32)
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")
        else:
            if cfg_attention.eval_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
                self.operation = "torch-softmax"
            elif cfg_attention.eval_sequence_op == "torch-relu":
                self.sequence_op = TorchReLU(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "torch-relu-norm":
                self.sequence_op = TorchReLU_Norm(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "torch-norm":
                self.sequence_op = TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.eval_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.exp_power_deg, cfg_attention.seq_op_in_fp32)
                self.operation = "exp"
            elif cfg_attention.eval_sequence_op == "exp_poly_app":
                self.sequence_op = exp_poly_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "exp_taylor_app":
                self.sequence_op = exp_taylor_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "poly":
                self.sequence_op = Polynorm(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "none":
                self.sequence_op = ScaledIdentity(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "cumsum":
                self.sequence_op = Cumsum(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "cumsumexp":
                self.sequence_op = CumsumExp(cfg_attention.seq_op_in_fp32)
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        self.attention_dropout: float = cfg_attention.dropout_prob

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # =====================
        # hidden_states: [sq, b, h]
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        if self.qkv_each:
            query_layer = self.query(hidden_states)
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)
            
            query_layer = query_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
            key_layer = key_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
            value_layer = value_layer.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)

        else:
            mixed_x_layer = self.query_key_value(hidden_states) # 128 128 2304
        
            # Q != K setting
            ################################
            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            mixed_x_layer = mixed_x_layer.view(
                hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, 3 * self.hidden_per_head
            )
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.hidden_per_head] * 3, dim=3)
            ################################
        
        if self.rotary_emb is not None:
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer)

        # ==================================
        # Attention computation
        # ==================================
        context_layer, matmul_result = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training) # C'
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous() # C''

        # [sq, b, np, hn] --> [sq, b, hp]
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size) # C'''
        
        return context_layer, matmul_result

 
class SeqFirstSelfAttention_modified(LegacySeqFirstSelfAttention_modified):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    This is a modified version of the neo-x implementation that I can manage to compile without graph breaks
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        # [b, np, sq, sk]
        
        # 32, 12, 128, 128
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        
        query_layer_matmul = query_layer.transpose(0, 1)
        key_layer_matmul = key_layer.transpose(0, 1)
        
        # Softmax
        if self.operation in ['torch-softmax']:
            matmul_result = torch.bmm(query_layer_matmul, key_layer_matmul.transpose(1, 2)) * self.norm_factor
        # GK
        elif self.operation in ['exp', 'exp_power_app']:   
            matmul_result = subtraction_gaussian_kernel_torch(query_layer_matmul, key_layer_matmul)
            matmul_result *= -self.norm_factor * 0.5

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])

        # ===========================
        # Attention probs and dropout
        # ===========================
        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.sequence_op(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [And in great ML tradition I keep this comment in place as it was in megatron and huggingface-bert before :>]
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout, training=training)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        value_layer = value_layer.transpose(0, 1)

        context_layer = torch.bmm(attention_probs, value_layer) # b*nh, sq, np

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size) # C'

        return context_layer, matmul_result
    
class TorchSoftmax(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)

        if attention_mask is not None:
            inputs = inputs + attention_mask
            
        probs = torch.softmax(inputs, dim=-1).to(dtype=input_dtype)
        return probs

class TorchReLU(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)  
                      
        if attention_mask is not None:
            inputs = inputs + attention_mask
            
        outputs = torch.nn.functional.relu(inputs).to(dtype=input_dtype)
        return outputs

class TorchReLU_Norm(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        if attention_mask is not None:
            inputs = inputs + attention_mask    #In Softmax you add  -infty and apply softmax?

        outputs = torch.nn.functional.relu(inputs).to(dtype=input_dtype)
        outputs = outputs / (torch.sum(outputs, dim=-1, keepdim=True) + 1e-7)
        
        return outputs

class TorchNormalize(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        
        if attention_mask is not None:
            inputs[attention_mask != 0] = 0

        norms = torch.nn.functional.layer_norm(inputs, inputs.shape[1:], eps=1e-05)
        norms = (norms * self.seq_gamma + self.seq_beta).to(dtype=input_dtype)
        return norms
    
class Polynorm(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, poly_type = 'sigmoid', norm_type = 2, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))
        self.poly_type = poly_type
        self.norm_type = norm_type

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        activ =  lambda x: x**2

        if self.poly_type == 'quadratic':
            activ = lambda x : x**2
        elif self.poly_type == 'cubic':
            activ = lambda x : x**3
        elif self.poly_type == 'tanh':
            activ = lambda x : x - x**3/3 + 2*x**5/15
        elif self.poly_type == 'sigmoid':
            activ = lambda x : 1/2 + x/4 - x ** 3 / 48 + x ** 5 /480 
        # elif self.poly_type == 'gelu':
        #     activ = lambda x : 

        inputs = activ(inputs)
        
        if attention_mask is not None:
            inputs = inputs + attention_mask
        
        if self.norm_type == 0:
            norms = torch.nn.functional.layer_norm(inputs, inputs.shape[1:], eps=1e-05)
        elif self.norm_type == 1:
            norms = inputs / (torch.sum(inputs, dim=-1, keepdim=True) + 1e-7)
        elif self.norm_type == 2:
            norms = inputs

        norms = (norms * self.seq_gamma + self.seq_beta).to(dtype=input_dtype)
        return norms
    
class Exp(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)

        if attention_mask is not None:
            inputs = inputs + attention_mask
        activ =  lambda x: torch.exp(x)
        outputs = activ(inputs)

        return outputs
      
class exp_power_app(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, exp_power_deg=9, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.exp_power_deg = exp_power_deg

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        
        # input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        
        if attention_mask is not None:
            inputs = inputs + attention_mask
            
        activ =  lambda x: (1 + x / (2 ** self.exp_power_deg)) ** (2 ** self.exp_power_deg)       
        outputs = activ(inputs)
        
        return outputs
 
class exp_taylor_app(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        activ =  lambda x: sum([x ** i / math.factorial(i) for i in range(15)])
            # 1 + x + x**2 / 2 + x**3 / 6 + x**4 / 24 + x**5 / 120 + x**6 / 720 + x**7 / 5040
        
        outputs = 1
        for i in range(1, 14):
            outputs += (inputs) ** i / math.factorial(i)

        if attention_mask is not None:
            inputs = inputs + attention_mask
            
        return outputs
   
class exp_poly_app(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        
        # x <- x/K            
        inputs = inputs / 300
        
        activ = lambda x: 0.0000195464058*(x**7) + 0.000482184328*(x**6)\
                            + 0.00533666219*(x**5)\
                            + 0.0355159261*(x**4) + 0.159281596*(x**3)\
                            + 0.495328581*(x**2) + 0.99874163*x + 0.999917605
        outputs = activ(inputs) ** 300
        
        if attention_mask is not None:
            inputs = inputs + attention_mask
            
        return outputs
    
class ScaledIdentity(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs * torch.as_tensor(inputs.shape[2]).rsqrt()).to(dtype=input_dtype)

class Cumsum(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input cumulative sum."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs.cumsum(dim=-1) * pow(inputs.shape[2], -0.5)).to(dtype=input_dtype)

class CumsumExp(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = True  # Required as of pytorch 1.13

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input cumulative sum."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs.logcumsumexp(dim=-1) * pow(inputs.shape[2], -0.5)).to(dtype=input_dtype)
