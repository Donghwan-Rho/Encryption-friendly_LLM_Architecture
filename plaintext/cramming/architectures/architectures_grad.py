import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification

from typing import Optional
from functools import partial
from omegaconf import OmegaConf
from termcolor import colored
import os
import time

from .embeddings import SinusoidalPositional, LearnablePositional, ScaledSinosoidal
from .components import (
    get_extended_attention_mask,
    _init_module,
)
from .attention_modified import get_attention_mechanism
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

INPLACE = False

class Custom_CrossEntropyLoss_grad(nn.Module):
    def __init__(self):
        super(Custom_CrossEntropyLoss_grad, self).__init__()

    def forward(self, logits, labels):
             
        log_probs = F.log_softmax(logits, dim=1)

        loss = F.nll_loss(log_probs, labels)

        return loss

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

# Updating the ReLU_torch function and other variables to use float64
def ReLU_torch_64(x):
    f1_coeffs = torch.tensor([1.34595769293910e-33, 2.45589415425004e1, 4.85095667238242e-32, -6.69660449716894e2,
                              -2.44541235853840e-30, 6.67299848301339e3, 1.86874811944640e-29, -3.06036656163898e4,
                              -5.76227817577242e-29, 7.31884032987787e4, 8.53680673009259e-29, -9.44433217050084e4,
                              -6.02701474694667e-29, 6.23254094212546e4, 1.62342843661940e-29, -1.64946744117805e4], dtype=torch.float64)
    
    f2_coeffs = torch.tensor([1.53261588585630e-47, 9.35625636035439, -3.68972123048249e-46, -5.91638963933626e1,
                              1.74254399703303e-45, 1.48860930626448e2, -3.20672110002213e-45, -1.75812874878582e2,
                              2.79115738948645e-45, 1.09111299685955e2, -1.22590309306100e-45, -3.66768839978755e1,
                              2.62189142557962e-46, 6.31846290311294, -2.16662326421275e-47, -4.37113415082177e-01], dtype=torch.float64)
    
    g_coeffs = torch.tensor([6.43551938319983e-48, 5.07813569758861, 8.12601038855762e-46, -3.07329918137186e1,
                             -1.60198474678427e-44, 1.44109746812809e2, 1.07463154460511e-43, -4.59661688826142e2,
                             -3.63448723044512e-43, 1.02152064470459e3, 7.25207125369784e-43, -1.62056256708877e3,
                             -9.27306397853655e-43, 1.86467646416570e3, 7.95843097354065e-43, -1.56749300877143e3,
                             -4.69190103147527e-43, 9.60970309093422e2, 1.90863349654016e-43, -4.24326161871646e2,
                             -5.27439678020696e-44, 1.31278509256003e2, 9.47044937974786e-45, -2.69812576626115e1,
                             -9.98181561763750e-46, 3.30651387315565, 4.69390466192199e-47, -1.82742944627533e-1], dtype=torch.float64)
    
    return 0.5*( x + x*(polyeval_torch(g_coeffs, polyeval_torch(f2_coeffs, polyeval_torch(f1_coeffs, x)))))

class Approx_ReLU_grad(nn.Module):
    def __init__(self):
        super(Approx_ReLU_grad, self).__init__()

    def forward(self, x):
        # Apply ReLU function: max(0, x)
        x = x.to(dtype=torch.float64)
        
        return (ReLU_torch_64(x / 100) * 100).half()
        # return (ReLU_torch_64(x / 100) * 100)

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
    
    # print(f'x: {x.dtype}')
    res = evalcheb(coeffs, x, 0.0, 1.0)
    res = 0.5*(res*(3-x*res*res))
    res = 0.5*(res*(3-x*res*res))
    res = 0.5*(res*(3-x*res*res))
    return res

class Approx_LayerNorm_grad(nn.Module):
    def __init__(self, normalized_shape, div_max, eps=1e-5):
        super(Approx_LayerNorm_grad, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.div_max = div_max
        self.sqrt = math.sqrt(div_max)        
        
    def forward(self, x):
        self.x = x
        self.mean = self.x.mean(dim=-1, keepdim=True)
        self.var = ((x - self.mean) ** 2).mean(dim=-1, keepdim=True)
        self.var_eps = self.var + self.eps
        
        self.inverse_sqrt = Inverse_sqrt_approx((self.var_eps) / self.div_max) / self.sqrt
        
        self.x_normalized = (x - self.mean) * self.inverse_sqrt
        self.output = self.weight * self.x_normalized + self.bias
        return self.output

def _get_nonlin_fn_grad(nonlin_name, use_gating=True):
    if nonlin_name == 'GELU_poly_11glu':
        if 'glu' in nonlin_name.lower():
            nonlin_name = nonlin_name.split("glu")[0] # GELU
            wrap_in_glu = use_gating # True
            nonlin_fn = nonlin_name # 'GELU_poly_11glu'
        else:
            wrap_in_glu = False
        if wrap_in_glu:
            return partial(GLU_grad, nonlin_fn)
    else:
        if "glu" in nonlin_name.lower():
            nonlin_name = nonlin_name.split("glu")[0] # GELU or ReLU or SiLU
            wrap_in_glu = use_gating # True
        else:
            wrap_in_glu = False
        if nonlin_name == "Approx_ReLU":
            nonlin_fn = Approx_ReLU_grad
        else:
            nonlin_fn = getattr(torch.nn, nonlin_name) 
        try:
            if nonlin_name == "Approx_ReLU":
                nonlin_fn = Approx_ReLU_grad
            else:
                nonlin_fn = partial(nonlin_fn, inplace=INPLACE)
            nonlin_fn()
        except TypeError:
            if nonlin_name == "Approx_ReLU":
                nonlin_fn = Approx_ReLU_grad
            else:
                nonlin_fn = getattr(torch.nn, nonlin_name)

        if wrap_in_glu:
            return partial(GLU_grad, nonlin_fn)
        else:
            return nonlin_fn

def _get_norm_fn_grad(norm_name):
    if norm_name == "ScaleNorm":
        # norm_fn = ScaleNorm
        ...
    elif norm_name == "RMSNorm":
        # norm_fn = RMSNorm
        ...
    elif norm_name == "ApexLayerNorm":
        from apex.normalization import FusedLayerNorm

        norm_fn = FusedLayerNorm
    elif norm_name == "Approx_LayerNorm":
        norm_fn = Approx_LayerNorm_grad
    else:
        norm_fn = getattr(torch.nn, norm_name)
    return norm_fn

class GLU_grad(torch.nn.Module):
    """*-GLU activation functions.

    Implementation mostly following megatron
    """

    def __init__(self, sub_activation):
        super().__init__()
        if sub_activation == 'GELU_poly_11':
            ...
        else:
            self.sub_activation = Approx_ReLU_grad()

    def forward(self, inputs):
        x, gate = inputs.chunk(2, dim=-1)
        return self.sub_activation(gate) * x
  
class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)

def subtraction_gaussian_kernel_torch(q, k):
    k = k.transpose(-1, -2) 
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
    
    return matA_square + matB_square - 2. * (q @ k)

def get_attention_mechanism(
    idx,
    hidden_size,
    cfg_attention,
):

    cfg_attention.type = cfg_attention['type']
    
    if  cfg_attention.type == "self-attention-modified":
        mechanism = SeqFirstSelfAttention_grad(hidden_size, cfg_attention)
    else:
        raise ValueError(f"Invalid attention type {cfg_attention.type} given.")
    return mechanism
   
class Exp_grad(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)

        if attention_mask is not None:
            inputs = inputs + attention_mask
        activ =  lambda x: torch.exp(x)
        outputs = activ(inputs)
        return outputs
 
class LegacySeqFirstSelfAttention_grad(torch.nn.Module):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    Self-attention layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def __init__(self, hidden_size: int, cfg_attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        # 64
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())
        
        # Linear(768, 2304)
        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=cfg_attention.qkv_bias)
        
        self.output_dim = hidden_size # 768
        self.rotary_emb = None
        self.sequence_op = Exp_grad(self.num_attention_heads, cfg_attention.seq_op_in_fp32)

        self.attention_dropout: float = cfg_attention.dropout_prob

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        mixed_x_layer = self.query_key_value(hidden_states) # 128 128 2304
      
        mixed_x_layer = mixed_x_layer.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, 3 * self.hidden_per_head
        )
        mixed_x_layer.retain_grad()
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.hidden_per_head] * 3, dim=3)

        # Ordinary
        context_layer, matmul_result, attention_outputs = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training) # C'
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous() # C''
       
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size) # C'''
        context_layer.retain_grad()
        return context_layer, matmul_result, attention_outputs
 
class SeqFirstSelfAttention_grad(LegacySeqFirstSelfAttention_grad):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    This is a modified version of the neo-x implementation that I can manage to compile without graph breaks
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False):
        attention_outputs = {}
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        
        query_layer_matmul = query_layer.transpose(0, 1)
        self.q = query_layer_matmul
        self.q.retain_grad()
        key_layer_matmul = key_layer.transpose(0, 1)
        self.k = key_layer_matmul
        self.k.retain_grad()
                
        matmul_result = self.subtraction_gaussian_kernel_torch(query_layer_matmul, key_layer_matmul)
        matmul_result = matmul_result * (-self.norm_factor) * 0.5
        self.GK_result = matmul_result
        self.GK_result.retain_grad()
        
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])
        
        attention_probs = self.sequence_op(attention_scores, attention_mask)
        self.after_exp = attention_probs
        self.after_exp.retain_grad()
        
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout, training=training)
        self.after_dropout = attention_probs
        self.after_dropout.retain_grad()
        
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        
        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        self.v = value_layer.transpose(0, 1)
        self.v.retain_grad()
        
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        self.transpose_of_dropout = attention_probs
        self.transpose_of_dropout.retain_grad()
        
        context_layer = torch.bmm(attention_probs, self.v) # C
        self.GK_v = context_layer
        self.GK_v.retain_grad()
        context_layer = context_layer.view(*output_size) # C'
        
        # Ordinary
        return context_layer, matmul_result, attention_outputs
    
    def subtraction_gaussian_kernel_torch(self, q, k):
        self.GK_q = q
        self.GK_q.retain_grad()
        
        k = k.transpose(-1, -2) 
        self.GK_k = k
        self.GK_k.retain_grad()
        
        matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
        self.matQ_square = matA_square
        self.matQ_square.retain_grad()
        
        matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
        self.matK_square = matA_square
        self.matK_square.retain_grad()
        
        return matA_square + matB_square - 2. * (q @ k)
        
 
class AttentionComponent_grad(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        self.attention_input = hidden_states
        self.attention_input.retain_grad()
        self.attention_output, matmul_result, attention_outputs = self.self_attention(self.attention_input, attention_mask)
        self.attention_output.retain_grad()
        self.after_dense = self.dense(self.attention_output)
        self.after_dense.retain_grad()
        
        return self.after_dense, matmul_result, attention_outputs

class FFNComponent_grad(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    It actually turned out better not to scale it, so here the block is effectively smaller than may be expected.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU_grad):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        self.before_dense_in = hidden_states
        self.before_dense_in.retain_grad()
        self.after_dense_in = self.dense_in(self.before_dense_in)
        self.after_dense_in.retain_grad()
            
        relu_inputs = self.after_dense_in.view(128,3072)

        self.after_nonlin = self.nonlin(self.after_dense_in)
        self.after_nonlin.retain_grad()
        self.ffn_output = self.dense_out(self.after_nonlin)
        self.ffn_output.retain_grad()
        
        return self.ffn_output

class TransformerLayer_grad(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.idx = idx
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        if cfg_arch.norm in ["Approx_LayerNorm"]:
            if idx == 0:
                div_max_1 = 10
                div_max_2 = 4000
                self.norm1 = _get_norm_fn_grad(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn_grad(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
            elif idx == 1:
                div_max_1 = 5000
                div_max_2 = 4000
                self.norm1 = _get_norm_fn_grad(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn_grad(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
            else:
                div_max_1 = 8000
                div_max_2 = 8000
                self.norm1 = _get_norm_fn_grad(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn_grad(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
                
        else:
            self.norm1 = _get_norm_fn_grad(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
            self.norm2 = _get_norm_fn_grad(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent_grad(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.cfg_arch = cfg_arch
        self.LAYOUT = self.attn.LAYOUT

        self.ffn = FFNComponent_grad(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            self.cfg_arch.get_input_range,
            _get_nonlin_fn_grad(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )  
        
    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):

        self.tf_inputs = states
        self.tf_inputs.retain_grad()


        norm1_inputs = self.tf_inputs
        var = torch.var(norm1_inputs.view(128,768), dim = 1)


        self.after_ln_before_att = self.norm1(self.tf_inputs)
        self.after_ln_before_att.retain_grad()
        self.after_att, matmul_result, attention_outputs = self.attn(self.after_ln_before_att, attention_mask)
        self.after_att.retain_grad()
        self.after_dropout_after_att = self.dropout(self.after_att)
        self.after_dropout_after_att.retain_grad()
        
        self.after_res_conn_after_att = self.tf_inputs + self.after_dropout_after_att
        self.after_res_conn_after_att.retain_grad()

        norm2_inputs = self.after_res_conn_after_att
        var = torch.var(norm2_inputs.view(128,768), dim = 1)


        self.after_ln_before_ffn = self.norm2(self.after_res_conn_after_att)
        self.after_ln_before_ffn.retain_grad()
        self.after_ffn = self.ffn(self.after_ln_before_ffn)
        self.after_ffn.retain_grad()
        self.after_dropout_after_ffn = self.dropout(self.after_ffn)
        self.after_dropout_after_ffn.retain_grad()
        
        self.tf_output = self.after_res_conn_after_att + self.after_dropout_after_ffn
        self.tf_output.retain_grad()
        
        return self.tf_output, matmul_result, attention_outputs, {'tf_inputs': self.tf_inputs, 'after_ln_before_att': self.after_ln_before_att, 'after_att': self.after_att,
                                       'after_dropout_after_att': self.after_dropout_after_att, 'after_res_conn_after_att': self.after_res_conn_after_att,
                                       'after_ln_before_ffn': self.after_ln_before_ffn, 'after_ffn': self.after_ffn, 'after_dropout_after_ffn': self.after_dropout_after_ffn,
                                       'tf_output': self.tf_output}


class EmbeddingComponent_grad(torch.nn.Module):
    def __init__(self, cfg_embedding, norm, norm_eps):
        super().__init__()
        self.word_embedding = torch.nn.Embedding(
            cfg_embedding.vocab_size, cfg_embedding.embedding_dim, padding_idx=cfg_embedding.pad_token_id
        )
        self.cfg_embedding = cfg_embedding
        if cfg_embedding.pos_embedding == "scaled-sinusoidal":
            self.pos_embedding = ScaledSinosoidal(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        else:
            self.pos_embedding = None
        
        self.dropout = torch.nn.Dropout(p=cfg_embedding.dropout_prob, inplace=INPLACE)
        if cfg_embedding.normalization:
            self.stabilize_low_precision = cfg_embedding.get("stable_low_precision", False) 
            norm = "Approx_LayerNorm"
            if norm in ["Approx_LayerNorm"]:
                div_max = 1
                self.norm = _get_norm_fn_grad(norm)(cfg_embedding.embedding_dim, div_max=div_max, eps=norm_eps)
            else:
                self.norm = _get_norm_fn_grad(norm)(cfg_embedding.embedding_dim, eps=norm_eps)
                
        else:
            self.stabilize_low_precision = False
            self.norm = torch.nn.Identity()

    def forward(self, input_ids):
        embeds = self.word_embedding(input_ids)
        after_word_embedding = embeds
        after_word_embedding.retain_grad()
        
        if self.pos_embedding is not None:
            embeds += self.pos_embedding(input_ids)
            after_pos_embedding = embeds
            after_pos_embedding.retain_grad()
        
        after_emb_ln = self.norm(embeds)
        after_emb_ln.retain_grad()
        after_emb_dropout = self.dropout(after_emb_ln)
        after_emb_dropout.retain_grad()
        hidden_states = after_emb_dropout
        return hidden_states, {'after_word_emb': after_word_embedding, 'after_pos_emb': after_pos_embedding, 'after_emb_ln': after_emb_ln, 'after_emb_dropout': after_emb_dropout}

class ScriptableLM_grad(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.cfg.embedding.get_emb_input_range = self.cfg.get_input_range

        self.embedding = EmbeddingComponent_grad(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer_grad(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
         
        self.seq_first = True
        self.use_causal_attention = self.cfg.attention.causal_attention

        self.number = 0
        self.save = None

        if self.cfg.final_norm:
            if self.cfg.norm in ["Approx_LayerNorm"]:
                div_max = 8000
                self.final_norm = _get_norm_fn_grad(self.cfg.norm)(self.cfg.hidden_size, div_max=div_max, eps=self.cfg.norm_eps)
            else:
                self.final_norm = _get_norm_fn_grad(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        matmuls = []
        attentions_outputs = []
        tf_outputs = []
        self.transformer_outputs = []
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)

            #########################################################################################
            ## Should be matched with the target folder name with ouput name                       ##
            ## For example, if the target folder name is 'mrpc_train_masks', which will be used    ## 
            ## the same name in the conversion and target data name is `mask(index)_2ly_mrpc.pth`, ##   
            ## we should set the output torch data as `mrpc_train_masks/mask(index)_2ly_mrpc.pth`  ##
            #########################################################################################
            attn_mask_cpu = attention_mask.cpu()
            
            if self.save_train_data:
                train_or_eval = 'train'
            else:
                train_or_eval = 'eval'
                
            torch.save(attn_mask_cpu, f'./fine-tuning_data/{self.cfg.task_name}_{train_or_eval}_masks/mask{self.number}_2ly_{self.cfg.task_name}.pth')
      
        hidden_states, embedding_outputs = self.embedding(input_ids)
    
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            embedding_outputs['tf_inputs_transpose'] = hidden_states
        
        ##########################################################################################
        ## Should be matched with the target folder name with ouput name                        ##
        ## For example, if the target folder name is 'mrpc_train_inputs', which will be used    ## 
        ## the same name in the conversion and target data name is `input(index)_2ly_mrpc.pth`, ##   
        ## we should set the output torch data as `mrpc_train_ipnuts/input(index)_2ly_mrpc.pth` ##
        ########################################################################################## 
        attn_mask_cpu = hidden_states.cpu()
        torch.save(attn_mask_cpu, f'./fine-tuning_data/{self.cfg.task_name}_{train_or_eval}_inputs/input{self.number}_2ly_{self.cfg.task_name}.pth')
        
        self.number += 1
        
        for i, layer_module in enumerate(self.layers):
            # Ordinary
            #######################################
            hidden_states, matmul, attention_outputs, tf_output = layer_module(hidden_states, attention_mask)
            hidden_states.retain_grad()
            self.transformer_outputs.append(hidden_states)
            tf_outputs.append(tf_output)
            attentions_outputs.append(attention_outputs)
          
            matmuls.append(matmul)
            
        if self.seq_first:
            self.last_tf_output = hidden_states
            self.last_tf_output.retain_grad()
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            self.after_final_transpose = hidden_states
            self.after_final_transpose.retain_grad()

        self.output = self.final_norm(hidden_states)
        self.output.retain_grad()
        return self.output, matmuls, embedding_outputs, attentions_outputs, tf_outputs
        
class ScriptableLMForPreTraining_grad(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        
        self.encoder = ScriptableLM_grad(config)
        
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
        self.tf_norm1_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.final_norm_inputs_var_maxs = []
        self.final_norm_inputs_var_mins = []
        self.final_norm_inputs_var_ratios = []
        self.nonlin_inputs_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.nonlin_inputs_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        if self.cfg.get_input_range:
            os.makedirs('norms')
        os.makedirs('loss')
        
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

        outputs, matmuls_from_enc , emb_outputs, attentions_outputs, tf_outputs = self.encoder(input_ids, attention_mask)    
        outputs = outputs.view(-1, outputs.shape[-1]) 

        if self.sparse_prediction and labels is not None:            
            masked_lm_loss = self._forward_sparse(outputs, labels) 
            original_loss = masked_lm_loss.item()
            
            self.loss_list.append(original_loss)
                    
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

        return {"loss": masked_lm_loss, "outputs": outputs}, emb_outputs, attentions_outputs, tf_outputs
  
class PoolingComponent_grad(torch.nn.Module):
    def __init__(self, cfg_head, main_model_hidden_size):
        super().__init__()
        self.dense = torch.nn.Linear(main_model_hidden_size, cfg_head.head_dim) if cfg_head.include_ff_layer else torch.nn.Identity()
        self.activation = _get_nonlin_fn_grad(cfg_head.nonlin, use_gating=False)()
        self.dropout = torch.nn.Dropout(cfg_head.classifier_dropout)
        self.pool_scheme: str = cfg_head.pooler

        self.get_input_range = cfg_head.get_input_range
        self.get_grad = cfg_head.get_grad

    def forward(self, hidden_states):
        self.pooler_input = hidden_states
        self.pooler_input.retain_grad()
            
        """A variety of pooling options. Some ignore the cls token. Input needs to be B S H."""
        if self.pool_scheme == "zero_index":
            self.first_token_tensor = self.pooler_input[:, 0]
            self.first_token_tensor.retain_grad()
            

        self.dense_out = self.dense(self.first_token_tensor) 
        self.dense_out.retain_grad()
        self.pooled_output = self.activation(self.dense_out)
        self.pooled_output.retain_grad()
        self.output = self.dropout(self.pooled_output)
        self.output.retain_grad()
        return self.output

class PoolingComponent_grad_lora(torch.nn.Module):
    def __init__(self, cfg_head, main_model_hidden_size):
        super().__init__()
        # 768 x 32
        self.lora_a = torch.nn.Linear(main_model_hidden_size, 32)
        # 32 x 1024
        self.lora_b = torch.nn.Linear(32, cfg_head.head_dim)
        
        self.activation = _get_nonlin_fn_grad(cfg_head.nonlin, use_gating=False)()
        self.dropout = torch.nn.Dropout(cfg_head.classifier_dropout)
        self.pool_scheme: str = cfg_head.pooler
        self.get_input_range = cfg_head.get_input_range
        self.get_grad = cfg_head.get_grad

    def forward(self, hidden_states):            
        """A variety of pooling options. Some ignore the cls token. Input needs to be B S H."""
        if self.pool_scheme == "zero_index":
            first_token_tensor = hidden_states[:, 0]
    

        dense_lora = self.lora_a(first_token_tensor)
        dense_out = self.lora_b(dense_lora)

        pooled_output = self.activation(dense_out)
        
        return self.dropout(pooled_output)

class ScriptableLMForSequenceClassification_grad(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels
        
        self.cfg.classification_head.get_input_range = self.cfg.get_input_range
        if not self.cfg.get_grad == None:
            self.cfg.classification_head.get_grad = self.cfg.get_grad

        self.encoder = ScriptableLM_grad(config)
        if self.cfg.poolinglora:
            self.pooler = PoolingComponent_grad_lora(self.cfg.classification_head, self.cfg.hidden_size)
        else:
            self.pooler = PoolingComponent_grad(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_graph_interval_loss_list = []
        
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
        
        encoder_output,  matmuls_from_enc, emb_outputs, attentions_outputs, tf_outputs = self.encoder(input_ids, attention_mask)

        pooler_output = self.pooler(encoder_output)
        self.logits = self.head(pooler_output)

        self.logits.retain_grad()
        
        
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
                    loss = loss_fct(self.logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(self.logits, labels)
            elif self.problem_type == "single_label_classification":

                loss_fct = Custom_CrossEntropyLoss_grad()

                loss = loss_fct(self.logits.view(-1, self.num_labels), labels.view(-1))

            elif self.problem_type == "multi_label_classification":

                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(self.logits, labels)
        else:
            loss = self.logits.new_zeros((1,))

        self.loss_list.append(loss.item())
        
        return dict(logits=self.logits, loss=loss), emb_outputs, attentions_outputs, tf_outputs