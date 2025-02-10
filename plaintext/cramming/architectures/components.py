"""Basic transformer components."""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
from .attention_modified import Block_Matmul_Module

from typing import Optional, Tuple
from functools import partial

from .embeddings import SinusoidalPositional, LearnablePositional, ScaledSinosoidal
from .attention import get_attention_mechanism

from termcolor import colored
import torch.nn.functional as F
import time
from datetime import datetime

INPLACE = False

class Custom_CrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(Custom_CrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits, labels):
        softmax_input_max = (logits / self.temperature).max().item() # logits / self.temperature: softmax input
        with open ('softmax_input_max.txt', 'a') as f:
            f.write(f'{softmax_input_max}\n')
        softmax_input_min = (logits / self.temperature).min().item()
        with open ('softmax_input_min.txt', 'a') as f:
            f.write(f'{softmax_input_min}\n')

        # LogSoftmax            
        log_probs = F.log_softmax(logits / self.temperature, dim=1)

        # Negative log likelihood loss
        loss = F.nll_loss(log_probs, labels)

        return loss
    
class EmbeddingComponent(torch.nn.Module):
    def __init__(self, cfg_embedding, norm, norm_eps):
        super().__init__()
        self.word_embedding = torch.nn.Embedding(
            cfg_embedding.vocab_size, cfg_embedding.embedding_dim, padding_idx=cfg_embedding.pad_token_id
        )
        if cfg_embedding.pos_embedding == "learned":
            self.pos_embedding = LearnablePositional(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "sinusoidal":
            self.pos_embedding = SinusoidalPositional(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "scaled-sinusoidal":
            self.pos_embedding = ScaledSinosoidal(cfg_embedding.embedding_dim, cfg_embedding.max_seq_length)
        else:
            self.pos_embedding = None
        self.dropout = torch.nn.Dropout(p=cfg_embedding.dropout_prob, inplace=INPLACE)
        if cfg_embedding.normalization: # True
            self.stabilize_low_precision = cfg_embedding.get("stable_low_precision", False)
            self.norm = _get_norm_fn(norm)(cfg_embedding.embedding_dim, eps=norm_eps)
        else:
            self.stabilize_low_precision = False
            self.norm = torch.nn.Identity()

    def forward(self, input_ids):
        embeds = self.word_embedding(input_ids)
        if self.pos_embedding is not None:
            embeds += self.pos_embedding(input_ids)

        if self.stabilize_low_precision:
            # Stabilize as in bnb StableEmbedding
            return self.dropout(self.norm(embeds.to(torch.get_default_dtype()))).to(embeds.dtype)
        else:
            return self.dropout(self.norm(embeds))
            

class EmbeddingComponent_modified(torch.nn.Module):
    def __init__(self, cfg, norm, norm_eps):
        # print('cfg_embedding', cfg_embedding)
        super().__init__()
        cfg_embedding = cfg.embedding
        
        if cfg.larger_embedding:
            self.real_emb_dim = cfg.larger_embedding_dim
        else:
            self.real_emb_dim = cfg_embedding.embedding_dim
        
        self.word_embedding = torch.nn.Embedding(
                cfg_embedding.vocab_size, self.real_emb_dim, padding_idx=cfg_embedding.pad_token_id
            )
        
        self.cfg_embedding = cfg_embedding
        if cfg_embedding.pos_embedding == "learned":
            self.pos_embedding = LearnablePositional(self.real_emb_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "sinusoidal":
            self.pos_embedding = SinusoidalPositional(self.real_emb_dim, cfg_embedding.max_seq_length)
        elif cfg_embedding.pos_embedding == "scaled-sinusoidal":
            self.pos_embedding = ScaledSinosoidal(self.real_emb_dim, cfg_embedding.max_seq_length)
        else:
            self.pos_embedding = None
        self.dropout = torch.nn.Dropout(p=cfg_embedding.dropout_prob, inplace=INPLACE)
        if cfg_embedding.normalization: # True
            self.stabilize_low_precision = cfg_embedding.get("stable_low_precision", False) # yaml 파일에서 false
            if norm in ["Approx_LayerNorm"]:
                div_max = 1
                # print(f'emb var div_max: {div_max}')
                self.norm = _get_norm_fn(norm)(self.real_emb_dim, div_max=div_max, eps=norm_eps)
            else:
                self.norm = _get_norm_fn(norm)(self.real_emb_dim, eps=norm_eps)
                
        else:
            self.stabilize_low_precision = False
            self.norm = torch.nn.Identity()

    def forward(self, input_ids):
        embeds = self.word_embedding(input_ids)

        if self.pos_embedding is not None:
            embeds += self.pos_embedding(input_ids)

        if self.stabilize_low_precision:
            # Stabilize as in bnb StableEmbedding
            return self.dropout(self.norm(embeds.to(torch.get_default_dtype()))).to(embeds.dtype)
        else:
            norm_inputs = embeds
            after_norm = self.norm(embeds)
            if self.cfg_embedding.get_emb_input_range:
                return self.dropout(after_norm), norm_inputs
            else:
                return self.dropout(after_norm)


class AttentionComponent(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        # print('cfg_attention.high_level_fusion', cfg_attention.high_level_fusion)

        if cfg_attention.high_level_fusion:
            self.self_attention = torch.jit.script(self.self_attention)

        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.dense(self.self_attention(hidden_states, attention_mask))


class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    Better do this manually and choose a sensible intermed_size that is nicely divisible.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU) or getattr(self.nonlin, "original_name", "") == "GLU":
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)

    def forward(self, hidden_states):
        return self.dense_out(self.nonlin(self.dense_in(hidden_states)))

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

class Approx_ReLU(nn.Module):
    def __init__(self, experiment_float64):
        super(Approx_ReLU, self).__init__()
        self.experiment_float64 = experiment_float64

    def forward(self, x):
        # Apply ReLU function: max(0, x)
        x = x.to(dtype=torch.float64)
       
        if self.experiment_float64: 
            return (ReLU_torch_64(x / 100) * 100)
        else:           
            return (ReLU_torch_64(x / 100) * 100).half()

class Approx_LayerNorm(nn.Module):
    def __init__(self, normalized_shape, div_max, eps=1e-5):
        super(Approx_LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.div_max = div_max
        self.sqrt = math.sqrt(div_max)
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)

        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        var_eps = var + self.eps
        var_eps = var_eps
        
        inverse_sqrt = Inverse_sqrt_approx((var_eps) / self.div_max) / self.sqrt
        inverse_sqrt = inverse_sqrt

        # x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        x_normalized = (x - mean) * inverse_sqrt
        
        return self.weight * x_normalized + self.bias

# Deg 11 Tanh approx using Least square
class Tanh_poly_11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # [-20, 20]
        self.activ = lambda x: -6.87734004e-02*((x/10)**11) - 1.91148650e-03*((x/100)**10)*1e-3\
                               + 8.09118502e-01*((x/10)**9) + 2.78022979e+02*((x/1000)**8)\
                               - 3.59640158e+00*((x/10)**7) - 1.31127527e-02*((x/1000)**6)\
                               + 7.50906080e+00*((x/10)**5) + 1.57859609e-03*((x/1000)**4)*1e-3\
                               - 7.51041204e+00*((x/10)**3) - 2.80712114e-03*((x/100)**2)*1e-4*1e-3*1e-3\
                               + 3.76675408e-01*(x**1) - 5.53103077e-4*1e-4*1e-4*1e-4
    def forward(self, inputs):
        return self.activ(inputs)

class PoolingComponent(torch.nn.Module):
    def __init__(self, cfg_head, main_model_hidden_size):
        super().__init__()
        self.dense = torch.nn.Linear(main_model_hidden_size, cfg_head.head_dim) if cfg_head.include_ff_layer else torch.nn.Identity()
        self.activation = _get_nonlin_fn(cfg_head.nonlin, use_gating=False)()
        self.dropout = torch.nn.Dropout(cfg_head.classifier_dropout)
        self.pool_scheme: str = cfg_head.pooler
        self.get_input_range = cfg_head.get_input_range
        self.get_grad = cfg_head.get_grad

    def forward(self, hidden_states):
        if self.get_grad:
            before_zero_indexing_hidden_states = hidden_states
            before_zero_indexing_hidden_states.retain_grad()
            
        """A variety of pooling options. Some ignore the cls token. Input needs to be B S H."""
        if self.pool_scheme == "zero_index":
            first_token_tensor = hidden_states[:, 0]
            if self.get_grad:
                first_token_tensor.retain_grad()
                
        elif self.pool_scheme == "avg":
            first_token_tensor = hidden_states.mean(dim=1)
        elif self.pool_scheme == "max":
            first_token_tensor = hidden_states.max(dim=1)[0]
        elif self.pool_scheme == "lse":
            first_token_tensor = hidden_states.logsumexp(dim=1)
        else:
            raise ValueError(f"Invalid pooling scheme {self.pool_scheme} given.")

        dense_out = self.dense(first_token_tensor) # Inputs of the activation ftn

        pooled_output = self.activation(dense_out)
        # Range of tanh output
        #######################################
        tanh_output_max = pooled_output.max().item()
        with open('tanh_output_max.txt', 'a') as f:
            f.write(f'{tanh_output_max}\n')
        tanh_output_min = pooled_output.min().item()
        with open('tanh_output_min.txt', 'a') as f:
            f.write(f'{tanh_output_min}\n')
        #######################################
        
        if self.get_input_range:
            return self.dropout(pooled_output), dense_out
        elif self.get_grad:
            return self.dropout(pooled_output), before_zero_indexing_hidden_states, first_token_tensor
        else:
            return self.dropout(pooled_output)

class PoolingComponent_lora(torch.nn.Module):
    def __init__(self, cfg_head, main_model_hidden_size):
        super().__init__()
        # 768 x 32
        self.lora_a = torch.nn.Linear(main_model_hidden_size, 32)
        # 32 x 1024
        self.lora_b = torch.nn.Linear(32, cfg_head.head_dim)
        
        self.activation = _get_nonlin_fn(cfg_head.nonlin, cfg_head.experiment_float64, use_gating=False)()
        self.dropout = torch.nn.Dropout(cfg_head.classifier_dropout)
        self.pool_scheme: str = cfg_head.pooler
        
        self.get_input_range = cfg_head.get_input_range
        self.get_grad = cfg_head.get_grad
        
        self.temperature = cfg_head.temperature

    def forward(self, hidden_states):            
        """A variety of pooling options. Some ignore the cls token. Input needs to be B S H."""
        if self.pool_scheme == "zero_index":
            first_token_tensor = hidden_states[:, 0]
            if self.get_grad:
                first_token_tensor.retain_grad()
                
        dense_lora = self.lora_a(first_token_tensor)        
        dense_out = self.lora_b(dense_lora) / self.temperature # inputs of tanh
        
        pooled_output = self.activation(dense_out)
        # Range of tanh output
        #######################################
        tanh_output_max = pooled_output.max().item()
        with open('tanh_output_max.txt', 'a') as f:
            f.write(f'{tanh_output_max}\n')
        tanh_output_min = pooled_output.min().item()
        with open('tanh_output_min.txt', 'a') as f:
            f.write(f'{tanh_output_min}\n')
        #######################################
        
        if self.get_input_range:
            return self.dropout(pooled_output), dense_out
        else:
            return self.dropout(pooled_output)


class PredictionHeadComponent(torch.nn.Module):
    def __init__(self, cfg_arch):
        super().__init__()

        if cfg_arch.embedding.embedding_dim == cfg_arch.hidden_size:
            output_size = cfg_arch.hidden_size
        else:
            output_size = cfg_arch.embedding.embedding_dim

        self.dense = torch.nn.Linear(cfg_arch.hidden_size, output_size, bias=cfg_arch.use_bias)
        self.nonlin = _get_nonlin_fn(cfg_arch.nonlin, use_gating=False)()
        self.norm = _get_norm_fn(cfg_arch.norm)(output_size, eps=cfg_arch.norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.norm(self.nonlin(self.dense(hidden_states)))
        return hidden_states


def _get_norm_fn(norm_name):
    if norm_name == "ScaleNorm":
        norm_fn = ScaleNorm
    elif norm_name == "RMSNorm":
        norm_fn = RMSNorm
    elif norm_name == "ApexLayerNorm":
        from apex.normalization import FusedLayerNorm

        norm_fn = FusedLayerNorm
    elif norm_name == "Approx_LayerNorm":
        norm_fn = Approx_LayerNorm
    else:
        norm_fn = getattr(torch.nn, norm_name)
    return norm_fn



def _get_nonlin_fn(nonlin_name, experiment_float64, use_gating=True):
    if nonlin_name == 'GELU_poly_11glu':
        if 'glu' in nonlin_name.lower():
            nonlin_name = nonlin_name.split("glu")[0] # GELU
            wrap_in_glu = use_gating # True
            nonlin_fn = nonlin_name # 'GELU_poly_11glu'
        else:
            wrap_in_glu = False
        if wrap_in_glu:
            return partial(GLU, nonlin_fn)
    elif nonlin_name == 'Tanh_poly_11':
        return Tanh_poly_11  
    else:
        if "glu" in nonlin_name.lower():
            nonlin_name = nonlin_name.split("glu")[0] # GELU or ReLU or SiLU
            wrap_in_glu = use_gating # True
        else:
            wrap_in_glu = False
        if nonlin_name == "Approx_ReLU":
            nonlin_fn = Approx_ReLU
        else:
            nonlin_fn = getattr(torch.nn, nonlin_name)  # dont mess this up :<
        try:
            if nonlin_name == "Approx_ReLU":
                nonlin_fn = Approx_ReLU
            else:
                nonlin_fn = partial(nonlin_fn, inplace=INPLACE)
            nonlin_fn()
        except TypeError:
            if nonlin_name == "Approx_ReLU":
                nonlin_fn = Approx_ReLU
            else:
                nonlin_fn = getattr(torch.nn, nonlin_name)

        if wrap_in_glu:
            return partial(GLU, nonlin_fn, experiment_float64)
        else:
            return nonlin_fn

# Deg 11 GLUE approx using Least square
class GLUE_poly_11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # [-30, 30]
        # Working only in float32
        # self.activ = lambda x: 3.71399574e-23*(x**11) + 2.29255806e-13*(x**10)\
        #                         - 9.58849981e-20*(x**9) - 5.68210065e-10*(x**8)\
        #                         + 9.21841075e-17*(x**7) + 5.26548340e-07*(x**6)\
        #                         - 4.03724289e-14*(x**5) - 2.31114239e-04*(x**4)\
        #                         + 7.78918393e-12*(x**3) + 6.14559207e-02*(x**2)\
        #                         + 4.99999999e-01*(x**1) + 7.74688089e-01
        # Working both in float32 and float16
        self.activ = lambda x: 3.71399574e-1*((x/100)**11) + 2.29255806e-3*((x/10)**10)\
                               - 9.58849981e-2*((x/100)**9) - 5.68210065e-2*((x/10)**8)\
                               + 9.21841075e-3*((x/100)**7) + 5.26548340e-01*((x/10)**6)\
                               - 4.03724289e-4*((x/100)**5) - 2.31114239e-00*((x/10)**4)\
                               + 7.78918393e-3*((x/1000)**3) + 6.14559207e-02*(x**2)\
                               + 4.99999999e-01*(x**1) + 7.74688089e-01
    def forward(self, inputs):
        return self.activ(inputs)
 
class CustomSiLU(nn.Module):
    def __init__(self):
        super(CustomSiLU, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class GLU(torch.nn.Module):
    """*-GLU activation functions.

    Implementation mostly following megatron
    """

    def __init__(self, sub_activation, experiment_float64):
        super().__init__()
        if sub_activation == 'GELU_poly_11':
            self.sub_activation = GLUE_poly_11()
        elif sub_activation == Approx_ReLU:
            self.sub_activation = sub_activation(experiment_float64)
        else:
            self.sub_activation = sub_activation()

    def forward(self, inputs):
        x, gate = inputs.chunk(2, dim=-1)
        atcivated_gate = self.sub_activation(gate)
        return atcivated_gate * x


class ScaleNorm(torch.nn.Module):
    """Quick and simple scale norm implementation.

    Do we also need FixNorm (cosine in the last layer)? It's a maybe here:
    https://github.com/lucidrains/performer-pytorch/issues/55#issuecomment-762544686
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.learnable_scale = torch.nn.Parameter(torch.tensor(float(hidden_size) ** -0.5))

    def forward(self, inputs):
        """This is the same eps clipping as in the original ScaleNorm implementation."""
        return inputs * self.learnable_scale / torch.norm(inputs, dim=-1, keepdim=True).clamp(min=self.eps)


class RMSNorm(torch.nn.Module):
    """The RMS variant of scaling norms."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.learnable_scale = torch.nn.Parameter(torch.ones(hidden_size) ** -0.5)

    def _legacy_forward(self, inputs):
        """This is the same eps clipping as in the original ScaleNorm implementation."""
        return inputs * self.learnable_scale / torch.norm(inputs, dim=-1, keepdim=True).clamp(min=1e-8)

    def _norm(self, x):
        """LLama implementation"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.learnable_scale


class Sequential(torch.nn.Module):
    """Modified sequential class."""

    def __init__(self, list_of_modules):
        super().__init__()
        self.seq_modules = torch.nn.ModuleList(list_of_modules)
        self.LAYOUT = self.seq_modules[0].LAYOUT

    def forward(self, states, *args, **kwargs):
        for module in self.seq_modules:
            states = module(states, *args, **kwargs)
        return states


def get_extended_attention_mask(attention_mask: torch.Tensor, input_shape: Tuple[int], causal_attention: bool = False) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.

    Method stolen from huggingface :)
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if causal_attention:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=attention_mask.device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones((batch_size, seq_length, prefix_seq_len), device=attention_mask.device, dtype=causal_mask.dtype),
                        causal_mask,
                    ],
                    axis=-1,
                )
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})")

    # extended_attention_mask = extended_attention_mask.to(dtype=self.setup["dtype"])  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


"""Collect inits."""


@torch.no_grad()
def _init_module(module, init_method="normal", init_std=0.02, hidden_size=768, num_layers=12):
    if init_method == "normal":
        std = init_std
    elif init_method == "small":
        # Transformers without Tears: Improving
        # the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010)
        std = torch.as_tensor(2 / (5 * hidden_size)).sqrt()
    elif init_method == "megatron":
        std = torch.as_tensor(1 / (3 * hidden_size)).sqrt()
    elif init_method == "wang":
        std = 2 / num_layers / torch.as_tensor(hidden_size).sqrt()
    elif init_method == "deepnorm":
        std = torch.as_tensor(8 * num_layers).pow(-0.25)  # todo: apply this only to some layers
    elif init_method == "agd-orthogonal":
        std = init_std  # no std modification necessary, setting to default
    else:
        raise ValueError(f"Invalid init method {init_method} given.")

    if isinstance(module, torch.nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, Block_Matmul_Module):
        module.weight.data.normal_(mean=0.0, std=std)
        

    if init_method == "agd-orthogonal":
        for name, p in module.named_parameters():
            if p.dim() == 1:
                print(f"WARNING: Biases are not supported. This breaks scaling of parameter {name} in theory.")
            if p.dim() == 2:
                torch.nn.init.orthogonal_(p)
                p *= singular_value(p.shape)
            if p.dim() == 4:
                for kx in range(p.shape[2]):
                    for ky in range(p.shape[3]):
                        torch.nn.init.orthogonal_(p[:, :, kx, ky])
                p *= singular_value(p.shape)


def singular_value(p_shape):
    """requires hashable input"""
    sv = math.sqrt(p_shape[0] / p_shape[1])
    if len(p_shape) == 4:
        sv /= math.sqrt(p_shape[2] * p_shape[3])
    return sv
