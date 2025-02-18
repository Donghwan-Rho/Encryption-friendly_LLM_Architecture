import torch
import os
from safetensors.torch import load_file


########################################################################
### Input data are converted in the below code                       ###
### To make an appropriate input data, we should generate the input  ###
### with the following code, where the output is a container.        ###
########################################################################

class Container(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def add(self, key: str, value: torch.Tensor):
        setattr(self, key, value)

DIM: int = 768
HIDDEN_DIM: int = 3072
HIDDEN_HALF_DIM: int = 1536
MAX_SEQ_LEN: int = 128

NUM_HEADS: int = 12
BATCH_SIZE: int = 16

# Caution: SUB_DIM = 128 (not 64)
SUBMATRIX_DIM: int = 128



def convert(weight: torch.Tensor, height: int, width: int) -> torch.Tensor:
    # print(f'weight: {weight.shape}')
    return (
        weight.transpose(0, 1)
        .view(
            height // SUBMATRIX_DIM,
            SUBMATRIX_DIM,
            width // SUBMATRIX_DIM,
            SUBMATRIX_DIM,
        )
        .transpose(1, 2)
    )


def convert_norm(weight: torch.Tensor) -> torch.Tensor:
    return (
        weight.repeat(MAX_SEQ_LEN, 1)
        .view(MAX_SEQ_LEN, DIM // SUBMATRIX_DIM, SUBMATRIX_DIM)
        .transpose(0, 1)
    )

def convert_norm_final(weight: torch.Tensor) -> torch.Tensor:
    return (
        weight.repeat(MAX_SEQ_LEN, 1)
        .view(MAX_SEQ_LEN, 1024 // SUBMATRIX_DIM, SUBMATRIX_DIM)
        .transpose(0, 1)
    )

# [2, 1024] -> 2 x [8 ,128, 128]
def convert_pooling_final(weight: torch.Tensor, height:int, width:int) ->torch.Tensor:
    tmp = torch.zeros(height,width)
    tmp[:,:2] = weight.transpose(0,1)
    return (
        tmp.view(8, 128, 128)
    )

def convert_pooling_bias(weight: torch.Tensor) -> torch.Tensor:
    tmp = torch.zeros(1024,128)
    tmp[:,:1] = weight.view(1024,1)
    return (
        tmp.view(8, 128, 128)
    )

def convert_pooling_weight(weight: torch.Tensor, height:int, width:int) -> torch.Tensor:
    return (
        weight.view(
            height // 128,
            128,
            width // 128,
            128,
        )
        .transpose(1, 2)
    )

def convert_embed_mask(weight: torch.Tensor) -> torch.Tensor:
    return (
        weight[0][0].reshape(1,128).repeat(128,1)
    )

def convert_pooling_weight_a(weight: torch.Tensor) -> torch.Tensor:
    weight = torch.split(weight, 128, dim = 1)
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    for i in range(3):
        data1[i*32:(i+1)*32, :] = weight[i*2]
        data2[i*32:(i+1)*32, :] = weight[i*2+1]
    return (torch.stack((data1, data2), dim = 0))

def convert_pooling_bias_a(weight: torch.Tensor) -> torch.Tensor:
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    data1[:32,0] = weight
    return (torch.stack((data1, data2), dim = 0))

def convert_pooling_weight_b(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.transpose(0,1)
    weight = torch.split(weight, 128, dim = 1)
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    for i in range(4):
        data1[i*32:(i+1)*32, :] = weight[i*2]
        data2[i*32:(i+1)*32, :] = weight[i*2+1]
    return (torch.stack((data1, data2), dim = 0))

def convert_pooling_bias_b(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.view(1,1024)
    weight = torch.split(weight, 128, dim = 1)
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    for i in range(4):
        data1[i*32:i*32+1, :] = weight[i*2]
        data2[i*32:i*32+1, :] = weight[i*2+1]
    return (torch.stack((data1, data2), dim = 0))

def convert_head_weight(weight: torch.Tensor) -> torch.Tensor:
    weight = torch.split(weight, 128, dim = 1)
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    for i in range(4):
        data1[i*32:i*32+2, :] = weight[i*2]
        data2[i*32:i*32+2, :] = weight[i*2+1]
    return (torch.stack((data1, data2), dim = 0))

def convert_head_bias(weight: torch.Tensor) -> torch.Tensor:
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    data1[:2,0] = weight
    return (torch.stack((data1, data2), dim = 0))






def main():

    ###################################################################################
    #### `weight_path` denotes the saved pre-train weight path (model.safetensors) ####
    #### For example, weight_path = `pre-trained_weights/{your_name}',             ####
    #### which corresponds to the defined path in plaintext running.               ####
    ###################################################################################
    weight_path = "Set your pre-trained weight path"

    ####################################################################
    #### `save_path` denotes the output weight path, which will be ####
    ####  used in the HE computation model.                         ####
    ####################################################################
    save_path = "Set output path that will be used as the same name in HE model"

    container = Container()

    weight = load_file(weight_path)

    # Extracting Q, K and V  weights from Mixed_QKV weight
    #####################################################
    q_idx = []
    k_idx = []
    v_idx = []
    for i in range(12 * 3 * 64):
        if (i % (3*64)) < 64:
            q_idx.append(i)
        elif (i % (3*64)) >= 64 and (i % (3*64)) < 2*64:
            k_idx.append(i)
        else:
            v_idx.append(i)

    ## Train ##
    # RTE: 2490
    # MRPC: 3668
    # COLA: 8551
    # STS-B: 5749
    # SST-2: 67349
    # QNLI: 104743

    #######################################################################################
    ## Torch path should be set as the same path in grad_test.py file,                   ##
    ## For example, since we set the train data set output path as                       ##
    ## `mrpc_train_inputs/input(index)_2ly_mrpc.pth`, we set the path as the following   ##
    ## The train/eval data are saved in plaintext/fine-tuning_data/. So, the target      ## 
    ## data path need to be set `plaintext/fine-tuning_data/(task)_train_inputs(masks)/  ##
    #######################################################################################

    for i in range(0,3668):
        inp = torch.load(f'/plaintext/fine-tuning_data/mrpc_train_inputs/input{i}_2ly_mrpc.pth')
        container.add(f'input_{i}', inp)
        inp = torch.load(f'/plaintext/fine-tuning_data/mrpc_train_masks/mask{i}_2ly_mrpc.pth')
        container.add(f'mask_{i}', inp)



    ## Eval ##
    # RTE: 277
    # MRPC: 408
    # COLA: 1043
    # STSB: 1500
    # SST2: 872
    # QNLI: 5463

    #################################################################################
    # Torch path should be set as the same path in grad_test.py file,              ##
    # For example, since we set the eval data output path as                       ##
    # `mrpc_eval_masks/mask(index)_2ly_mrpc.pth`, we set the path as the following ##
    #################################################################################

    #MRPC
    """ for i in range(0,408):
        inp = torch.load(f'./plaintext/fine-tuning_data/mrpc_eval_inputs/input{i}_2ly_mrpc.pth')
        container.add(f'input_{i}', inp)
        inp = torch.load(f'./plaintext/fine-tuning_data/mrpc_eval_masks/mask{i}_2ly_mrpc.pth')
        container.add(f'mask_{i}', inp) """



    for i in range(2):
        prefix = f'encoder.layers.{i}.'
        qkv_w = weight[f'{prefix}attn.self_attention.query_key_value.weight']
        w_q = qkv_w[q_idx]
        w_k = qkv_w[k_idx]
        w_v = qkv_w[v_idx]
        container.add(
            f'{i}_wq', convert(w_q, DIM, DIM)
        )
        container.add(
            f'{i}_wk', convert(w_k, DIM, DIM)
        )
        container.add(
            f'{i}_wv', convert(w_v, DIM, DIM)
        )
        container.add(
            f'{i}_wd', convert(weight[f'{prefix}attn.dense.weight'], DIM, DIM)
        )
        container.add(
            f'{i}_wdin', convert(weight[f'{prefix}ffn.dense_in.weight'], DIM, HIDDEN_DIM)
        )
        container.add(
            f'{i}_wdout', convert(weight[f'{prefix}ffn.dense_out.weight'], HIDDEN_HALF_DIM, DIM)
        )
        container.add(
            f'{i}_norm1_b', convert_norm(weight[f'{prefix}norm1.bias'])
        )
        container.add(
            f'{i}_norm1_w', convert_norm(weight[f'{prefix}norm1.weight'])
        )
        container.add(
            f'{i}_norm2_b', convert_norm(weight[f'{prefix}norm2.bias'])
        )
        container.add(
            f'{i}_norm2_w', convert_norm(weight[f'{prefix}norm2.weight'])
        )

        container.add(
            f'{i}_final_norm_w', convert_norm(weight['encoder.final_norm.weight'])
        )
        container.add(
            f'{i}_final_norm_b', convert_norm(weight['encoder.final_norm.bias'])
        )


    torch.jit.script(container).save(save_path)

if __name__ == "__main__":
    main()
