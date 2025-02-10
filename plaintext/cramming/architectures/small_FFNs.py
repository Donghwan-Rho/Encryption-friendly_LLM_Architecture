import torch
from .attention_modified import Exp
from typing import Optional

def subtraction_gaussian_kernel_torch(q, k):
    k = k.transpose(-1, -2) 
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
    # print('matA_square', matA_square.dtype)
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)

class FFNComponent_SmallMatmul_1(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        intermed_output_size = intermed_size // 2
        self.FFN_W = torch.nn.Linear(128, intermed_output_size, bias=False)
        self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        hidden_states_id = hidden_states
        # print(f'hidden_states: {hidden_states.shape}')
        hidden_states_dense_in = self.dense_in(hidden_states)
        concated_hidden_states = torch.cat((hidden_states_id, hidden_states_dense_in), dim=-1)
        concated_hidden_states = self.nonlin(concated_hidden_states)
        # print(f'concated_hidden_states: {concated_hidden_states.shape}')
        # print(f'self.FFN_W.weight.data: {self.FFN_W.weight.data.shape}')
        FFN_W_weight = self.dropout(self.FFN_W.weight.data).T.unsqueeze(1)
        # print(f'FFN_W_weight: {FFN_W_weight.shape}')
        Hadamard_output = concated_hidden_states * FFN_W_weight
        dense_output = self.dense_out(Hadamard_output)
        
        if self.get_input_range:
            return dense_output, concated_hidden_states
        else:
            return dense_output

class FFNComponent_SmallMatmul_2(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        self.intermed_output_size = intermed_size // 2
        self.FFN_W = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.dropout = torch.nn.Dropout(0.1, inplace=False)
        # self.dense_out = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        hidden_states_id = hidden_states
        hidden_states_dense_in = self.dense_in(hidden_states)
        concated_hidden_states = torch.cat((hidden_states_id, hidden_states_dense_in), dim=-1)
        concated_hidden_states = self.nonlin(concated_hidden_states)
        
        FFN_W_weight = self.dropout(self.FFN_W.weight.data).T.unsqueeze(1)
        Hadamard_output = concated_hidden_states * FFN_W_weight
        Hadamard_A = Hadamard_output[:, :, :self.intermed_output_size//2]
        # Hadamard_A = self.dense_out(Hadamard_A)
        Hadamard_B = Hadamard_output[:, :, self.intermed_output_size//2:]
        dense_output = Hadamard_A * Hadamard_B
        
        if self.get_input_range:
            return dense_output, concated_hidden_states
        else:
            return dense_output

class FFNComponent_SmallMatmul_3(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        # self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        self.intermed_output_size = intermed_size
        self.FFN_W_weight1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_weight2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_weight3 = torch.nn.Linear(128, hidden_size, bias=False)
        # self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.FFN_W_bias1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_bias2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_bias3 = torch.nn.Linear(128, hidden_size)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        # print(f'hidden_states: {hidden_states.shape}')
        Y = torch.cat((hidden_states, hidden_states, hidden_states, hidden_states), dim=-1) * self.FFN_W_weight1.weight.data.T.unsqueeze(1) + self.FFN_W_bias1.weight.data.T.unsqueeze(1) # 4h
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1) # 2h, 2h
        Y_2 = self.nonlin(Y_2)
        # print(f'self.FFN_W_weight2.weight.data.T.unsqueeze(1): {self.FFN_W_weight2.weight.data.T.unsqueeze(1).shape}')
        Z = Y_1 * Y_2 * self.FFN_W_weight2.weight.data.T.unsqueeze(1) + self.FFN_W_bias2.weight.data.T.unsqueeze(1) # 2h
        Z_1, Z_2 = torch.chunk(Z, 2, dim=-1) # h, h
        Z_2 = self.nonlin(Z_2)
        
        output = Z_1 * Z_2  * self.FFN_W_weight3.weight.data.T.unsqueeze(1) + self.FFN_W_bias3.weight.data.T.unsqueeze(1) # h
        # print(f'output: {output.shape}')
        
        
        if self.get_input_range:
            return output, Y_2
        else:
            return output

class FFNComponent_SmallMatmul_4(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        # self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        self.intermed_output_size = intermed_size
        self.FFN_W_weight1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_weight2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_weight3 = torch.nn.Linear(128, hidden_size, bias=False)
        self.dense_output = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.FFN_W_bias1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_bias2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_bias3 = torch.nn.Linear(128, hidden_size)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        # print(f'hidden_states: {hidden_states.shape}')
        Y = torch.cat((hidden_states, hidden_states, hidden_states, hidden_states), dim=-1) * self.FFN_W_weight1.weight.data.T.unsqueeze(1) + self.FFN_W_bias1.weight.data.T.unsqueeze(1) # 4h
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1) # 2h, 2h
        Y_2 = self.nonlin(Y_2)
        # print(f'self.FFN_W_weight2.weight.data.T.unsqueeze(1): {self.FFN_W_weight2.weight.data.T.unsqueeze(1).shape}')
        Z = Y_1 * Y_2 * self.FFN_W_weight2.weight.data.T.unsqueeze(1) + self.FFN_W_bias2.weight.data.T.unsqueeze(1) # 2h
        Z_1, Z_2 = torch.chunk(Z, 2, dim=-1) # h, h
        Z_2 = self.nonlin(Z_2)
        
        Z_3 = Z_1 * Z_2  * self.FFN_W_weight3.weight.data.T.unsqueeze(1) + self.FFN_W_bias3.weight.data.T.unsqueeze(1) # h
        output = self.dense_output(Z_3) + hidden_states
        # print(f'output: {output.shape}')
        
        
        if self.get_input_range:
            return output, Y_2
        else:
            return output

class FFNComponent_SmallMatmul_5(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        # self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        self.intermed_output_size = intermed_size
        self.FFN_W_weight1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_weight2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_weight3 = torch.nn.Linear(128, hidden_size, bias=False)
        self.dense_output = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.FFN_W_bias1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_bias2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_bias3 = torch.nn.Linear(128, hidden_size)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        # print(f'hidden_states: {hidden_states.shape}')
        Y = torch.cat((hidden_states, hidden_states, hidden_states, hidden_states), dim=-1) * self.FFN_W_weight1.weight.data.T.unsqueeze(1) + self.FFN_W_bias1.weight.data.T.unsqueeze(1) # 4h
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1) # 2h, 2h
        Y_2 = self.nonlin(Y_2) + Y_1
        # print(f'self.FFN_W_weight2.weight.data.T.unsqueeze(1): {self.FFN_W_weight2.weight.data.T.unsqueeze(1).shape}')
        Z = Y_1 * Y_2 * self.FFN_W_weight2.weight.data.T.unsqueeze(1) + self.FFN_W_bias2.weight.data.T.unsqueeze(1) # 2h
        Z_1, Z_2 = torch.chunk(Z, 2, dim=-1) # h, h
        Z_2 = self.nonlin(Z_2) + Z_1
        
        Z_3 = Z_1 * Z_2  * self.FFN_W_weight3.weight.data.T.unsqueeze(1) + self.FFN_W_bias3.weight.data.T.unsqueeze(1) # h
        output = self.dense_output(Z_3) + hidden_states
        # print(f'output: {output.shape}')
        
        
        if self.get_input_range:
            return output, Y_2
        else:
            return output

class FFNComponent_SmallMatmul_6(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        # self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        self.intermed_output_size = intermed_size
        self.FFN_W_weight4 = torch.nn.Linear(128, self.intermed_output_size * 2, bias=False)
        self.FFN_W_weight3 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_weight2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_weight1 = torch.nn.Linear(128, hidden_size, bias=False)
        self.dense_output = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.FFN_W_bias4 = torch.nn.Linear(128, self.intermed_output_size * 2, bias=False)
        self.FFN_W_bias3 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_bias2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_bias1 = torch.nn.Linear(128, hidden_size)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        # print(f'hidden_states: {hidden_states.shape}')
        x = torch.cat((hidden_states, hidden_states, hidden_states, hidden_states, hidden_states, hidden_states, hidden_states, hidden_states), dim=-1) * self.FFN_W_weight4.weight.data.T.unsqueeze(1) + self.FFN_W_bias4.weight.data.T.unsqueeze(1) # 8h
        x_1, x_2 = torch.chunk(x, 2, dim=-1) # 4h, 4h
        Y = x_1 * x_2 * self.FFN_W_weight3.weight.data.T.unsqueeze(1) + self.FFN_W_bias3.weight.data.T.unsqueeze(1) + x_1 # 4h
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1) # 2h, 2h
        Y_2 = self.nonlin(Y_2) + Y_1
        # print(f'self.FFN_W_weight2.weight.data.T.unsqueeze(1): {self.FFN_W_weight2.weight.data.T.unsqueeze(1).shape}')
        Z = Y_1 * Y_2 * self.FFN_W_weight2.weight.data.T.unsqueeze(1) + self.FFN_W_bias2.weight.data.T.unsqueeze(1) # 2h
        Z_1, Z_2 = torch.chunk(Z, 2, dim=-1) # h, h
        Z_2 = self.nonlin(Z_2) + Z_1
        
        Z_3 = Z_1 * Z_2  * self.FFN_W_weight1.weight.data.T.unsqueeze(1) + self.FFN_W_bias1.weight.data.T.unsqueeze(1) # h
        output = self.dense_output(Z_3) + hidden_states
        # print(f'output: {output.shape}')
        
        
        if self.get_input_range:
            return output, Y_2
        else:
            return output

class FFNComponent_SmallMatmul_7(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        # self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        self.intermed_output_size = intermed_size
        self.FFN_W_weight1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_weight2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_weight3 = torch.nn.Linear(128, hidden_size, bias=False)
        self.dense_output = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.FFN_W_bias1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_bias2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_bias3 = torch.nn.Linear(128, hidden_size)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        # print(f'hidden_states: {hidden_states.shape}')
        matmul_output = self.dense_output(hidden_states)
        Y = torch.cat((hidden_states, hidden_states, matmul_output, matmul_output), dim=-1) * self.FFN_W_weight1.weight.data.T.unsqueeze(1) + self.FFN_W_bias1.weight.data.T.unsqueeze(1) # 4h
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1) # 2h, 2h
        Y_2 = self.nonlin(Y_2) + Y_1
        # print(f'self.FFN_W_weight2.weight.data.T.unsqueeze(1): {self.FFN_W_weight2.weight.data.T.unsqueeze(1).shape}')
        Z = Y_1 * Y_2 * self.FFN_W_weight2.weight.data.T.unsqueeze(1) + self.FFN_W_bias2.weight.data.T.unsqueeze(1) # 2h
        Z_1, Z_2 = torch.chunk(Z, 2, dim=-1) # h, h
        Z_2 = self.nonlin(Z_2) + Z_1
        
        Z_3 = Z_1 * Z_2  * self.FFN_W_weight3.weight.data.T.unsqueeze(1) + self.FFN_W_bias3.weight.data.T.unsqueeze(1) # h
        output = Z_3 + hidden_states
        # print(f'output: {output.shape}')
        
        
        if self.get_input_range:
            return output, Y_2
        else:
            return output

class FFNComponent_SmallMatmul_8(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        # self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        self.intermed_output_size = intermed_size
        self.FFN_W_weight1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_weight2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_weight3 = torch.nn.Linear(128, hidden_size, bias=False)
        self.dense_output = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.FFN_W_bias1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_bias2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_bias3 = torch.nn.Linear(128, hidden_size)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        # print(f'hidden_states: {hidden_states.shape}')
        concat = torch.cat((hidden_states, hidden_states), dim=-1)
        matmul_output = self.dense_output(hidden_states)
        Y = torch.cat((matmul_output, matmul_output, matmul_output, matmul_output), dim=-1) * self.FFN_W_weight1.weight.data.T.unsqueeze(1) + self.FFN_W_bias1.weight.data.T.unsqueeze(1) # 4h
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1) # 2h, 2h
        Y_2 = self.nonlin(Y_2) + concat
        # print(f'self.FFN_W_weight2.weight.data.T.unsqueeze(1): {self.FFN_W_weight2.weight.data.T.unsqueeze(1).shape}')
        Z = Y_1 * Y_2 * self.FFN_W_weight2.weight.data.T.unsqueeze(1) + self.FFN_W_bias2.weight.data.T.unsqueeze(1) # 2h
        Z_1, Z_2 = torch.chunk(Z, 2, dim=-1) # h, h
        Z_2 = self.nonlin(Z_2) + Z_1
        
        Z_3 = Z_1 * Z_2  * self.FFN_W_weight3.weight.data.T.unsqueeze(1) + self.FFN_W_bias3.weight.data.T.unsqueeze(1) # h
        output = Z_3 + hidden_states
        # print(f'output: {output.shape}')
        
        if self.get_input_range:
            return output, Y_2
        else:
            return output

class FFNComponent_SmallMatmul_9(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        # self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        self.intermed_output_size = intermed_size
        self.FFN_W_weight1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_weight2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_weight3 = torch.nn.Linear(128, hidden_size, bias=False)
        self.dense_output = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        # self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.FFN_W_bias1 = torch.nn.Linear(128, self.intermed_output_size, bias=False)
        self.FFN_W_bias2 = torch.nn.Linear(128, self.intermed_output_size // 2, bias=False)
        self.FFN_W_bias3 = torch.nn.Linear(128, hidden_size)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        # print(f'hidden_states: {hidden_states.shape}')
        concat = torch.cat((hidden_states, hidden_states), dim=-1)
        matmul_output = self.dense_output(hidden_states)
        Y = torch.cat((hidden_states, hidden_states, matmul_output, matmul_output), dim=-1) * self.FFN_W_weight1.weight.data.T.unsqueeze(1) + self.FFN_W_bias1.weight.data.T.unsqueeze(1) # 4h
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1) # 2h, 2h
        Y_2 = self.nonlin(Y_2) + concat
        # print(f'self.FFN_W_weight2.weight.data.T.unsqueeze(1): {self.FFN_W_weight2.weight.data.T.unsqueeze(1).shape}')
        Z = Y_1 * Y_2 * self.FFN_W_weight2.weight.data.T.unsqueeze(1) + self.FFN_W_bias2.weight.data.T.unsqueeze(1) # 2h
        Z_1, Z_2 = torch.chunk(Z, 2, dim=-1) # h, h
        Z_2 = self.nonlin(Z_2) + Z_1
        
        Z_3 = Z_1 * Z_2  * self.FFN_W_weight3.weight.data.T.unsqueeze(1) + self.FFN_W_bias3.weight.data.T.unsqueeze(1) # h
        output = Z_3 + hidden_states
        # print(f'output: {output.shape}')
        
        if self.get_input_range:
            return output, Y_2
        else:
            return output

class FFNComponent_SmallMatmul_10(torch.nn.Module):
    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()        
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.nonlin = torch.nn.ReLU()
        intermed_output_size = intermed_size // 2
        self.FFN_W = torch.nn.Linear(128, intermed_output_size, bias=False)
        self.FFN_bias = torch.nn.Linear(128, hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(0.1, inplace=False)
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        hidden_states_dense_in = self.dense_in(hidden_states)
        after_nonlin = self.nonlin(hidden_states_dense_in) + hidden_states
        concated_hidden_states = torch.cat((hidden_states_dense_in, after_nonlin), dim=-1)
        # print(f'concated_hidden_states: {concated_hidden_states.shape}')
        # print(f'self.FFN_W.weight.data: {self.FFN_W.weight.data.shape}')
        FFN_W_weight = self.dropout(self.FFN_W.weight.data).T.unsqueeze(1)
        FFN_W_bias = self.dropout(self.FFN_bias.weight.data).T.unsqueeze(1)
        # print(f'FFN_W_weight: {FFN_W_weight.shape}')
        Hadamard_output = concated_hidden_states * FFN_W_weight
        dense_output = self.dense_out(Hadamard_output) + FFN_W_bias + hidden_states
        
        if self.get_input_range:
            return dense_output, concated_hidden_states
        else:
            return dense_output

