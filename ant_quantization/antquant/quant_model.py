import torch
import torch.nn as nn
import numpy as np
import copy
from quant_modules import TensorQuantizer, Conv2dQuantizer, LinearQuantizer
from multihead_attention import MultiheadAttentionQuantizer
from quant_utils import *
import torch.distributed as dist


def quantize_model(model ,quant_args):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # quantize layers
    # global quant_args
    if type(model) == nn.Conv2d:
        quant_mod = Conv2dQuantizer(**quant_args)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = LinearQuantizer(**quant_args)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.MultiheadAttention:
        quant_mod = MultiheadAttentionQuantizer(**quant_args)
        quant_mod.set_param(model)
        return quant_mod
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(model= m ,quant_args= quant_args))
        return nn.Sequential(*mods)
    elif type(model) == nn.ModuleList:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(model= m ,quant_args= quant_args))
        return nn.Sequential(*mods)
    elif isinstance(model, nn.Sequential):
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(model= m ,quant_args= quant_args))
        return nn.Sequential(*mods)
    else:
        # recursively use the quantized module to replace the single-precision module
        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                setattr(q_model, attr, quantize_model(model= mod ,quant_args= quant_args))
        return q_model

def set_first_last_layer(model):
    module_list_weight = []
    module_list_input = []
    for m in model.modules():
        if isinstance(m, TensorQuantizer) and m.is_input == False:
            module_list_weight += [m]
        if isinstance(m, TensorQuantizer) and m.is_input == True:
            module_list_input += [m]

def set_8_bit_layer_l(model, layer_list):
    if layer_list == "None":
        return
    layer_list = list(map(lambda x: int(x), layer_list.split(',')))
    module_list = []

    for m in model.modules():
        if isinstance(m, TensorQuantizer):            
            module_list += [m]
            m.has_inited_quant_para.data = torch.zeros_like(m.has_inited_quant_para)  

    if dist.get_rank() == 0:
        print("------------- 8-bit Re-SET -------------")
        print(len(layer_list))
    assert len(layer_list) > 0

    for i in range(int(len(module_list) / 2)):
        if i in layer_list:
            if dist.get_rank() == 0:
                print(module_list[i * 2].name, i )
                print(module_list[i * 2 + 1].name, i)
            module_list[i*2].bit.data = torch.tensor(8, device=module_list[i*2].bit.device)
            module_list[i*2+1].bit.data = torch.tensor(8, device=module_list[i*2+1].bit.device)

    if dist.get_rank() == 0:
        print("------------- 8-bit Re-SET -------------")

def set_8_bit_layer_n(model, l_num):
    #set l_num layers with 8-bit
    module_list = []
    mse_list    = []

    mse_linear_list = []
    linear_list = []
    mse = 0
    for m in model.modules():
        if isinstance(m, TensorQuantizer):            
            module_list += [m]
            mse_list    += [m.mse.item()]
            mse += m.mse.item()
            m.has_inited_quant_para.data = torch.zeros_like(m.has_inited_quant_para)  

    if dist.get_rank() == 0:
        print("------------- 8-bit Re-SET -------------")
        print(l_num)
    assert l_num > 0
    l_num *= 2

    first_num = 0 * 2
    for i in range(0, first_num):
        if dist.get_rank() == 0:
            print(module_list[i].name)
        module_list[i].bit.data = torch.tensor(8, device=module_list[i].bit.device)

    # For BERT last n layers.
    last_num = 2 * 2
    for i in range(len(mse_list) - last_num, len(mse_list)):
        if dist.get_rank() == 0:
            print(module_list[i].name)
        module_list[i].bit.data = torch.tensor(8, device=module_list[i].bit.device)

    if dist.get_rank() == 0:
        print("------------- First and Last end -------------")


    module_list = module_list[first_num: len(mse_list) - last_num]
    mse_list = mse_list[first_num: len(mse_list) - last_num]

    mse_list_pair = []
    for i in range(0, int(len(mse_list)/ 2)):
        mse_list_pair += [mse_list[i * 2] + mse_list[i * 2 + 1]]

    mses = np.array(mse_list_pair)
    mse_idx = np.argsort(-mses)
    l_num -= first_num
    l_num -= last_num
    l_num = int(l_num / 2)

    if l_num > 0:
        for i in mse_idx[0:l_num]:
            if dist.get_rank() == 0:
                print(module_list[i * 2].name, mses[i], i )
                print(module_list[i * 2 + 1].name, mses[i], i)
            module_list[i*2].bit.data = torch.tensor(8, device=module_list[i*2].bit.device)
            module_list[i*2+1].bit.data = torch.tensor(8, device=module_list[i*2+1].bit.device)

    if dist.get_rank() == 0:
        print("------------- 8-bit Re-SET -------------")

def load_ant_state_dict(model, checkpoint):
    print (checkpoint.keys())
    for name, module in model.named_modules():
        if name + ".quant_grid" in checkpoint.keys():
            module.quant_grid.data = checkpoint[name + ".quant_grid"]
