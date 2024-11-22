import argparse
import copy
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import flow_models.flows as fnn
#import utils


def get_model(num_inputs=20, num_hidden=100, num_blocks=5, flow='realnvp'):
    '''
    inputs
    -----
    num_inputs: input dim of flow

    returns
    ------
    model
    '''

    if sys.version_info < (3, 6):
        print('Sorry, this code might need Python 3.6 or higher')

    act = 'relu' #'tanh' if args.dataset is 'GAS' else 'relu'
    modules = []
    num_cond_inputs = None

    assert flow in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
    if flow == 'glow':
        mask = torch.arange(0, num_inputs) % 2
        # mask = mask.to(device).float()

        print("Warning: Results for GLOW are not as good as for MAF yet.")
        for _ in range(num_blocks):
            modules += [
                fnn.BatchNormFlow(num_inputs),
                fnn.LUInvertibleMM(num_inputs),
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu')
            ]
            mask = 1 - mask
    elif flow == 'realnvp':
        mask = torch.arange(0, num_inputs) % 2
        # mask = mask.to(device).float()

        for _ in range(num_blocks):
            modules += [
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs)
            ]
            mask = 1 - mask
    elif flow == 'maf':
        for _ in range(num_blocks):
            modules += [
                fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]
    elif flow == 'maf-split':
        for _ in range(num_blocks):
            modules += [
                fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                            s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
            ]
    elif flow == 'maf-split-glow':
        for _ in range(num_blocks):
            modules += [
                fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                            s_act='tanh', t_act='relu'),
                fnn.BatchNormFlow(num_inputs),
                fnn.InvertibleMM(num_inputs)
            ]

    model = fnn.FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
    return model

def get_inv_model(num_inputs=20, num_hidden=100, num_blocks=5, flow='realnvp'):
    '''
    inputs
    -----
    num_inputs: input dim of flow

    returns
    ------
    model
    '''

    if sys.version_info < (3, 6):
        print('Sorry, this code might need Python 3.6 or higher')

    act = 'relu' #'tanh' if args.dataset is 'GAS' else 'relu'
    modules = []
    num_cond_inputs = None
    assert flow in ['realnvp']
    if flow == 'realnvp':
        mask = torch.arange(0, num_inputs) % 2

        # Put BN first and then coupling layer
        for _ in range(num_blocks):
            modules += [
                fnn.BatchNormFlow(num_inputs),
                fnn.CouplingLayer(
                    num_inputs, num_hidden, mask, num_cond_inputs,
                    s_act='tanh', t_act='relu')              
            ]
            mask = 1 - mask

    model = fnn.FlowSequential(*modules)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
    return model

# if __name__=="__main__":
#     model = get_model()
#     print(model)