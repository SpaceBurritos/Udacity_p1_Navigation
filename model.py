#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:17:52 2021

@author: andy
"""

import torch
from torch import nn
#from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt

device = "cuda"if torch.cuda.is_available() else "cpu"

class Model(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        super(Model, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(state_size, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512, action_size),
                nn.Softmax(dim=1)
                )
    
    def forward(self, state):
        f = self.linear_relu_stack(state)
        return f
    
