# -*- coding: UTF-8 -*-
"""
    @description: classes of models
"""
import numpy as np
import math
from functools import reduce
np.set_printoptions(precision=4, suppress=True)
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import datasets, transforms
from torch.nn import init
import argparse
from random import shuffle


class sinNN(nn.Module):
    def __init__(self, input_size):
        super(sinNN, self).__init__()   # nn.Module子类的函数必须在构造函数中执行父类的构造函数
		# self.weights = nn.Parameter(torch.Tensor(emb_size, emb_size), requires_grad=requires_grad)
		# self.weights = nn.Parameter(torch.Tensor(input_size, input_size*2), requires_grad=requires_grad)
        # if bias:
        #     self.bias = nn.Parameter(torch.Tensor(output_features))
        # else:
        #     # You should always register all possible parameters, but the
        #     # optional ones can be None if you want.
        #     self.register_parameter('bias', None)
		# 初始化参数
    def forward(self, inputs):
        outputs_1 = torch.sin(inputs)
        outputs_2 = torch.cos(inputs)
        inputs = torch.cat((outputs_1, outputs_2), dim=1)
        # print("after cos\sin , shape is ",inputs.shape)
        return inputs

class fk_model_back(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(fk_model_back, self).__init__()
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        lys = []
        
        self.hidden_0 = torch.nn.Linear(input_size, hidden_size)
        for _ in range(hidden_layer):
            lys.append(nn.Linear(hidden_size, hidden_size))
            lys.append(nn.ReLU())
        self.ly = nn.Sequential(*lys)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = self.hidden_0(x)
        x = self.out(self.ly(x))
        return x

class fk_model(nn.Module):
    def __init__(self, config):
        super(fk_model, self).__init__()

        self.config = config
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        lys = []
        for i, (name, param) in enumerate(self.config):
            if name is 'linear':
                # [ch_out, ch_in]
                param = list(reversed(param))
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars = None, bn_training = True):
        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

class ik_model(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(ik_model, self).__init__()
        lys = []
        lys.append(nn.Linear(input_size, hidden_size))
        lys.append(nn.ReLU())
        for _ in range(hidden_layer):
            lys.append(nn.Linear(hidden_size, hidden_size))
            lys.append(nn.ReLU())
        self.ly = nn.Sequential(*lys)
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size))
    
    def forward(self, x):
        x = self.ly(x)
        x = self.out(x)
        return x