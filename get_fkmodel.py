# -*- coding: UTF-8 -*-
"""
    @description: 
        read datas from "./data/frame_data_0301.txt" or others
        outputs length of links to "links.txt"
        only estimate 6 link_length
"""
import numpy as np
import math
from functools import reduce
np.set_printoptions(precision=4, suppress=True)

import copy
import matplotlib.pyplot as plt
import numpy as np
import argparse
from random import shuffle
import sympy as sym
from sympy import symbols as sb
from sympy import *
from numpy import eye,dot,mat
from numpy.linalg import inv

import random
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima_model import ARIMA
from fk_models import FK
plt.style.use("fivethirtyeight")

warnings.filterwarnings("ignore")

pi = sym.pi
a1 = sb('a1')
a7 = sb('a7')
d1 = sb('d1')
d3 = sb('d3')
d5 = sb('d5')

# [-1.90,-23.53,10.59,11.85,6.35]
# DH_ = [
#         [0,     -pi/2,      -pi/2,   -pi/2,     pi/2,   pi/2,        pi/2],      # alpha
#         [0,     -1.25,          0,      0,         0,      0,      -25.50],      # a
#         [10,        0,      10.55,      0,      8.50,      0,           0],      # d
#         [0,     np.pi/2,        0,      0,         0,  -pi/2,           0],      # theta
#       ]

DH_ = [
        [0,     -pi/2,      -pi/2,   -pi/2,     pi/2,   pi/2,        pi/2],      # alpha
        [0,     a1,          0,      0,         0,      0,      a7],      # a
        [d1,     0,      d3,      0,      d5,      0,           0],      # d
        [0,     np.pi/2,        0,      0,         0,  -pi/2,           0],      # theta
      ]
DH_ = np.array(DH_)

def get_Robot_():
    
    Robot_ = FK(DH_)

    return Robot_


def load_data(file, is_fk = True, test_data_scale = 1):
    with open(file,"r") as rf:
        lines = rf.readlines()
        shuffle(lines)
        p = []
        q = []
        for line in lines:
            datas = line.split(" ")
            p.append([float(x) for x in datas[0:3]])
            q.append([float(x)/180 * np.pi for x in datas[3:-1]])
            q[-1].append(0)
        q = np.array(q)
        p = np.array(p)
    inputs = q
    outputs = p
    test_set = [inputs[int(inputs.shape[0]*test_data_scale):-1], outputs[int(inputs.shape[0]*test_data_scale):-1]]
    inputs = np.array(inputs[0:int(q.shape[0]*test_data_scale)])
    outputs = np.array(outputs[0:int(p.shape[0]*test_data_scale)])
    return inputs, outputs, test_set[0], test_set[1]

def distance(positions_a, positions_b):
    # assert positions_a.shape == positions_b.shape
    sum_ = 0
    for i in range(3):
        sum_+= (positions_a[i] - positions_b[i])**2
    return np.sqrt(sum_)



# input x[m,n],coeff[n],y[m,1]
def loss(x,y,coeff):
    assert x.shape[1]==coeff.shape[0]
    assert x.shape[0]==y.shape[0]
    # print("shape")
    # print(x.shape)
    # print(coeff.shape)
    # print(y.shape)
    total_error = []
    for i in range(x.shape[0]):
        total_error.append(abs(dot(x[i],coeff) - y[i]))
    total_error = np.array(total_error)
    return total_error

def getErrorBar(x,y,coeff):
    errors = loss(x,y,coeff)
    y = np.array(y)
    mx = y.max()
    mi = y.min()
    step_y = (y.max() - y.min())/9
    size = 10
    error = [0 for i in range(10)]
    errors = [[] for i in range(10)]
    for x_,y_ in zip(x,y):
        losses = dot(x_,coeff) - y_
        errors[int((y_ - mi)/step_y)].append(losses)
    labels = [str(round((i+0.5)*step_y+mi,2)) for i in range(10)]
    bplot = plt.boxplot(errors,showmeans=True,patch_artist=True, labels=labels)  # 设置箱型图可填充
    plt.grid(True)
    plt.ylim(-5,5)
    plt.ylabel("distance (cm)")
    plt.xlabel("range of x (cm)")
    plt.show()
    # df = pd.DataFrame(data)
    # # df.plot.box(title="Consumer spending in each country", vert=False)
    # df.plot.box(title="Error in each distance ranges")
    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.errorbar(x, y, yerr=errors, fmt="o", color="black",
    #     ecolor="lightgray", elinewidth=3)
    return

def minMSE(x,y):
    x_ = np.array(x,dtype='float')
    y_ = np.array(y,dtype='float')
    a = dot(dot(inv(dot(x_.T,x_)),x_.T),y_)
    a = np.array(a)
    return a,loss(x,y,a)


# input x[n],coeff[n],y[1]
def gradient(x,y,coeff):
    total_gradient = 0
    # print("x.shape[0]:",x.shape[0])
    # print("coeff.shape[0]:",coeff.shape[0])
    for i in range(x.shape[0]):
        total_gradient += x[i]*coeff[i]
    total_gradient = total_gradient - y[0]
    return total_gradient

# input x[m,n],coeff_init[n],y[m,1]
def gradient_decent(x,y,coeff_init,lr,epoch):
    m = x.shape[0]
    n = coeff_init.shape[0]
    coeff = copy.copy(coeff_init)
    losses = []
    for i in range(epoch):
        coeff_gradient = [0 for i in range(n)]
        for j in range(0,m):
            for k in range(n):
                coeff_gradient[k] += (1.0/m) * x[j][k] * gradient(x[j],y[j],coeff)
        for k in range(n):
            coeff[k] = coeff[k] - lr*coeff_gradient[k]
        losses.append(loss(x,y,coeff).mean())
    return coeff,losses

if __name__ == "__main__":
    # joints = [sym.symbols('q_1'),
    #          sym.symbols('q_2'),
    #          sym.symbols('q_3'),
    #          sym.symbols('q_4'),
    #          sym.symbols('q_5'),
    #          sym.symbols('q_6'),
    #          sym.symbols('q_7')]

    # 32.7929 -0.4943 -3.7116 19.63 -9.96 72.95 83.79 -8.50 -8.51

    
    Robot_ = get_Robot_()
    # print("outputs shape is ",outputs.shape)
    load_raw_data = True
    coeff_ = [[],[],[]]
    outputs = []
    if load_raw_data:
        coeff_out = [[],[],[]]
        files = "./data/frame_data_0301.txt"
        inputs_,outputs_,test_inputs,test_outputs = load_data(files)
        for joints,positions in zip(inputs_,outputs_):
            fk_array = Robot_.cal_fk(joints)
            # a1 a7 d1 d3 d5
            # !!!!!!!!!!!here!!!!!!!!!  posintion[x,y,z](real) = [-y,-z,x](in fk models)
            position = [-fk_array[1][3],-fk_array[2][3],fk_array[0][3]]        
            for i in range(3):
                tmp_coeff = []
                a1_ = position[i].evalf(subs={a1:1,a7:0,d1:0,d3:0,d5:0})
                a7_ = position[i].evalf(subs={a1:0,a7:1,d1:0,d3:0,d5:0})
                d1_ = position[i].evalf(subs={a1:0,a7:0,d1:1,d3:0,d5:0})
                d3_ = position[i].evalf(subs={a1:0,a7:0,d1:0,d3:1,d5:0})
                d5_ = position[i].evalf(subs={a1:0,a7:0,d1:0,d3:0,d5:1})
                # print(a1_,a7_,d1_,d3_,d5_)
                if i ==0 :
                    tmp_coeff = [a1_,a7_,d3_,d5_]
                elif i== 1:
                    tmp_coeff = [a7_,d1_,d3_,d5_]
                else:
                    tmp_coeff = [a1_,a7_,d3_,d5_]
                coeff_out[i].append(tmp_coeff)
        coeff_out = np.array(coeff_out)
        outputs_file =  "./data/coeff_sub_03_01.txt"
        with open(outputs_file,"w") as wf:
            for j in range(coeff_out[1].shape[0]):
                for i in range(3):
                    for k in range(coeff_out[i].shape[1]):
                        wf.write(str(coeff_out[i][j][k]))
                        if k == coeff_out[i].shape[1] - 1:
                            wf.write(',')
                        else:
                            wf.write(' ')
                wf.write(str(outputs_[j][0])+','+str(outputs_[j][1])+','+str(outputs_[j][2]))
                wf.write('\n')

    files = "./data/coeff_sub_03_01.txt"
    with open(files) as rf:
        lines = rf.readlines()
        for line in lines:
            datas = line.split(',')
            coeff_[0].append([float(data) for data in datas[0].split(' ')])
            coeff_[0][-1].append(1)
            coeff_[1].append([float(data) for data in datas[1].split(' ')])
            coeff_[2].append([float(data) for data in datas[2].split(' ')])
            coeff_[2][-1].append(1)
            # outputs.append([float(data) for data in datas[3:6]])
            outputs.append(float(datas[3]))
        coeff_ = [np.array(coeff_i) for coeff_i in coeff_]
        outputs = np.array(outputs)
    test_data_scale = 0.5
    inputs = np.array(coeff_[0])
    outputs = np.array(outputs)
    print("shape:")
    print(outputs.shape)
    print(inputs.shape)
    size = inputs.shape[0]
    test_set = [inputs[int(size*test_data_scale):-1], outputs[int(size*test_data_scale):-1]]
    inputs = np.array(inputs[0:int(size*test_data_scale)])
    outputs = np.array(outputs[0:int(size*test_data_scale)])
    final_coeff_grad = []   # why list?
    final_coeff_mse = []
    coeff_init = [  [-1.25,-25.5,10,5.8,0],
                    [-25.5,10,10.5,8],
                    [-1.25,-25.5,10,5.8,0]]
    coeff_init = [np.array(cof_init) for cof_init in coeff_init]

    # x = mat(coeff_[i])
    # print(outputs[:,0].shape)
    print("shape:")
    print(outputs.shape)
    print(inputs.shape)
    y = mat(outputs).reshape(-1,1)
    x = mat(inputs)
    # y = mat(outputs[:,i]).reshape(-1,1)
    coeff_mse,loss_mse = minMSE(x,y)

    # print("coeff mse is:")
    # print(coeff_mse)
    print("\n***************train****************\n")
    # print("coeff_mse is:")
    # print(coeff_mse)  
    # print("loss mse is:")
    # print(loss_mse.mean())
    # final_coeff_mse.append(coeff_mse)

    # # y = np.array(outputs[:,i]).reshape(-1,1)
    # # x = np.array(coeff_[i]).reshape(y.shape[0],-1)
    y = np.array(outputs).reshape(-1,1)
    x = np.array(inputs)
    # getErrorBar(x,y,coeff_mse)
    coeff_inits = coeff_init[0]
    coeff_grad, loss_grad = gradient_decent(x,y,coeff_inits,1e-1,250)
    coeff_grad, loss_grad = gradient_decent(x,y,coeff_grad,1e-2,500)
    
    print("coeff grad is:")
    print(coeff_grad)
    print("loss grad is:")
    print(loss_grad[-1])
    print("loss init is:")
    print(loss(x,y,coeff_inits).mean())
    print(coeff_inits,coeff_grad)
    getErrorBar(x,y,coeff_grad)
    getErrorBar(x,y,coeff_inits)

    print("\n****************train***************\n")

    print("\n***************test****************\n")
    x = mat(test_set[0])
    y = mat(test_set[1]).reshape(-1,1)
    # print(outputs[:,0].shape)
    print("shape:")
    print(x.shape)
    print(y.shape)
    # not use mse, only gradient_decent
    # coeff_mse,loss_mse = minMSE(x,y)
    # print("loss mse is:")
    # print(loss_mse.mean())
    # final_coeff_mse.append(coeff_mse)

    y = np.array(test_set[1]).reshape(-1,1)
    x = np.array(test_set[0]).reshape(y.shape[0],-1)
    # coeff_grad, loss_grad = gradient_decent(x,y,coeff_inits,1e-2,250)
    print("coeff grad is:")
    print(coeff_grad)
    print("loss grad is:")
    print(loss_grad[-1])
    print("loss init is:")
    print(loss(x,y,coeff_inits).mean())
    print(coeff_inits,"\n",coeff_grad)
    print("\n***************test****************\n")
    final_coeff_grad.append(coeff_grad)
    
    wf_name="./data/links.txt"
    with open(wf_name,'w') as wf:
        for coeff in coeff_grad:
            wf.write(str(coeff))
            wf.write(" ")
    
    final_coeff_grad = np.array(final_coeff_grad)
