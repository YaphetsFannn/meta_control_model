# -*- coding: UTF-8 -*-
"""
    @description: 
        read datas from "./data/frame_data_0301.txt" or others
        outputs length of links to "links.txt"
        only estimate 6 link_length
        not use mse, only grad
"""
import numpy as np
import math
from functools import reduce
np.set_printoptions(precision=4, suppress=True)
import  argparse
import os
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import argparse
from random import shuffle
import sympy as sym
from sympy import symbols as sb
from numpy import eye,dot,mat
from numpy.linalg import inv

import random
import warnings
import requests
import numpy as np
# import pandas as pd
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
            datas = line.strip().split(" ")
            p.append([float(x) for x in datas[0:3]])
            q.append([float(x) for x in datas[3:]])
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
        errors[int((y_ - mi)/step_y)].append(abs(losses))
    labels = [str(int((i+0.5)*step_y+mi)) for i in range(10)]
    bplot = plt.boxplot(errors,showmeans=True,patch_artist=True, labels=labels)  # 设置箱型图可填充
    plt.grid(True)
    plt.ylim(0,5)
    plt.ylabel("distance (cm)")
    plt.xlabel("range (cm)")
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
def gradient_decent(x,y,coeff_init,lr_start,epoch,decay=0):
    m = x.shape[0]
    n = coeff_init.shape[0]
    coeff = copy.copy(coeff_init)
    losses = []
    lr = lr_start
    for i in range(epoch):
        lr = lr_start * 1.0 /(1.0 + decay*i)
        coeff_gradient = [0 for _ in range(n)]
        for j in range(0,m):
            for k in range(n):
                coeff_gradient[k] += (1.0/m) * x[j][k] * gradient(x[j],y[j],coeff)
        for k in range(n):
            # if k == 0:
            #     continue
            coeff[k] = coeff[k] - lr*coeff_gradient[k]
        losses.append(loss(x,y,coeff).mean())
    return coeff,losses

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file', type=str, help='file name(or path)', default="real_data")
    argparser.add_argument('--n', type=int, help='data_num', default=100)
    argparser.add_argument('--r', type=bool, help='real raw data from file', default=True)
    argparser.add_argument('--e', type=int, help='epoch', default=50)
    argparser.add_argument('--lr', type=float, help='learning rate', default=1e-1/2)
    args = argparser.parse_args()
    file_names = args.file

    Robot_ = get_Robot_()
    # print("outputs shape is ",outputs.shape)
    load_raw_data = args.r
    coeff_A = []
    outputs = []
    if load_raw_data:
        coeff_out = []
        files = os.path.join("./data",file_names+'.txt')
        inputs_,outputs_,test_inputs,test_outputs = load_data(files)
        outputs_xyz = []
        counts_data = 0        
        for joints,positions in zip(inputs_,outputs_):
            if counts_data > args.n:
                    break
            counts_data = counts_data + 1
            # a1 a7 d1 d3 d5
            position = Robot_.cal_fk(joints,True)
            
            for i in range(3):
                tmp_coeff = []
                a1_ = position[i].evalf(subs={a1:1,a7:0,d1:0,d3:0,d5:0})
                a7_ = position[i].evalf(subs={a1:0,a7:1,d1:0,d3:0,d5:0})
                d1_ = position[i].evalf(subs={a1:0,a7:0,d1:1,d3:0,d5:0})
                d3_ = position[i].evalf(subs={a1:0,a7:0,d1:0,d3:1,d5:0})
                d5_ = position[i].evalf(subs={a1:0,a7:0,d1:0,d3:0,d5:1})
                # print(a1_,a7_,d1_,d3_,d5_)
                if i ==0 :
                    tmp_coeff = [a1_,a7_,0,d3_,d5_,1,0,0]
                elif i== 1:
                    tmp_coeff = [0,a7_,d1_,d3_,d5_,0,1,0]
                else:
                    tmp_coeff = [a1_,a7_,0,d3_,d5_,0,0,1]
                coeff_out.append(tmp_coeff)
                outputs_xyz.append(positions[i])

        coeff_out = np.array(coeff_out)
        outputs_file =  os.path.join("./data",file_names+'_coeff.txt')
        with open(outputs_file,"w") as wf:
            for coeffs,Y in zip(coeff_out,outputs_xyz):
                for coeff in coeffs:
                    wf.write(str(coeff))
                    wf.write(",")
                wf.write(str(Y)+'\n')


    files = os.path.join("./data",file_names+'_coeff.txt')
    with open(files) as rf:
        lines = rf.readlines()
        shuffle(lines)
        for line in lines:
            datas = line.strip().split(',')
            datas = [float(num) for num in datas ]
            coeff_A.append(datas[0:-1])
            outputs.append(datas[-1])
        coeff_A = np.array(coeff_A)
        outputs = np.array(outputs)
    
    test_data_scale = 0.5
    inputs = np.array(coeff_A)
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
    # coeff_init = [  [-1.25,-25.5,10.5,8,0],
    #                 [-25.5,10,10.5,8],
    #                 [-1.25,-25.5,10.5,8,0]]
    coeff_init = [-1.25,-21.5,10,10.5,11,0,0,0]
    coeff_init = np.array(coeff_init)

    # x = mat(coeff_[i])
    # print(outputs[:,0].shape)
    print("shape:")
    print(outputs.shape)
    print(inputs.shape)
    # print(inputs[0])
    # print(outputs[0])
    # y = mat(outputs).reshape(-1,1)
    # x = mat(inputs)
    # y = mat(outputs[:,i]).reshape(-1,1)
    # coeff_mse,loss_mse = minMSE(x,y)

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
    coeff_grad, loss_grad = gradient_decent(x,y,coeff_init,args.lr,args.e,0.5/args.e)
    x_table = range(0,len(loss_grad))
    plt.plot(x_table,loss_grad)
    plt.ylabel("mean distance (cm)")
    plt.xlabel("epoch")
    plt.show()
    # coeff_grad, loss_grad = gradient_decent(x,y,coeff_grad,1e-2,200)
    # x_table = range(0,len(loss_grad))
    # plt.plot(x_table,loss_grad)
    # plt.show()
    print("coeff grad is:")
    print(coeff_grad)
    print("loss grad is:")
    print(loss_grad[-1])
    print("loss init is:")
    print(loss(x,y,coeff_init).mean())
    print(coeff_init,coeff_grad)
    getErrorBar(x,y,coeff_grad)
    getErrorBar(x,y,coeff_init)

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
    # coeff_grad, loss_grad = gradient_decent(x,y,coeff_init,1e-2,250)
    print("coeff grad is:")
    print(coeff_grad)
    print("loss grad is:")
    print(loss(x,y,coeff_grad).mean())
    print("loss init is:")
    print(loss(x,y,coeff_init).mean())
    print(coeff_init,"\n",coeff_grad)
    print("\n***************test****************\n")
    final_coeff_grad.append(coeff_grad)
    
    wf_name="./data/links.txt"
    with open(wf_name,'w') as wf:
        for coeff in coeff_grad:
            wf.write(str(coeff))
            wf.write(" ")
    
    final_coeff_grad = np.array(final_coeff_grad)
