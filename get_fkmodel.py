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

plt.style.use("fivethirtyeight")

warnings.filterwarnings("ignore")

def floatsin(theta):
    if(isinstance(theta,sym.Basic)):
        return sin(theta)
    if abs(sin(theta)) < 0.01:
        return 0
    if abs(sin(theta) - 1) < 0.01:
        return 1
    if abs(sin(theta) + 1) < 0.01:
        return -1
    return sin(theta);

def floatcos(theta):
    if(isinstance(theta,sym.Basic)):
        return cos(theta)
    if abs(cos(theta)) < 0.01:
        return 0
    if abs(cos(theta) - 1) < 0.01:
        return 1
    if abs(cos(theta) + 1) < 0.01:
        return -1
    return cos(theta);

class FK():
    def __init__(self,DH_):
        # assert DH_.ndim == 2
        # assert DH_.shape[0] == 3
        # assert DH_[0].shape[0] == DH_[1].shape[0]
        # assert DH_[0].shape[0] == DH_[2].shape[0]

        self.alpha = DH_[0][:]
        self.A = DH_[1][:]
        self.D = DH_[2][:]
        self.theta = DH_[3][:]
        # print("self.d is ", self.D)
        
    def rotate(self, axis, deg):
        AXIS = ('X', 'Y', 'Z')
        axis = str(axis).upper()
        if axis not in AXIS:
            print(axis," is unknown axis, should be one of ",AXIS)
            return
        rot_x = axis == 'X'
        rot_y = axis == 'Y'
        rot_z = axis == 'Z'
        rot_mat = np.array([[(cos(deg), 1)[rot_x], (0, -sin(deg))[rot_z], (0, sin(deg))[rot_y], 0],
                            [(0, floatsin(deg))[rot_z], (cos(deg), 1)[rot_y], (0, -sin(deg))[rot_x], 0],
                            [(0, -sin(deg))[rot_y], (0, sin(deg))[rot_x], (cos(deg), 1)[rot_z], 0],
                            [0, 0, 0, 1]])
        # rot_mat = np.where(np.abs(rot_mat) < 1e-5, 0, rot_mat)  # get a small value when cos(np.pi/2)
        return rot_mat

    def trans(self, axis, dis):
        AXIS = ('X', 'Y', 'Z')
        axis = str(axis).upper()
        if axis not in AXIS:
            print(axis," is unknown axis, should be one of ",AXIS)
            return
        trans_mat = eye(4)
        print(dis)
        trans_mat[AXIS.index(axis), 3] = dis
        return trans_mat

    def get_DH(self,joints):
        # assert len(joints)==len(self.alpha)
        if(len(joints)!=len(self.alpha)):
            joints = np.append(joints,0)
        ans = []
        DOF = len(joints)
        for i in range(DOF):
            tmp = [self.A[i], self.alpha[i], self.D[i],self.theta[i] + joints[i]]
            ans.append(tmp)
        ans = np.array(ans)
        return ans

    def cal_fk(self, joints):
        # thea_1, thea_2, thea_3, thea_4, thea_5, thea_6 = joints
        # DH_pramater: [link, a, d, thea]，注意这里的单位是m
        DH = self.get_DH(joints)
        T = []
        for DH_ in DH:
            a_i, alpha_i, d_i, thea_i = DH_[0], DH_[1], DH_[2], DH_[3]
            # print("DH_: ",DH_)
            T_ = [
                [cos(thea_i), -sin(thea_i), 0 , a_i],
                [sin(thea_i) * floatcos(alpha_i), cos(thea_i) * floatcos(alpha_i), - floatsin(alpha_i),-floatsin(alpha_i)*d_i],
                [sin(thea_i) * floatsin(alpha_i), cos(thea_i) * floatsin(alpha_i), floatcos(alpha_i),   floatcos(alpha_i)*d_i],
                [0,0,0,1]
            ]
            T.append(T_)
        # T = [self.rotate('z', thea_i).dot(self.trans('z', d_i)).dot(self.trans('x', l_i)).dot(self.rotate('x', a_i))
        #     for thea_i, d_i, l_i, a_i in DH]
        # robot = reduce(np.dot, T)
        T_base = eye(4)
        for i in range(len(T)):
            T_ = T[i]
            T_base = np.dot(T_base,T_)
            # T_base = np.where(np.abs(T_base) < 1e-5, 0, T_base)  # get a small value when cos(np.pi/2)
            # print("\n")            
            # print("T_"+str(i)+" = *********************************")
            # print(np.array(T_))
            # print("T_base_"+str(i+1)+" = *********************************")
            # # print(np.array(T_base)[:,-1][0:3])
            # position = [-T_base[1][3],-T_base[2][3],T_base[0][3]]

            # print(position[0])
            # print(position[1])
            # print(position[2])
            # print("\n")
        return  T_base

pi = sym.pi
a1 = sb('a1')
a7 = sb('a7')
d1 = sb('d1')
d3 = sb('d3')
d5 = sb('d5')

[-1.90,-23.53,10.59,11.85,6.35]
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
    # DH_ = [ [-np.pi/2,0,0,np.pi/2],         # alpha
    #     [sym.symbols('d_0'),0,0,0],                  # A
    #     [sym.symbols('l_0'), sym.symbols('l_1'), sym.symbols('l_2'), sym.symbols('l_3')]]     # L
    # DH_ = [
    #         [0,-np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2], #alpha
    #         [0,sym.symbols('a_1'),0,0,0,0],                    #a
    #         [sym.symbols('l_0'),0,sym.symbols('l_3'),0,sym.symbols('l_5'),0]                     #d
    #       ]
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

def noramlization(data,has_equle=False):
    minVals = data.min(0)
    maxVals = data.max(0)
    print("min:", minVals,"max",maxVals)
    ranges = maxVals - minVals
    normData = (data - minVals)/ranges


    return normData


def generate_delta_data(q_0,p_0, data_nums = 500, test_data_scale = 0.8):
        """
        生成围绕p_0，以输入为 delta_p,输出为 delta_q 的数据
        """
        q, p, _, _ = generate_data(data_nums, q_0, test_data_scale = 1, is_delta = True, delta_range = np.pi/10)
        delta_p = np.array([p_i - p_0 for p_i in p])
        delta_q = np.array([q_i - q_0 for q_i in q])

        # delta_p = [p_.extend(delta_p_) for p_,delta_p_ in zip(p,delta_p)]
        # delta_p = [delta_p_.extend(q_) for q_,delta_p_ in zip(q,delta_p)]
        # delta_p = np.hstack((p,delta_p))
        # delta_p = np.hstack((delta_p,q))
        delta_p = np.array(delta_p)
        print("delta_p.shape",delta_p.shape)
        delta_p_range = [delta_p.min(0), delta_p.max(0) - delta_p.min(0)]
        delta_q_range = [delta_q.min(0), delta_q.max(0) - delta_q.min(0)]

        # p_0_s = [p_0 for i in range(p.shape[0])]
        # inputs = np.hstack((delta_p,q))
        inputs = np.array(delta_p)
        inputs = np.array(noramlization(inputs))
        outputs = np.array(noramlization(delta_q))
        test_set = [inputs[int(inputs.shape[0] * test_data_scale):-1], 
                outputs[int(inputs.shape[0] * test_data_scale):-1]]
        inputs = inputs[0:int(q.shape[0] * test_data_scale)]
        outputs = outputs[0:int(p.shape[0] * test_data_scale)]
        return inputs, outputs,test_set, delta_p_range, delta_q_range

def generate_data(data_nums = 1000, q_e =[0,0,0,0,0,0], is_fk = True, test_data_scale = 0.5, is_delta = False,delta_range = 0):
    q = []
    p = []
    with open("data.txt","w") as wf:
        # pre_joint = np.random.rand(4) * np.pi
        # step = 0.1
        for i in range(data_nums):
            if is_delta:
                joint = q_e + np.random.rand(6) * delta_range
                # joint[-1] = 0
            else:
                joint = np.random.rand(6) * np.pi/2
                # joint[-1] = 0
            q.append(joint)
            # print("joint is ", joint)
            robot_ = get_Robot_()
            DH_robot_ = robot_.cal_fk(joint)
            # print(DH_robot_)
            p.append(DH_robot_[:,-1][0:3])

            wf.write(str(np.concatenate((joint,DH_robot_[:,-1][0:3]))))
            wf.write('\n')
        p = np.array(p)
        q = np.array(q)
    if is_fk:
        inputs = q
        outputs = p
    else:
        inputs = p
        outputs = q
    # inputs = noramlization(inputs)
    # outputs = noramlization(outputs,True)
    test_set = [inputs[int(inputs.shape[0]*test_data_scale):-1], 
                outputs[int(inputs.shape[0]*test_data_scale):-1]]
    inputs = inputs[0:int(q.shape[0]*test_data_scale)]
    outputs = outputs[0:int(p.shape[0]*test_data_scale)]
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
    load_raw_data = False
    coeff_ = [[],[],[]]
    outputs = []
    if load_raw_data:
        files = "./data/frame_data_0301.txt"
        inputs,outputs,test_inputs,test_outputs = load_data(files)
        for joints,positions in zip(inputs,outputs):
            fk_array = Robot_.cal_fk(joints)
            # a1 a7 d1 d3 d5
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
                coeff_[i].append(tmp_coeff)
        coeff_ = np.array(coeff_)
        outputs_file =  "./data/coeff_sub_03_01.txt"
        with open(outputs_file,"w") as wf:
            for j in range(coeff_[1].shape[0]):
                for i in range(3):
                    for k in range(coeff_[i].shape[1]):
                        wf.write(str(coeff_[i][j][k]))
                        if k == coeff_[i].shape[1] - 1:
                            wf.write(',')
                        else:
                            wf.write(' ')
                wf.write(str(outputs[j][0])+','+str(outputs[j][1])+','+str(outputs[j][2]))
                wf.write('\n')
    else:
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
    coeff_grad, loss_grad = gradient_decent(x,y,coeff_inits,1e-1,500)
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
    coeff_grad, loss_grad = gradient_decent(x,y,coeff_inits,1e-2,250)
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
