# -*- coding: UTF-8 -*-
'''
    description:
        using mse to esitimate all link length from data [p,q]
        all link means that not only 6 link, but all the link length of pkuhr6
'''
import numpy as np
import math
from functools import reduce
np.set_printoptions(precision=4, suppress=True)

import matplotlib.pyplot as plt
import numpy as np
import argparse
from random import shuffle
import sympy as sym
from sympy import symbols as sb
from sympy import *
from numpy import eye,dot,mat
from numpy.linalg import inv

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
        assert len(joints)==len(self.alpha)
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
a = [sb('a' + str(i)) for i in range(7)]
d = [sb('d' + str(i)) for i in range(7)]
syms = a+d
# DH_ = [
#         [0,     -pi/2,      -pi/2,   -pi/2,     pi/2,   pi/2,        pi/2],      # alpha
#         [0,     -1.25,          0,      0,         0,      0,      -25.50],      # a
#         [10,        0,      10.55,      0,      8.50,      0,           0],      # d
#         [0,     np.pi/2,        0,      0,         0,  -pi/2,           0],      # theta
#       ]
DH_ = [
        [0,            -pi/2,      -pi/2,   -pi/2,   pi/2,   pi/2,        pi/2],      # alpha
        [syms[0],      syms[1],   syms[2],   syms[3],     syms[4],   syms[5],     syms[6]],      # a
        [syms[7],      syms[8],   syms[9],  syms[10],     syms[11],  syms[12],    syms[13]],      # d
        [0,             pi/2,        0,      0,         0,  -pi/2,           0],      # theta                     #d
          ]
DH_ = np.array(DH_)
def get_Robot_():
    Robot_ = FK(DH_)
    return Robot_


def load_data(file, is_fk = True, test_data_scale = 0.8):
    with open(file,"r") as rf:
        lines = rf.readlines()
        shuffle(lines)
        p = []
        q = []
        for line in lines:
            datas = line.split(" ")
            p.append([float(x) for x in datas[0:3]])
            q.append([float(x)/180 * np.pi for x in datas[3:-1]])
            q[-1].append(0) # change to 7 dof
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
    dis = [np.sqrt(np.sum(np.square(p_a - p_b))) for p_a, p_b in zip(positions_a, positions_b)]
    dis = np.array(dis)
    mean = np.mean(dis)
    return dis,mean

def equals(a,b):
    return abs(a-b)<0.001

# input x[m,n],coeff[n],y[m,1]
def loss(x,y,coeff):
    assert x.shape[1]==coeff.shape[0]
    assert x.shape[0]==y.shape[0]
    # print("shape")
    # print(x.shape)
    # print(coeff.shape)
    # print(y.shape)
    total_error = 0
    for i in range(x.shape[0]):
        total_error = total_error + (dot(x[i],coeff) - y[i])**2
    return total_error/float(x.shape[0])/2

# input x[n],coeff[n],y[1]
def gradient(x,y,coeff):
    total_gradient = 0
    for i in range(x.shape[0]):
        total_gradient += x[i]*coeff[i]
    total_gradient = y[0] - total_gradient
    return total_gradient

# input x[m,n],coeff_init[n],y[m,1]
def gradient_decent(x,y,coeff_init,lr,epoch):
    m = x.shape[0]
    n = coeff_init.shape[0]
    coeff = coeff_init
    losses = []
    for i in range(epoch):
        coeff_gradient = [0 for i in range(n)]
        for j in range(0,m):
            for k in range(n):
                coeff_gradient[k] -= (1.0/m) * x[j][k] * gradient(x[j],y[j],coeff)
        for k in range(n):
            coeff[k] = coeff[k] - lr*coeff_gradient[k]
        losses.append(loss(x,y,coeff))
    return coeff,losses


def minMSE(x,y):
    """
    param x: variables of a multivariate linear equation
    param y: dependent variables of a multivariate linear equation
    return: coeff of variables x and the loss of it
    """
    a = dot(dot(inv(dot(x.T,x)),x.T),y)
    a = np.array(a)
    return a,loss(x,y,a)

if __name__ == "__main__":
    # joints = [sym.symbols('q_1'),
    #          sym.symbols('q_2'),
    #          sym.symbols('q_3'),
    #          sym.symbols('q_4'),
    #          sym.symbols('q_5'),
    #          sym.symbols('q_6'),
    #          sym.symbols('q_7')]

    load_raw_data = False   # load raw data of just load coeff
    coeff_ = [[],[],[]]
    outputs = []
    if load_raw_data:
        files = "./data/frame_data_0301.txt"
        inputs,outputs,test_inputs,test_outputs = load_data(files,test_data_scale=1)
        Robot_ = get_Robot_()
        # print("outputs shape is ",outputs.shape)
        not_in = [[0,7],[0,1,8],[7]]    
        
        for joints,positions in zip(inputs,outputs):
            fk_array = Robot_.cal_fk(joints)
            # a1 a7 d1 d3 d5
            position = [-fk_array[1][3],-fk_array[2][3],fk_array[0][3]]
            for i in range(3):
                tmp_coeff = []
                for j in range(len(syms)):
                    if j in not_in[i]:
                        continue
                    sub = {}
                    for k in range(len(syms)):
                        if k == j:
                            sub[syms[k]]=1.0
                        else:
                            sub[syms[k]]=0.0
                    # print(sub)
                    tmp = position[i].evalf(subs=sub)
                    tmp_coeff.append(tmp)
                    # print("tmp is ",tmp)
                # print(a1_,a7_,d1_,d3_,d5_)
                # print("position ",i, "len is ",len(tmp_coeff))
                # tmp_coeff = np.array(tmp_coeff)
                coeff_[i].append(tmp_coeff)
        coeff_ = [np.array(coeff_i) for coeff_i in coeff_]
        outputs_file =  "./data/coeff_0312.txt"
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
        files = "./data/coeff_0312.txt"
        with open(files) as rf:
            lines = rf.readlines()
            for line in lines:
                datas = line.split(',')
                coeff_[0].append([float(data) for data in datas[0].split(' ')])
                coeff_[1].append([float(data) for data in datas[1].split(' ')])
                coeff_[2].append([float(data) for data in datas[2].split(' ')])
                outputs.append([float(data) for data in datas[3:6]])
            coeff_ = [np.array(coeff_i) for coeff_i in coeff_]
            outputs = np.array(outputs)
    # print(coeff_.shape)
    # print(coeff_[0].shape)
    # print(coeff_[0][0].shape)
    # print(coeff_[0][0])
    final_coeff_mse = []
    final_coeff_grad = []
    coeff_init = [  
                    [-1.25, 0, 0, 0, 0, -25.50, 0,10.55,  0,  8.50,0,0],
                    [0, 0, 0, 0, -25.50, 10, 10.55,  0,  8.50,0,0],
                    [0, -1.25, 0, 0, 0, 0, -25.50, 0,10.55,  0,  8.50,0,0]
                 ]
    for i in range(3):
        y = mat(outputs[:,i]).reshape(-1,1)        
        x = mat(coeff_[i])
        # print(outputs[:,0].shape)
        # print(x.shape)
        # print(y.shape)
        # print(x[0])
        coeff_mse,loss_mse = minMSE(x,y)
        # print("coeff mse is:")
        # print(coeff_mse)
        print("\n*******************************\n")        
        print("loss mse is:")
        print(loss_mse)
        final_coeff_mse.append(coeff_mse)
        
        y = np.array(outputs[:,i]).reshape(-1,1)
        x = np.array(coeff_[i]).reshape(y.shape[0],-1)
        
        coeff_grad, loss_grad = gradient_decent(x,y,np.array(coeff_init[i]),1e-2,300)
        print("coeff grad is:")
        print(coeff_grad)
        print("loss grad is:")
        print(loss_grad[-1])
        print("\n*******************************\n")
        final_coeff_grad.append(coeff_grad)
    
    i = 128
    print("coeff_mse[0][i] is ",coeff_[0][i])
    print(coeff_[0][i].dot(final_coeff_mse[0]))
    print(coeff_[0][i].dot(final_coeff_grad[0]))
    print(coeff_[0][i].dot(coeff_init[0]))
    print(outputs[i][0])



    # joints = [19.63,-9.96,72.95,83.79,-8.50,-8.51,0]
    # # joints = [0,0,0,0,0,0,0]
    # joints = [joint/180.0 * np.pi for joint in joints]
    # Robot_ = get_Robot_()
    # fk_array = Robot_.cal_fk(joints)
    # print("**************************************")
    # # print(np.array(fk_array))
    # position = [-fk_array[1][3],-fk_array[2][3],fk_array[0][3]]
    # print("position = ")

    # print(position[0])
    # print('\n')
    # print(position[1])
    # print('\n')
    # print(position[2])



# - a1*sin(q_1) 
# - a7*((((-sin(q_1)*sin(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*cos(q_4) + sin(q_1)*sin(q_4)*cos(q_2))*cos(q_5) + (sin(q_1)*sin(q_2)*sin(q_3) - cos(q_1)*cos(q_3))*sin(q_5))*sin(q_6) - ((-sin(q_1)*sin(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*sin(q_4) - sin(q_1)*cos(q_2)*cos(q_4))*cos(q_6)) 
# + d3*sin(q_1)*cos(q_2) 
# + d5*(-(-sin(q_1)*sin(q_2)*cos(q_3) - sin(q_3)*cos(q_1))*sin(q_4) + sin(q_1)*cos(q_2)*cos(q_4))


# - a7*(((-sin(q_2)*sin(q_4) - cos(q_2)*cos(q_3)*cos(q_4))*cos(q_5) + sin(q_3)*sin(q_5)*cos(q_2))*sin(q_6) - (sin(q_2)*cos(q_4) - sin(q_4)*cos(q_2)*cos(q_3))*cos(q_6)) 
# - d1 
# - d3*sin(q_2) 
# + d5*(-sin(q_2)*cos(q_4) + sin(q_4)*cos(q_2)*cos(q_3))


# a1*cos(q_1) 
# + a7*((((sin(q_1)*sin(q_3) - sin(q_2)*cos(q_1)*cos(q_3))*cos(q_4) + sin(q_4)*cos(q_1)*cos(q_2))*cos(q_5) + (sin(q_1)*cos(q_3) + sin(q_2)*sin(q_3)*cos(q_1))*sin(q_5))*sin(q_6) - ((sin(q_1)*sin(q_3) - sin(q_2)*cos(q_1)*cos(q_3))*sin(q_4) - cos(q_1)*cos(q_2)*cos(q_4))*cos(q_6)) 
# - d3*cos(q_1)*cos(q_2) - 
# d5*(-(sin(q_1)*sin(q_3) - sin(q_2)*cos(q_1)*cos(q_3))*sin(q_4) + cos(q_1)*cos(q_2)*cos(q_4))


# [0, -1.25, 0, 0, 0, 0, -25.50, 
# 10, 0,10.55,  0,  8.50,0,0]
# [-1.25, 0, 0, 0, 0, -25.50, 0,10.55,  0,  8.50,0,0]
# - a1*sin(q_1) 
# + a2*sin(q_1)*sin(q_2) 
# - a3*(-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1)) 
# - a4*((-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1))*cos(q_4) + 1.0*sin(q_1)*sin(q_4)*cos(q_2)) 
# - a5*(((-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1))*cos(q_4) + 1.0*sin(q_1)*sin(q_4)*cos(q_2))*cos(q_5) + (1.0*sin(q_1)*sin(q_2)*sin(q_3) - 1.0*cos(q_1)*cos(q_3))*sin(q_5)) 
# - a6*((((-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1))*cos(q_4) + 1.0*sin(q_1)*sin(q_4)*cos(q_2))*cos(q_5) + (1.0*sin(q_1)*sin(q_2)*sin(q_3) - 1.0*cos(q_1)*cos(q_3))*sin(q_5))*sin(q_6) - ((-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1))*sin(q_4) - 1.0*sin(q_1)*cos(q_2)*cos(q_4))*cos(q_6)) 
# - *d1*cos(q_1) 
# + *d2*sin(q_1)*cos(q_2) 
# - d3*(1.0*sin(q_1)*sin(q_2)*sin(q_3) - 1.0*cos(q_1)*cos(q_3)) 
# + d4*(-(-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1))*sin(q_4) + 1.0*sin(q_1)*cos(q_2)*cos(q_4)) 
# + d5*(-((-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1))*cos(q_4) + 1.0*sin(q_1)*sin(q_4)*cos(q_2))*sin(q_5) + (1.0*sin(q_1)*sin(q_2)*sin(q_3) - 1.0*cos(q_1)*cos(q_3))*cos(q_5)) 
# + d6*((((-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1))*cos(q_4) + 1.0*sin(q_1)*sin(q_4)*cos(q_2))*cos(q_5) + (1.0*sin(q_1)*sin(q_2)*sin(q_3) - 1.0*cos(q_1)*cos(q_3))*sin(q_5))*cos(q_6) + ((-1.0*sin(q_1)*sin(q_2)*cos(q_3) - 1.0*sin(q_3)*cos(q_1))*sin(q_4) - 1.0*sin(q_1)*cos(q_2)*cos(q_4))*sin(q_6))

# [0, 0, 0, 0, -25.50, 10, 10.55,  0,  8.50,0,0]
# a2*cos(q_2) 
# + a3*cos(q_2)*cos(q_3) 
# - a4*(-1.0*sin(q_2)*sin(q_4) - 1.0*cos(q_2)*cos(q_3)*cos(q_4)) 
# - a5*((-1.0*sin(q_2)*sin(q_4) - 1.0*cos(q_2)*cos(q_3)*cos(q_4))*cos(q_5) + 1.0*sin(q_3)*sin(q_5)*cos(q_2)) 
# - a6*(((-1.0*sin(q_2)*sin(q_4) - 1.0*cos(q_2)*cos(q_3)*cos(q_4))*cos(q_5) + 1.0*sin(q_3)*sin(q_5)*cos(q_2))*sin(q_6) - (1.0*sin(q_2)*cos(q_4) - 1.0*sin(q_4)*cos(q_2)*cos(q_3))*cos(q_6)) 
# - d0 
# - d2*sin(q_2) 
# - d3*sin(q_3)*cos(q_2) 
# + d4*(-1.0*sin(q_2)*cos(q_4) + sin(q_4)*cos(q_2)*cos(q_3)) 
# + d5*(-(-1.0*sin(q_2)*sin(q_4) - 1.0*cos(q_2)*cos(q_3)*cos(q_4))*sin(q_5) + 1.0*sin(q_3)*cos(q_2)*cos(q_5)) 
# + d6*(((-1.0*sin(q_2)*sin(q_4) - 1.0*cos(q_2)*cos(q_3)*cos(q_4))*cos(q_5) + 1.0*sin(q_3)*sin(q_5)*cos(q_2))*cos(q_6) + (1.0*sin(q_2)*cos(q_4) - 1.0*sin(q_4)*cos(q_2)*cos(q_3))*sin(q_6))


# [0, -1.25, 0, 0, 0, 0, -25.50, 0,10.55,  0,  8.50,0,0]
# a0 
# + a1*cos(q_1) 
# - a2*sin(q_2)*cos(q_1) 
# + a3*(1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3)) 
# + a4*((1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3))*cos(q_4) + 1.0*sin(q_4)*cos(q_1)*cos(q_2)) 
# + a5*(((1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3))*cos(q_4) + 1.0*sin(q_4)*cos(q_1)*cos(q_2))*cos(q_5) + (1.0*sin(q_1)*cos(q_3) + 1.0*sin(q_2)*sin(q_3)*cos(q_1))*sin(q_5)) 
# + a6*((((1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3))*cos(q_4) + 1.0*sin(q_4)*cos(q_1)*cos(q_2))*cos(q_5) + (1.0*sin(q_1)*cos(q_3) + 1.0*sin(q_2)*sin(q_3)*cos(q_1))*sin(q_5))*sin(q_6) - ((1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3))*sin(q_4) - 1.0*cos(q_1)*cos(q_2)*cos(q_4))*cos(q_6)) 
# - d1*sin(q_1)
# - d2*cos(q_1)*cos(q_2) 
# + d3*(1.0*sin(q_1)*cos(q_3) + 1.0*sin(q_2)*sin(q_3)*cos(q_1)) 
# - d4*(-(1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3))*sin(q_4) + 1.0*cos(q_1)*cos(q_2)*cos(q_4)) 
# - d5*(-((1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3))*cos(q_4) + 1.0*sin(q_4)*cos(q_1)*cos(q_2))*sin(q_5) + (1.0*sin(q_1)*cos(q_3) + 1.0*sin(q_2)*sin(q_3)*cos(q_1))*cos(q_5)) 
# - d6*((((1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3))*cos(q_4) + 1.0*sin(q_4)*cos(q_1)*cos(q_2))*cos(q_5) + (1.0*sin(q_1)*cos(q_3) + 1.0*sin(q_2)*sin(q_3)*cos(q_1))*sin(q_5))*cos(q_6) + ((1.0*sin(q_1)*sin(q_3) - 1.0*sin(q_2)*cos(q_1)*cos(q_3))*sin(q_4) - 1.0*cos(q_1)*cos(q_2)*cos(q_4))*sin(q_6))