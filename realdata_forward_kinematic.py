# -*- coding: UTF-8 -*-
"""
    @description: using measure data to calculate fk model coeff
"""
import numpy as np
import math
from functools import reduce
np.set_printoptions(precision=4, suppress=True)

import matplotlib.pyplot as plt
import numpy as np
import argparse
from random import shuffle
import sympy as sym
# from sympy import *
from numpy import sin,cos,eye,dot,mat
from numpy.linalg import inv
 
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
        print("self.d is ", self.D)
        
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
                            [(0, sin(deg))[rot_z], (cos(deg), 1)[rot_y], (0, -sin(deg))[rot_x], 0],
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
            print("DH_: ",DH_)
            T_ = [
                [cos(thea_i), -sin(thea_i), 0 , a_i],
                [sin(thea_i) * cos(alpha_i), cos(thea_i) * cos(alpha_i), - sin(alpha_i),-sin(alpha_i)*d_i],
                [sin(thea_i) * sin(alpha_i), cos(thea_i) * sin(alpha_i), cos(alpha_i),   cos(alpha_i)*d_i],
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
            print("\n")            
            print("T_"+str(i)+" = *********************************")
            print(np.array(T_))
            print("T_base_"+str(i+1)+" = *********************************")
            # print(np.array(T_base)[:,-1][0:3])
            position = [-T_base[1][3],-T_base[2][3],T_base[0][3]]

            print(position[0])
            print(position[1])
            print(position[2])
            print("\n")
        return  T_base

pi = np.pi


def get_Robot_():
    link_file = "./data/links.txt"
    with open(link_file,'r') as rf:
        line = rf.readline().split(" ")
        links_len = [round(float(num),2) for num in line]
    DH_ = [
        [0,     -pi/2,      -pi/2,   -pi/2,     pi/2,   pi/2,        pi/2],      # alpha
        [0,     a1,          0,      0,         0,      0,      a7],      # a
        [d1,     0,      d3,      0,      d5,      0,           0],      # d
        [0,     np.pi/2,        0,      0,         0,  -pi/2,           0],      # theta
      ]
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
        q = np.array(q)
        p = np.array(p)
    inputs = q
    outputs = p
    test_set = [inputs[int(inputs.shape[0]*test_data_scale):-1], outputs[int(inputs.shape[0]*test_data_scale):-1]]
    inputs = inputs[0:int(q.shape[0]*test_data_scale)]
    outputs = outputs[0:int(p.shape[0]*test_data_scale)]
    return inputs, outputs, test_set[0], test_set[1]


def distance(positions_a, positions_b):
    # assert positions_a.shape == positions_b.shape
    dis = [np.sqrt(np.sum(np.square(p_a - p_b))) for p_a, p_b in zip(positions_a, positions_b)]
    dis = np.array(dis)
    mean = np.mean(dis)
    return dis,mean

DH_ = [
        [0,     -pi/2,      -pi/2,   -pi/2,     pi/2,   pi/2,        pi/2],      # alpha
        [0,     -1.25,          0,      0,         0,      0,      -25.50],      # a
        [10,        0,      10.55,      0,      8.50,      0,           0],      # d
        [0,     np.pi/2,        0,      0,         0,  -pi/2,           0],      # theta
      ]
DH_ = np.array(DH_)

if __name__ == "__main__":
    files = "/home/pku-hr6/yyf_ws/data/test_data.txt"
    inputs,outputs,test_inputs,test_outputs = load_data(files)
    x = mat(inputs)
    y = mat(outputs)
    a = dot(dot(inv(np.dot(x.T,x)),x.T),y)
    print(a)
    # joints = [sym.symbols('q_1'),
    #          sym.symbols('q_2'),
    #          sym.symbols('q_3'),
    #          sym.symbols('q_4'),
    #          sym.symbols('q_5'),
    #          sym.symbols('q_6'),
    #          sym.symbols('q_7')]
    # 22.5969 -3.3345 10.6589 19.04 32.23 96.97 93.46 -8.50 -50.69 
    # 32.7929 -0.4943 -3.7116 19.63 -9.96 72.95 83.79 -8.50 -8.51
    # joints = [19.04,32.23,96.97,93.46,-8.50,-50.69,0]
    # joints = [0,0,0,0,0,0,0]
    # joints = [joint/180.0 * np.pi for joint in joints]
    
    # Robot_ = get_Robot_()
    # fk_array = Robot_.cal_fk(joints)
    # print("**************************************")
    # print(np.array(fk_array))
    # position = [-fk_array[1][3],-fk_array[2][3],fk_array[0][3]]
    # print("position = ")

    # print(position[0])
    # print(position[1])
    # print(position[2])