# -*- coding: UTF-8 -*-
# don't know which version of the backup is
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
a1 = sb('a1')
a7 = sb('a7')
d1 = sb('d1')
d3 = sb('d3')
d5 = sb('d5')

DH_ = [
        [0,     -pi/2,      -pi/2,   -pi/2,     pi/2,   pi/2,        pi/2],      # alpha
        [0,     -1.25,          0,      0,         0,      0,      -25.50],      # a
        [10,        0,      10.55,      0,      8.50,      0,           0],      # d
        [0,     np.pi/2,        0,      0,         0,  -pi/2,           0],      # theta
      ]
# DH_ = [
#         [0,     -pi/2,      -pi/2,   -pi/2,     pi/2,   pi/2,        pi/2],      # alpha
#         [0,     a1,          0,      0,         0,      0,      a7],      # a
#         [d1,        0,      d3,      0,      d5,      0,           0],      # d
#         [0,     np.pi/2,        0,      0,         0,  -pi/2,           0],      # theta
#       ]
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
    dis = [np.sqrt(np.sum(np.square(p_a - p_b))) for p_a, p_b in zip(positions_a, positions_b)]
    dis = np.array(dis)
    mean = np.mean(dis)
    return dis,mean

if __name__ == "__main__":
    joints = [sym.symbols('q_1'),
             sym.symbols('q_2'),
             sym.symbols('q_3'),
             sym.symbols('q_4'),
             sym.symbols('q_5'),
             sym.symbols('q_6'),
             sym.symbols('q_7')]

    # 32.7929 -0.4943 -3.7116 19.63 -9.96 72.95 83.79 -8.50 -8.51

    # files = "/home/pku-hr6/yyf_ws/data/test_data.txt"
    # inputs,outputs,test_inputs,test_outputs = load_data(files)
    # Robot_ = get_Robot_()
    # # print("outputs shape is ",outputs.shape)
    # coeff_ = [[],[],[]]
    # for joints,positions in zip(inputs,outputs):
    #     fk_array = Robot_.cal_fk(joints)
    #     # a1 a7 d1 d3 d5
    #     position = [-fk_array[1][3],-fk_array[2][3],fk_array[0][3]]        
    #     for i in range(3):
    #         tmp_coeff = []
    #         a1_ = position[i].evalf(subs={a1:1,a7:0,d1:0,d3:0,d5:0})
    #         a7_ = position[i].evalf(subs={a1:0,a7:1,d1:0,d3:0,d5:0})
    #         d1_ = position[i].evalf(subs={a1:0,a7:0,d1:1,d3:0,d5:0})
    #         d3_ = position[i].evalf(subs={a1:0,a7:0,d1:0,d3:1,d5:0})
    #         d5_ = position[i].evalf(subs={a1:0,a7:0,d1:0,d3:0,d5:1})
    #         # print(a1_,a7_,d1_,d3_,d5_)
    #         if i ==0 :
    #             tmp_coeff = [a1_,a7_,d3_,d5_]
    #         elif i== 1:
    #             tmp_coeff = [a7_,d1_,d3_,d5_]
    #         else:
    #             tmp_coeff = [a1_,a7_,d3_,d5_]

    #         coeff_[i].append(tmp_coeff)
    # coeff_ = np.array(coeff_)
    # final_coeff = []
    # for i in range(3):
    #     x = mat(coeff_[i])
    #     # print(outputs[:,0].shape)
    #     y = mat(outputs[:,i]).reshape(-1,1)
    #     # print(x.shape)
    #     # print(y.shape)
    #     a = dot(dot(inv(dot(x.T,x)),x.T),y)
    #     print(a)
    #     final_coeff.append(a)
    # i = 25
    # print("coeff[0][i] is ",coeff_[0][i])
    # print(coeff_[0][i].dot(final_coeff[0]))
    # print(outputs[i][0])

    # joints = [19.63,-9.96,72.95,83.79,-8.50,-8.51,0]
    # joints = [0,0,0,0,0,0,0]
    joints = [joint/180.0 * np.pi for joint in joints]
    Robot_ = get_Robot_()
    position = Robot_.cal_fk(joints)
    print("**************************************")
    # print(np.array(fk_array))
    print("position = ")

    print(position[0])
    print('\n')
    print(position[1])
    print('\n')
    print(position[2])

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