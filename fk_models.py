# -*- coding: UTF-8 -*-
"""
    @description: using link length coeff to  calculate fk model
"""
import numpy as np
import math
from functools import reduce
np.set_printoptions(precision=4, suppress=True)

import matplotlib.pyplot as plt
import numpy as np
import argparse
from random import shuffle
from numpy import sin,cos,eye,dot,mat
from numpy.linalg import inv
import sympy as sym
from sympy import symbols as sb
from sympy import *
import warnings
import requests
import random

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
        self.A_init = DH_[1][:]
        self.D_init = DH_[2][:]
        self.dx = 0
        self.dy = 0
        self.dz = 0
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
        # print(dis)
        trans_mat[AXIS.index(axis), 3] = dis
        return trans_mat
    
    def change_DH_to_random(self):
        # change link lenth with randoms
        self.A = [A_init_nums * random.uniform(0.9,1.1) + random.uniform(-0.5,0.5) for A_init_nums in self.A_init]
        self.D = [D_init_nums * random.uniform(0.9,1.1) + random.uniform(-0.5,0.5) for D_init_nums in self.D_init]

    def change_DH_to_init(self):
        self.A = [A_init_num for A_init_num in self.A_init]
        self.D = [D_init_num for D_init_num in self.D_init]

    def get_DH(self,joints):
        if len(joints)==len(self.alpha)-1:
            joints = np.append(joints,0)
        assert len(joints)==len(self.alpha)
        ans = []
        DOF = len(joints)
        for i in range(DOF):
            tmp = [self.A[i], self.alpha[i], self.D[i],self.theta[i] + joints[i]]
            ans.append(tmp)
        ans = np.array(ans)
        return ans

    def cal_fk(self, joints,need_d2r = False):
        # thea_1, thea_2, thea_3, thea_4, thea_5, thea_6 = joints
        # DH_pramater: [link, a, d, thea]，注意这里的单位是m
        if joints[0] > np.pi*2 or joints[0] < -np.pi*2:
            need_d2r = True
        if need_d2r:
            d2r = np.pi/180
            joints = [joint_ * d2r for joint_ in joints]
        need_debug = False
        DH = self.get_DH(joints)
        if need_debug:
            print(DH)
        T = []
        for DH_ in DH:
            a_i, alpha_i, d_i, thea_i = DH_[0], DH_[1], DH_[2], DH_[3]
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
            # T_base = np.where(np.abs(T_base) < 1e-5, 0, T_base)
            # # get a small value when cos(np.pi/2)
            if need_debug:
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
        T_base[0][3] = T_base[0][3] + self.dz
        T_base[1][3] = T_base[1][3] - self.dx
        T_base[2][3] = T_base[2][3] - self.dy
        #!!! notice that p_real[x,y,z] = p_fk[-y,-z,x]
        position = [-T_base[1][3],-T_base[2][3],T_base[0][3]]
        return  position

pi = np.pi


def get_Robot():
    link_file = "./data/links.txt"
    with open(link_file,'r') as rf:
        line = rf.readline().strip().split(" ")
        # print("read line :")
        # print(line)
        links_len = [round(float(num),2) for num in line]
        DH_ = [
            [0,     -pi/2,      -pi/2,   -pi/2,     pi/2,   pi/2,        pi/2],      # alpha
            [0,     links_len[0],          0,      0,         0,      0,      links_len[1]],      # a
            [links_len[2],     0,      links_len[3],      0,      links_len[4],      0,           0],      # d
            [0,     np.pi/2,        0,      0,         0,  -pi/2,           0],      # theta
        ]
        Robot_ = FK(DH_)  
        if len(links_len) == 8:
            Robot_.dx = links_len[5]
            Robot_.dy = links_len[6]
            Robot_.dz = links_len[7]
    return Robot_

def load_data(file, is_fk = True, test_data_scale = 0.8):
    with open(file,"r") as rf:
        lines = rf.readlines()
        shuffle(lines)
        p = []
        q = []
        for line in lines:
            datas = line.strip().split(" ")
            p_tmp = np.array([float(x) for x in datas[0:3]])            
            p.append(p_tmp)
            q.append([float(x)/180 * np.pi for x in datas[3:]])
        q = np.array(q)
        p = np.array(p)
    inputs = p
    outputs = q
    p_range = [p.min(0), p.max(0) - p.min(0)]
    q_range = [q.min(0), q.max(0) - q.min(0)]
    inputs = np.array(noramlization(inputs))
    outputs = np.array(noramlization(outputs))
    test_set = [inputs[int(inputs.shape[0]*test_data_scale):-1], 
                outputs[int(inputs.shape[0]*test_data_scale):-1]]
    inputs = inputs[0:int(q.shape[0]*test_data_scale)]
    outputs = outputs[0:int(p.shape[0]*test_data_scale)]
    return inputs, outputs, test_set[0], test_set[1],p_range,q_range

def cal_dis(p_a, p_b):
    ret = 0
    for p_a_, p_b_ in zip(p_a,p_b):
        tmp = np.square(p_a_ - p_b_)
        ret = ret + tmp
    # print(ret)
    ret = math.sqrt(ret)
    return ret

def distance(positions_a, positions_b):
    assert positions_a.shape == positions_b.shape
    # print(positions_a)
    # print(positions_b)
    dis = [cal_dis(p_a, p_b) for p_a, p_b in zip(positions_a, positions_b)]
    dis = np.array(dis)
    mean = np.mean(dis)
    return dis,mean


def noramlization(data,has_equle=False):
    """
        normData = (data - min)/(max - min)
    """
    minVals = data.min(0)
    maxVals = data.max(0)
    print("nmlzt: min:", minVals,"max",maxVals)
    ranges = maxVals - minVals
    normData = (data - minVals)/ranges
    return normData

def noramlization_with_delta_range(data,delta_range):
    """
        normData = (data - min)/(max - min)
    """
    minVals = delta_range[0]
    rangeVals = delta_range[1]
    print("nmlzt_with_delta: min:", minVals,"max",minVals+rangeVals)
    normData = (data - minVals)/rangeVals
    return normData

def read_range_from_file(file):
    with open(file,"r") as rf:
        lines = rf.readlines()
        delta_p_range = lines[0].strip().split(" ")
        delta_p_range = [float(p) for p in delta_p_range]
        delta_p_range = np.array([delta_p_range[0:3],delta_p_range[3:]])
        delta_q_range = lines[1].strip().split(" ")
        delta_q_range = [float(q) for q in delta_q_range]
        delta_q_range = np.array([delta_q_range[0:6],delta_q_range[6:]])
        return delta_p_range, delta_q_range
def generate_delta_data(q_0,p_0, data_nums = 500, test_data_scale = 0.6, delta_range_ = np.pi/20):
    """
    生成围绕p_0，以输入为 delta_p,输出为 delta_q 的数据
    """
    q, p, _, _ ,_,_= generate_data(data_nums, q_0, test_data_scale = 1, is_delta = True,
        delta_range = delta_range_,need_norm=False)
    delta_p = np.array([p_i - p_0 for p_i in p])
    delta_q = np.array([q_i - q_0 for q_i in q])

    # delta_p = [p_.extend(delta_p_) for p_,delta_p_ in zip(p,delta_p)]
    # delta_p = [delta_p_.extend(q_) for q_,delta_p_ in zip(q,delta_p)]
    # delta_p = np.hstack((p,delta_p))
    # delta_p = np.hstack((delta_p,q))
    delta_p = np.array(delta_p)
    print("delta_p.shape",delta_p.shape)

    delta_p_range, delta_q_range = read_range_from_file("./data/delta_min_max.txt")
    # inputs = np.hstack((delta_p,q))
    
    inputs = np.array(delta_p)
    inputs = np.array(noramlization_with_delta_range(inputs,delta_p_range))
    outputs = np.array(noramlization_with_delta_range(delta_q,delta_q_range))
    test_set = [inputs[int(data_nums * test_data_scale):-1], 
            outputs[int(data_nums * test_data_scale):-1]]
    inputs = inputs[0:int(data_nums * test_data_scale)]
    outputs = outputs[0:int(data_nums * test_data_scale)]
    return inputs, outputs,test_set, delta_p_range, delta_q_range

def generate_data(data_nums = 1000, q_e =[0,0,0,0,0,0], is_fk = True, 
                test_data_scale = 0.8, is_delta = False,
                delta_range = np.pi/10, need_d2r = False,need_norm = True,write_to_file = False):
    # (0,512) (0,512) (-298,302),(0,53),(-438,101),(-358,120)
    # (0,pi),(0,pi)
    joint_start = q_e
    if joint_start[0] > np.pi*2 or joint_start[0] < -np.pi*2:
        need_d2r = True
    if need_d2r:
        d2r = np.pi/180
        joint_start = [q_e_ * d2r for q_e_ in q_e]
    q = []
    p = []
    with open("data.txt","w") as wf:
        # pre_joint = np.random.rand(4) * np.pi
        # step = 0.1
        robot_ = get_Robot()
        uniform_ = [(pi/2,pi),(pi/3,pi*3/4),(-pi/2,pi/2),(0,pi/12),(-pi/2,pi/6),(-pi/2,pi/6)]
        for i in range(data_nums):
            if is_delta:
                joint = joint_start + (np.random.rand(6) * 2 - 1) * delta_range
            else:
                joint = []
                for ranges in uniform_:
                    joint_ = random.uniform(ranges[0],ranges[1])
                    joint.append(joint_)
                # np.random.rand(6) * np.pi/2
                # joint[-1] = 0
            q.append(joint)
            # print("joint is ", joint)
            p_tmp = robot_.cal_fk(joint)
            p.append(p_tmp)
            if write_to_file:
                for j in range(3):
                    wf.write(str(round(p[-1][j],2)))
                    wf.write(" ")
                for j in range(6):
                    wf.write(str(round(joint[j],2)))
                    if(j!=5):
                        wf.write(" ")
                wf.write('\n')
        p = np.array(p)
        q = np.array(q)
    if is_fk:
        inputs = q
        outputs = p
    else:
        inputs = p
        outputs = q
    p_range = [p.min(0), p.max(0) - p.min(0)]
    q_range = [q.min(0), q.max(0) - q.min(0)]
    if need_norm:
        inputs = np.array(noramlization(inputs))
        outputs = np.array(noramlization(outputs))
    test_set = [inputs[int(inputs.shape[0]*test_data_scale):-1], 
                outputs[int(inputs.shape[0]*test_data_scale):-1]]
    inputs = inputs[0:int(q.shape[0]*test_data_scale)]
    outputs = outputs[0:int(p.shape[0]*test_data_scale)]
    return inputs, outputs, test_set[0], test_set[1],p_range,q_range


if __name__ == "__main__":
    joints = [1.8184, -0.3251,  0.6989, -0.1285, -0.2991, -0.2028]
    robot = get_Robot()
    pos = robot.cal_fk(joints)
    print("position is:")
    print(pos)