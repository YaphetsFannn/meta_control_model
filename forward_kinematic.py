# -*- coding: UTF-8 -*-
"""
    @description: forward kinematic models of a 4 DOF robot arm, change DH_ and joint size to change DOF
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
from sympy import symbols

class FK():
    def __init__(self,DH_):
        assert DH_.ndim == 2
        assert DH_.shape[0] == 3
        assert DH_[0].shape[0] == DH_[1].shape[0]
        assert DH_[0].shape[0] == DH_[2].shape[0]

        self.alpha = DH_[0][:]
        self.D = DH_[1][:]
        self.L = DH_[2][:]

    def rotate(self, axis, deg):
        AXIS = ('X', 'Y', 'Z')
        axis = str(axis).upper()
        if axis not in AXIS:
            print(axis,"is unknown axis, should be one of ",AXIS)
            return
        rot_x = axis == 'X'
        rot_y = axis == 'Y'
        rot_z = axis == 'Z'
        rot_mat = np.array([[(np.cos(deg), 1)[rot_x], (0, -np.sin(deg))[rot_z], (0, np.sin(deg))[rot_y], 0],
                            [(0, np.sin(deg))[rot_z], (np.cos(deg), 1)[rot_y], (0, -np.sin(deg))[rot_x], 0],
                            [(0, -np.sin(deg))[rot_y], (0, np.sin(deg))[rot_x], (np.cos(deg), 1)[rot_z], 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        rot_mat = np.where(np.abs(rot_mat) < 1e-10, 0, rot_mat)  # get a small value when np.cos(np.pi/2)
        return rot_mat

    def trans(self, axis, dis):
        AXIS = ('X', 'Y', 'Z')
        axis = str(axis).upper()
        if axis not in AXIS:
            print(axis,"is unknown axis, should be one of ",AXIS)
            return
        trans_mat = np.eye(4)
        trans_mat[AXIS.index(axis), 3] = dis
        return trans_mat

    def get_DH(self,joints):
        assert len(joints)==len(self.alpha)
        ans = []
        DOF = len(joints)
        for i in range(DOF):
            tmp = [joints[i], self.alpha[i], self.D[i], self.L[i]]
            ans.append(tmp)
        ans = np.array(ans)
        return ans

    def fk(self, joints):
        # thea_1, thea_2, thea_3, thea_4, thea_5, thea_6 = joints
        # DH_pramater: [link, a, d, thea]，注意这里的单位是m
        DH=self.get_DH(joints)
        T = [self.rotate('z', thea_i).dot(self.trans('z', d_i)).dot(self.trans('x', l_i)).dot(self.rotate('x', a_i))
            for thea_i, d_i, l_i, a_i in DH]
        robot = reduce(np.dot, T)
        return  robot

bias = [0,0,0,0]
def get_Dobot():
    DH_ = [ [-np.pi/2,0,0,np.pi/2],         # alpha
            [0.079,0,0,0],                  # D
            [0.138 + bias[0], 0.135 + bias[1], 0.147 + bias[2], 0.0597 + bias[3]]]     # L
    DH_ = np.array(DH_)
    dobot = FK(DH_)

    return dobot

def load_data(file, is_fk = True, test_data_scale = 0.5):
    with open(file,"r") as rf:
        lines = rf.readlines()
        shuffle(lines)
        p = []
        q = []
        for line in lines:
            datas = line.split(" ")
            p.append([float(x)*100 for x in datas[0:3]])
            q.append([float(x)/180 * np.pi for x in datas[3:]])
        q = np.array(q)
        p = np.array(p)
    if is_fk:
        inputs = []
        for q_ in q:
            tmp_cos = [math.cos(joint) for joint in q_]
            tmp_sin = [math.sin(joint) for joint in q_]
            tmp = q_.tolist()
            # tmp.extend(tmp_cos)
            # tmp.extend(tmp_sin)
            # print(tmp)
            inputs.append(tmp)
        inputs = np.array(inputs)
        outputs = p
    else:
        inputs = p
        outputs = q
    test_set = [inputs[int(inputs.shape[0]*test_data_scale):-1], outputs[int(inputs.shape[0]*test_data_scale):-1]]
    inputs = inputs[0:int(q.shape[0]*test_data_scale)]
    outputs = outputs[0:int(p.shape[0]*test_data_scale)]
    return inputs, outputs, test_set[0], test_set[1]

def noramlization(data):
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
        q, p, _, _ = generate_data(data_nums, q_0, test_data_scale = 1, is_delta = True)
        delta_p = np.array([p_i - p_0 for p_i in p])
        delta_q = np.array([q_i - q_0 for q_i in q])

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

def generate_data(data_nums = 1000, q_e =[0,0,0,0], is_fk = True, test_data_scale = 0.5, is_delta = False):
    q = []
    p = []
    with open("data.txt","w") as wf:
        # pre_joint = np.random.rand(4) * np.pi
        # step = 0.1
        for i in range(data_nums):
            if is_delta:
                joint = np.random.rand(4) * np.pi/20 + q_e
            else:
                joint = np.random.rand(4) * np.pi
            q.append(joint[0:4])
            # print("joint is ", joint)
            dobot = get_Dobot()
            DH_dobot = dobot.fk(joint)
            # print(DH_dobot)
            p.append(DH_dobot[:,-1][0:3] * 100)

            wf.write(str(np.concatenate((joint,DH_dobot[:,-1][0:3]*100))))
            wf.write('\n')
        p = np.array(p)
        q = np.array(q)
    if is_fk:
        inputs = q
        outputs = p
    else:
        inputs = p
        outputs = q
    test_set = [inputs[int(inputs.shape[0]*test_data_scale):-1], 
                outputs[int(inputs.shape[0]*test_data_scale):-1]]
    inputs = inputs[0:int(q.shape[0]*test_data_scale)]
    outputs = outputs[0:int(p.shape[0]*test_data_scale)]
    return inputs, outputs, test_set[0], test_set[1]

def distance(positions_a, positions_b):
    # assert positions_a.shape == positions_b.shape
    dis = [np.sqrt(np.sum(np.square(p_a - p_b))) for p_a, p_b in zip(positions_a, positions_b)]
    dis = np.array(dis)
    mean = np.mean(dis)
    return dis,mean

if __name__ == "__main__":
    joint = [sym.symbols('q_0'),
             sym.symbols('q_1'),
             sym.symbols('q_2'),
             sym.symbols('q_3')]
    dobot = get_Dobot()
    print(dobot.fk(joint))
